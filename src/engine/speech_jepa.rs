//! Speech-JEPA speech encoder (labhamlet, 2026) in pure Candle.
//!
//! Architecture, as in the released `modeling_speech_jepa.py` (`SpeechJEPAModel`):
//! - Input: the raw 16 kHz waveform, normalised to zero mean and unit variance per
//!   utterance (`Wav2Vec2FeatureExtractor(do_normalize=True)`).
//! - wav2vec 2.0 feature encoder: 7 bias-free `Conv1d` (512 channels, kernels
//!   `[10, 3, 3, 3, 3, 2, 2]`, strides `[5, 2, 2, 2, 2, 2, 2]`), GELU after each, and a
//!   per-channel `GroupNorm(512, 512)` after the first only. One token per 320 samples
//!   (20 ms), receptive field 400 samples.
//! - Feature projection: `LayerNorm(512) -> Linear(512, D) -> LayerNorm(D)` (ε=1e-5).
//! - **Post-norm** Transformer layers `LN(x + attn(x))`, `LN(h + ffn(h))` (ε from the
//!   manifest, 1e-6), GELU, no final norm. Positions are **interleaved-pair RoPE**
//!   (θ = 10000) on queries and keys: there is no positional table and no conv
//!   positional embedding.
//! - The pooled representation is the mean over all frames.
//!
//! The masking embedding (`masked_spec_embed`) is a pre-training artefact and is not
//! loaded.

use std::path::Path;
use std::time::Instant;

use candle_core::{D, DType, Device, Module, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, GroupNorm, LayerNorm, Linear, VarBuilder, VarMap, conv1d_no_bias, group_norm, layer_norm,
    linear,
};

use crate::engine::frozen_builder;
use crate::engine::vit::map_checkpoint_vars;
use crate::media::audio::{AudioClip, resample};
use crate::types::{JepaError, ModelManifest, WeightReport};

/// wav2vec 2.0 base feature encoder: `(channels, kernel, stride)` per layer.
const CONV_LAYERS: [(usize, usize, usize); 7] =
    [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)];
/// Samples covered by one output token (`conv` receptive field).
pub const RECEPTIVE_FIELD: usize = 400;
const ROPE_THETA: f32 = 10_000.0;
/// Longest clip embedded in one pass. Attention is quadratic in the number of 20 ms
/// frames; 30 s is 1499 frames, about 110 MB of scores per layer.
const MAX_SECONDS: usize = 30;

struct ConvLayer {
    conv: Conv1d,
    norm: Option<GroupNorm>,
}

struct EncoderLayer {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    layer_norm: LayerNorm,
    intermediate_dense: Linear,
    output_dense: Linear,
    final_layer_norm: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

/// Rotate interleaved pairs `(x₂ᵢ, x₂ᵢ₊₁)` of `x [B, S, H, Dh]` by `cos/sin [S, Dh/2]`.
fn apply_rotary(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    let (b, s, h, d) = x.dims4()?;
    let pairs = x.reshape((b, s, h, d / 2, 2))?;
    let x0 = pairs.narrow(4, 0, 1)?;
    let x1 = pairs.narrow(4, 1, 1)?;
    let cos = cos.reshape((1, s, 1, d / 2, 1))?;
    let sin = sin.reshape((1, s, 1, d / 2, 1))?;
    let r0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let r1 = (x1.broadcast_mul(&cos)? + x0.broadcast_mul(&sin)?)?;
    Tensor::cat(&[r0, r1], 4)?.reshape((b, s, h, d))
}

impl EncoderLayer {
    fn new(dim: usize, num_heads: usize, hidden: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let attn = vb.pp("attention");
        let ffn = vb.pp("feed_forward");
        Ok(Self {
            q_proj: linear(dim, dim, attn.pp("q_proj"))?,
            k_proj: linear(dim, dim, attn.pp("k_proj"))?,
            v_proj: linear(dim, dim, attn.pp("v_proj"))?,
            out_proj: linear(dim, dim, attn.pp("out_proj"))?,
            layer_norm: layer_norm(dim, eps, vb.pp("layer_norm"))?,
            intermediate_dense: linear(dim, hidden, ffn.pp("intermediate_dense"))?,
            output_dense: linear(hidden, dim, ffn.pp("output_dense"))?,
            final_layer_norm: layer_norm(dim, eps, vb.pp("final_layer_norm"))?,
            num_heads,
            head_dim: dim / num_heads,
        })
    }

    fn attention(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let (b, s, dim) = x.dims3()?;
        let heads = |t: Tensor| t.reshape((b, s, self.num_heads, self.head_dim));
        let q = apply_rotary(&heads(self.q_proj.forward(x)?)?, cos, sin)?.transpose(1, 2)?.contiguous()?;
        let k = apply_rotary(&heads(self.k_proj.forward(x)?)?, cos, sin)?.transpose(1, 2)?.contiguous()?;
        let v = heads(self.v_proj.forward(x)?)?.transpose(1, 2)?.contiguous()?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * scale)?;
        let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let context = attn.matmul(&v)?.transpose(1, 2)?.reshape((b, s, dim))?;
        self.out_proj.forward(&context)
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
        let h = self.layer_norm.forward(&(x + self.attention(x, cos, sin)?)?)?;
        let ff =
            self.output_dense.forward(&candle_nn::Activation::Gelu.forward(&self.intermediate_dense.forward(&h)?)?)?;
        self.final_layer_norm.forward(&(h + ff)?)
    }
}

/// `(pooled embedding, per-frame tokens, latency in ms)`.
pub type EncoderOutput = (Vec<f32>, Vec<Vec<f32>>, f64);

/// Speech-JEPA encoder instance.
pub struct SpeechJepaModel {
    pub manifest: ModelManifest,
    pub device: Device,
    pub weights: WeightReport,
    sample_rate: u32,
    conv_layers: Vec<ConvLayer>,
    fp_norm: LayerNorm,
    fp_proj: Linear,
    fp_post_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
}

impl SpeechJepaModel {
    fn build(manifest: ModelManifest, device: &Device) -> Result<(VarMap, Self), JepaError> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let mut model = Self::build_with(manifest, vb, device)?;
        model.weights.expected = varmap.data().lock().map(|d| d.len()).unwrap_or(0);
        Ok((varmap, model))
    }

    fn build_with(manifest: ModelManifest, vb: VarBuilder, device: &Device) -> Result<Self, JepaError> {
        let dim = manifest.embed_dim;
        if manifest.num_heads == 0 || !dim.is_multiple_of(manifest.num_heads) {
            return Err(JepaError::InvalidPayload("embed_dim must be divisible by num_heads".into()));
        }
        let fe = vb.pp("feature_extractor").pp("conv_layers");
        let mut conv_layers = Vec::with_capacity(CONV_LAYERS.len());
        let mut in_dim = 1;
        for (i, &(out_dim, kernel, stride)) in CONV_LAYERS.iter().enumerate() {
            let cfg = Conv1dConfig { stride, ..Default::default() };
            let conv = conv1d_no_bias(in_dim, out_dim, kernel, cfg, fe.pp(i).pp("conv"))?;
            let norm = if i == 0 { Some(group_norm(out_dim, out_dim, 1e-5, fe.pp(i).pp("layer_norm"))?) } else { None };
            conv_layers.push(ConvLayer { conv, norm });
            in_dim = out_dim;
        }
        let fp = vb.pp("feature_projection");
        let fp_norm = layer_norm(in_dim, 1e-5, fp.pp("layer_norm"))?;
        let fp_proj = linear(in_dim, dim, fp.pp("projection"))?;
        let fp_post_norm = layer_norm(dim, 1e-5, fp.pp("post_layer_norm"))?;
        let hidden = (dim as f64 * manifest.mlp_ratio()) as usize;
        let layers_vb = vb.pp("encoder").pp("layers");
        let layers = (0..manifest.num_layers)
            .map(|i| EncoderLayer::new(dim, manifest.num_heads, hidden, manifest.norm_eps(), layers_vb.pp(i)))
            .collect::<candle_core::Result<Vec<_>>>()?;
        Ok(Self {
            sample_rate: manifest.audio.map_or(16_000, |a| a.sample_rate),
            manifest,
            device: device.clone(),
            weights: WeightReport { loaded: 0, expected: 0, source: "random".into() },
            conv_layers,
            fp_norm,
            fp_proj,
            fp_post_norm,
            layers,
        })
    }

    /// Load from the Hugging Face `SpeechJEPAModel` safetensors file. Fails unless
    /// every encoder parameter is covered.
    pub fn load(manifest: ModelManifest, weights_path: &Path, device: Device) -> Result<Self, JepaError> {
        let name = manifest.name.clone();
        if !weights_path.is_file() {
            return Err(JepaError::ModelNotFound(format!(
                "Weights for '{}' not found at {}. Run `jepctl pull {}` first.",
                name,
                weights_path.display(),
                name
            )));
        }
        let (varmap, model) = Self::build(manifest, &device)?;
        let mmap = unsafe {
            candle_core::safetensors::MmapedSafetensors::new(weights_path)
                .map_err(|e| JepaError::InferenceError(format!("Could not mmap safetensors: {}", e)))?
        };
        let (loaded, missing) = map_checkpoint_vars(&varmap, &mmap, &device)?;
        let expected = model.weights.expected;
        if loaded != expected {
            let preview: Vec<&str> = missing.iter().take(5).map(String::as_str).collect();
            tracing::error!(
                "Checkpoint {} covers {}/{} parameters of '{}'. Missing (first 5): {:?}",
                weights_path.display(),
                loaded,
                expected,
                name,
                preview
            );
            return Err(JepaError::WeightsIncomplete { loaded, expected });
        }
        tracing::info!("Loaded {}/{} tensors into '{}' from {}", loaded, expected, name, weights_path.display());
        // Rebuild on detached weights so that inference records no autograd graph
        // (see `engine::frozen_builder`).
        let mut model = Self::build_with(model.manifest, frozen_builder(&varmap, &device)?, &device)?;
        model.weights = WeightReport { loaded, expected, source: "safetensors".into() };
        Ok(model)
    }

    /// Randomly initialised instance for tests.
    #[doc(hidden)]
    pub fn load_random(manifest: ModelManifest, device: Device) -> Result<Self, JepaError> {
        Ok(Self::build(manifest, &device)?.1)
    }

    /// `cos/sin [S, Dh/2]` for interleaved RoPE at positions `0..S`.
    fn rope_tables(&self, seq_len: usize) -> candle_core::Result<(Tensor, Tensor)> {
        let head_dim = self.manifest.embed_dim / self.manifest.num_heads;
        let half = head_dim / 2;
        let inv_freq: Vec<f32> = (0..half).map(|i| 1.0 / ROPE_THETA.powf((2 * i) as f32 / head_dim as f32)).collect();
        let (mut cos, mut sin) = (Vec::with_capacity(seq_len * half), Vec::with_capacity(seq_len * half));
        for p in 0..seq_len {
            for f in &inv_freq {
                let a = p as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
        Ok((
            Tensor::from_vec(cos, (seq_len, half), &self.device)?,
            Tensor::from_vec(sin, (seq_len, half), &self.device)?,
        ))
    }

    /// Embed a normalised waveform `[1, T]` at the model's sample rate.
    pub fn forward_waveform(&self, wave: &Tensor) -> Result<EncoderOutput, JepaError> {
        let start = Instant::now();
        let mut h = wave.to_device(&self.device)?.unsqueeze(1)?; // [B, 1, T]
        for layer in &self.conv_layers {
            h = layer.conv.forward(&h)?;
            if let Some(norm) = &layer.norm {
                h = norm.forward(&h)?;
            }
            h = candle_nn::Activation::Gelu.forward(&h)?;
        }
        let h = h.transpose(1, 2)?.contiguous()?; // [B, frames, 512]
        let mut h = self.fp_post_norm.forward(&self.fp_proj.forward(&self.fp_norm.forward(&h)?)?)?;
        let (cos, sin) = self.rope_tables(h.dim(1)?)?;
        for layer in &self.layers {
            h = layer.forward(&h, &cos, &sin)?;
        }
        let tokens = h.squeeze(0)?;
        let pooled: Vec<f32> = tokens.mean(0)?.to_vec1()?;
        let frames: Vec<Vec<f32>> = tokens.to_vec2()?;
        Ok((pooled, frames, start.elapsed().as_secs_f64() * 1000.0))
    }

    /// Embed decoded audio: resample, keep at most [`MAX_SECONDS`], normalise per
    /// utterance as `Wav2Vec2FeatureExtractor` does (`(x - mean) / sqrt(var + 1e-7)`).
    pub fn embed_clip(&self, clip: &AudioClip) -> Result<EncoderOutput, JepaError> {
        let clip = resample(clip, self.sample_rate);
        let mut samples = clip.samples;
        samples.truncate(MAX_SECONDS * self.sample_rate as usize);
        if samples.len() < RECEPTIVE_FIELD {
            return Err(JepaError::InvalidPayload(format!(
                "Audio is shorter than {RECEPTIVE_FIELD} samples (one 25 ms frame)"
            )));
        }
        let n = samples.len() as f64;
        let mean = samples.iter().map(|&x| x as f64).sum::<f64>() / n;
        let var = samples.iter().map(|&x| (x as f64 - mean).powi(2)).sum::<f64>() / n;
        let inv = 1.0 / (var + 1e-7).sqrt();
        let normalized: Vec<f32> = samples.iter().map(|&x| ((x as f64 - mean) * inv) as f32).collect();
        let len = normalized.len();
        self.forward_waveform(&Tensor::from_vec(normalized, (1, len), &Device::Cpu)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AudioFrontend, AudioSpec, ModelModality};

    fn manifest() -> ModelManifest {
        ModelManifest {
            name: "test/speech-jepa-tiny".into(),
            repo_id: "test/speech-jepa-tiny".into(),
            architecture: "Speech-JEPA tiny".into(),
            modality: ModelModality::Audio,
            patch_size: 320,
            embed_dim: 16,
            num_layers: 1,
            num_heads: 2,
            image_size: 1,
            frames: None,
            parameter_count: "tiny".into(),
            disk_size_bytes: 0,
            weights_file: "model.safetensors".into(),
            created_at: chrono::Utc::now(),
            variant: Some(crate::engine::vit::VitVariant::SpeechJepa),
            normalization: None,
            mlp_ratio: None,
            tubelet_size: None,
            input_width: None,
            in_chans: Some(1),
            audio: Some(AudioSpec { frontend: AudioFrontend::Waveform, ..AudioSpec::default() }),
            pooling: None,
            norm_eps: None,
        }
    }

    #[test]
    fn one_token_per_320_samples_and_full_parameter_count() {
        let (varmap, m) = SpeechJepaModel::build(manifest(), &Device::Cpu).unwrap();
        // 7 conv kernels + group norm (2) + projection (6) + one layer (16)
        assert_eq!(varmap.data().lock().unwrap().len(), 7 + 2 + 6 + 16);
        let samples: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let (pooled, frames, _) = m.embed_clip(&AudioClip { samples, sample_rate: 16_000 }).unwrap();
        assert_eq!(pooled.len(), 16);
        // HF `_get_feat_extract_output_lengths`: floor((L - k) / s) + 1 per layer.
        let expected = CONV_LAYERS.iter().fold(16_000usize, |l, &(_, k, s)| (l - k) / s + 1);
        assert_eq!(frames.len(), expected);
        assert_eq!(expected, 49);
    }

    #[test]
    fn rotation_matches_reference_formula() {
        // x = [1, 2, 3, 4] at position 1: pairs (1, 2), (3, 4) rotated by θ = 1 and θ = 0.01.
        let x = Tensor::from_vec(vec![0f32, 0., 0., 0., 1., 2., 3., 4.], (1, 2, 1, 4), &Device::Cpu).unwrap();
        let (a, b) = (1f32, 0.01f32);
        let cos = Tensor::from_vec(vec![1f32, 1., a.cos(), b.cos()], (2, 2), &Device::Cpu).unwrap();
        let sin = Tensor::from_vec(vec![0f32, 0., a.sin(), b.sin()], (2, 2), &Device::Cpu).unwrap();
        let r: Vec<f32> = apply_rotary(&x, &cos, &sin).unwrap().flatten_all().unwrap().to_vec1().unwrap();
        let want = [
            1.0 * a.cos() - 2.0 * a.sin(),
            2.0 * a.cos() + 1.0 * a.sin(),
            3.0 * b.cos() - 4.0 * b.sin(),
            4.0 * b.cos() + 3.0 * b.sin(),
        ];
        for (got, want) in r[4..].iter().zip(want) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }
}
