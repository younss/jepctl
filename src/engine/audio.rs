//! Audio encoder: a ViT over log-mel spectrograms (AudioMAE / AST layout).
//!
//! The backbone is the shared 2D [`VitBackbone`] with one input channel and a
//! rectangular patch grid (`frames / patch` × `n_mels / patch`); the front-end lives in
//! `media::audio`. Any checkpoint whose tensors map onto the backbone loads here -
//! AudioMAE (timm naming, CLS token, learned positional embedding) is the verified one.

use std::path::Path;

use candle_core::{Device, Tensor};

use crate::engine::vit::VitBackbone;
use crate::engine::{build_backbone, load_checkpoint_strict};
use crate::media::audio::{clip_to_spectrogram_tensor, AudioClip};
use crate::types::{AudioSpec, JepaError, ModelManifest, WeightReport};

/// `(pooled embedding, patch tokens, latency in ms)`.
pub type EncoderOutput = (Vec<f32>, Vec<Vec<f32>>, f64);

pub struct AudioModel {
    pub manifest: ModelManifest,
    pub backbone: VitBackbone,
    pub device: Device,
    pub weights: WeightReport,
    pub spec: AudioSpec,
}

impl AudioModel {
    fn spec_for(manifest: &ModelManifest) -> AudioSpec {
        let mut spec = manifest.audio.unwrap_or_default();
        // The manifest geometry is authoritative: rows = frames, columns = mel bins.
        spec.frames = manifest.image_size;
        spec.n_mels = manifest.input_width.unwrap_or(spec.n_mels);
        spec
    }

    pub fn load(manifest: ModelManifest, weights_path: &Path, device: Device) -> Result<Self, JepaError> {
        let (varmap, mut backbone) = build_backbone(&manifest, &device)?;
        let weights = load_checkpoint_strict(&varmap, &mut backbone, weights_path, &device, &manifest.name)?;
        let spec = Self::spec_for(&manifest);
        Ok(Self { manifest, backbone, device, weights, spec })
    }

    #[doc(hidden)]
    pub fn load_random(manifest: ModelManifest, device: Device) -> Result<Self, JepaError> {
        let (varmap, backbone) = build_backbone(&manifest, &device)?;
        let expected = varmap.data().lock().map(|d| d.len()).unwrap_or(0);
        let spec = Self::spec_for(&manifest);
        Ok(Self {
            manifest,
            backbone,
            device,
            weights: WeightReport { loaded: 0, expected, source: "random".into() },
            spec,
        })
    }

    /// Embed a spectrogram tensor `[1, 1, frames, n_mels]`.
    pub fn forward_spectrogram(&self, spec: &Tensor) -> Result<EncoderOutput, JepaError> {
        let start = std::time::Instant::now();
        let x = spec.to_device(&self.device)?;
        let (patch_tokens, pooled) = self.backbone.forward(&x)?;
        let pooled: Vec<f32> = pooled.squeeze(0)?.to_vec1()?;
        let patches: Vec<Vec<f32>> = patch_tokens.squeeze(0)?.to_vec2()?;
        Ok((pooled, patches, start.elapsed().as_secs_f64() * 1000.0))
    }

    /// Embed decoded audio (any sample rate; resampled by the front-end).
    pub fn embed_clip(&self, clip: &AudioClip) -> Result<EncoderOutput, JepaError> {
        let t = clip_to_spectrogram_tensor(clip, &self.spec, &self.device)?;
        self.forward_spectrogram(&t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelModality;

    fn manifest() -> ModelManifest {
        ModelManifest {
            name: "test/audio-tiny".into(),
            repo_id: "test/audio-tiny".into(),
            architecture: "AudioMAE tiny".into(),
            modality: ModelModality::Audio,
            patch_size: 16,
            embed_dim: 32,
            num_layers: 1,
            num_heads: 4,
            image_size: 64,
            frames: None,
            parameter_count: "tiny".into(),
            disk_size_bytes: 0,
            weights_file: "model.safetensors".into(),
            created_at: chrono::Utc::now(),
            variant: Some(crate::engine::vit::VitVariant::Cls),
            normalization: None,
            mlp_ratio: None,
            tubelet_size: None,
            input_width: Some(32),
            in_chans: Some(1),
            audio: Some(AudioSpec { sample_rate: 16_000, n_mels: 32, frames: 64, mean: 0.0, std: 1.0 }),
            pooling: None,
        }
    }

    #[test]
    fn audio_model_embeds_a_clip() {
        let m = AudioModel::load_random(manifest(), Device::Cpu).unwrap();
        assert_eq!((m.spec.frames, m.spec.n_mels), (64, 32));
        let samples: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.3).sin() * 0.3).collect();
        let clip = AudioClip { samples, sample_rate: 16_000 };
        let (pooled, patches, _) = m.embed_clip(&clip).unwrap();
        assert_eq!(pooled.len(), 32);
        assert_eq!(patches.len(), (64 / 16) * (32 / 16));
    }
}
