//! Audio encoder: a ViT over log-mel spectrograms (AudioMAE / AST layout).
//!
//! The backbone is the shared 2D [`VitBackbone`] with one input channel and a
//! rectangular patch grid (`frames / patch` × `n_mels / patch`); the front-end lives in
//! `media::audio`. Any checkpoint whose tensors map onto the backbone loads here -
//! AudioMAE (timm naming, CLS token, learned positional embedding) is the verified one.

use std::path::Path;

use candle_core::{Device, Tensor};

use crate::engine::vit::VitBackbone;
use crate::engine::{build_backbone, freeze_backbone, load_checkpoint_strict};
use crate::media::audio::{AudioClip, clip_to_spectrogram_tensor, htk_mel_filterbank, periodic_hann};
use crate::types::{AudioFrontend, AudioSpec, JepaError, ModelManifest, WeightReport};

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
        // The manifest geometry is authoritative: rows x columns of the spectrogram,
        // frames x mel bins (Kaldi) or mel bins x frames (torchaudio).
        let (rows, cols) = (manifest.image_size, manifest.input_width);
        if spec.frontend.mel_rows() {
            spec.n_mels = rows;
            spec.frames = cols.unwrap_or(spec.frames);
        } else {
            spec.frames = rows;
            spec.n_mels = cols.unwrap_or(spec.n_mels);
        }
        spec
    }

    pub fn load(manifest: ModelManifest, weights_path: &Path, device: Device) -> Result<Self, JepaError> {
        let (varmap, mut backbone) = build_backbone(&manifest, &device)?;
        let weights = load_checkpoint_strict(&varmap, &mut backbone, weights_path, &device, &manifest.name)?;
        let backbone = freeze_backbone(&manifest, &varmap, &backbone, &device)?;
        let spec = Self::spec_for(&manifest);
        check_frontend_against_checkpoint(&spec, weights_path)?;
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

/// Audio-JEPA exports carry the mel filterbank and STFT window as buffers. When they
/// are present, the front-end we compute must reproduce them: a silently different
/// spectrogram would give plausible but meaningless embeddings.
fn check_frontend_against_checkpoint(spec: &AudioSpec, weights_path: &Path) -> Result<(), JepaError> {
    let AudioFrontend::TorchMelDb { n_fft, win_length, f_min, f_max, .. } = spec.frontend else {
        return Ok(());
    };
    let mmap = unsafe {
        candle_core::safetensors::MmapedSafetensors::new(weights_path)
            .map_err(|e| JepaError::InferenceError(format!("Could not mmap safetensors: {e}")))?
    };
    let max_diff = |name: &str, ours: Vec<f32>| -> Result<Option<f32>, JepaError> {
        let Ok(t) = mmap.load(name, &Device::Cpu) else { return Ok(None) };
        let theirs: Vec<f32> = t.flatten_all()?.to_vec1()?;
        if theirs.len() != ours.len() {
            return Err(JepaError::InferenceError(format!(
                "Checkpoint '{name}' has {} values, the front-end computes {}",
                theirs.len(),
                ours.len()
            )));
        }
        Ok(Some(theirs.iter().zip(&ours).map(|(a, b)| (a - b).abs()).fold(0.0, f32::max)))
    };
    let filters = htk_mel_filterbank(n_fft, spec.n_mels, f_min, f_max, spec.sample_rate);
    let checks = [
        ("adapter.spectrogram.mel_spec.mel_scale.fb", filters.concat()),
        ("adapter.spectrogram.mel_spec.spectrogram.window", periodic_hann(win_length)),
    ];
    for (name, ours) in checks {
        match max_diff(name, ours)? {
            // torchaudio builds the filters in float32 (observed diff 1.4e-5); a wrong
            // parameter (f_max, n_fft, mel scale) moves them by 1e-2 or more.
            Some(d) if d > 1e-4 => {
                return Err(JepaError::InferenceError(format!(
                    "Audio front-end does not reproduce checkpoint '{name}' (max abs diff {d:e})"
                )));
            }
            Some(d) => tracing::info!("Front-end matches checkpoint '{}' (max abs diff {:e})", name, d),
            None => {}
        }
    }
    Ok(())
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
            audio: Some(AudioSpec {
                sample_rate: 16_000,
                n_mels: 32,
                frames: 64,
                mean: 0.0,
                std: 1.0,
                frontend: crate::types::AudioFrontend::KaldiFbank,
            }),
            pooling: None,
            norm_eps: None,
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
