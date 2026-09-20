//! V-JEPA Spatio-Temporal Video Encoder for representation learning over [B, C, T, H, W].

use std::path::Path;
use std::time::Instant;
use candle_core::{Device, Module, Tensor};

use crate::engine::vit::VitBackbone;
use crate::engine::{build_backbone, load_checkpoint_strict};
use crate::types::{JepaError, ModelManifest, WeightReport};

/// V-JEPA video encoder instance
pub struct VJepaModel {
    pub manifest: ModelManifest,
    pub backbone: VitBackbone,
    pub device: Device,
    pub temporal_frames: usize,
    pub weights: WeightReport,
}

impl VJepaModel {
    /// Load a V-JEPA model from a safetensors checkpoint. Fails unless every
    /// backbone parameter is covered by the checkpoint.
    pub fn load(manifest: ModelManifest, weights_path: &Path, device: Device) -> Result<Self, JepaError> {
        let temporal_frames = manifest.frames.unwrap_or(crate::config::VJEPA_TEMPORAL_FRAMES);
        let (varmap, mut backbone) = build_backbone(&manifest, &device)?;
        let weights = load_checkpoint_strict(&varmap, &mut backbone, weights_path, &device, &manifest.name)?;
        Ok(Self {
            manifest,
            backbone,
            device,
            temporal_frames,
            weights,
        })
    }

    /// Forward pass over spatio-temporal video tensor [B, C, T, H, W]
    /// Produces a pooled latent representation across time and space
    pub fn forward_video(&self, video_tensor: &Tensor) -> Result<(Vec<f32>, f64), JepaError> {
        let start = Instant::now();
        let video = video_tensor.to_device(&self.device)?;

        // Shape: [B, C, T, H, W]
        let (b, c, t, h, w) = video.dims5()?;

        // Permute to [B, T, C, H, W] then reshape to [B * T, C, H, W] for frame-level patch extraction
        let permuted = video.transpose(1, 2)?.contiguous()?; // [B, T, C, H, W]
        let flat_frames = permuted.reshape((b * t, c, h, w))?;

        // Extract spatial patch tokens: [B * T, num_patches, D]
        let mut spatial_tokens = self.backbone.patch_embed.forward(&flat_frames)?;
        // Every frame shares the same 2D positional embedding (spatial position only).
        if let Some(pos) = &self.backbone.pos_embed {
            spatial_tokens = spatial_tokens.broadcast_add(pos)?;
        }
        let num_patches = spatial_tokens.dim(1)?;
        let d = spatial_tokens.dim(2)?;

        // Reshape back to spatio-temporal tokens: [B, T * num_patches, D]
        let st_tokens = spatial_tokens.reshape((b, t * num_patches, d))?;

        // Pass through Transformer blocks
        let mut tokens = st_tokens;
        for block in &self.backbone.blocks {
            tokens = block.forward(&tokens)?;
        }

        // LayerNorm and Global Average Pooling across all spatio-temporal tokens
        let normalized = self.backbone.norm.forward(&tokens)?;
        let pooled = normalized.mean(1)?.squeeze(0)?; // [D]

        let embedding: Vec<f32> = pooled.to_vec1()?;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        Ok((embedding, latency_ms))
    }
}
