//! I-JEPA 2D Image Encoder and Patch Representation Extractor.

use std::path::Path;
use std::time::Instant;
use candle_core::{Device, Tensor};

use crate::engine::vit::VitBackbone;
use crate::engine::{build_backbone, load_checkpoint_strict};
use crate::types::{JepaError, ModelManifest, WeightReport};

/// I-JEPA image encoder instance
pub struct IJepaModel {
    pub manifest: ModelManifest,
    pub backbone: VitBackbone,
    pub device: Device,
    pub weights: WeightReport,
}

impl IJepaModel {
    /// Load an I-JEPA model from a safetensors checkpoint. Fails unless every
    /// backbone parameter is covered by the checkpoint.
    pub fn load(manifest: ModelManifest, weights_path: &Path, device: Device) -> Result<Self, JepaError> {
        let (varmap, mut backbone) = build_backbone(&manifest, &device)?;
        let weights = load_checkpoint_strict(&varmap, &mut backbone, weights_path, &device, &manifest.name)?;
        Ok(Self {
            manifest,
            backbone,
            device,
            weights,
        })
    }

    /// Build the architecture with random weights. Only useful for tests and
    /// benchmarks: embeddings from such a model carry no semantic meaning.
    #[doc(hidden)]
    pub fn load_random(manifest: ModelManifest, device: Device) -> Result<Self, JepaError> {
        let (_varmap, backbone) = build_backbone(&manifest, &device)?;
        let expected = _varmap.data().lock().map(|d| d.len()).unwrap_or(0);
        Ok(Self {
            manifest,
            backbone,
            device,
            weights: WeightReport { loaded: 0, expected, source: "random".into() },
        })
    }

    /// Forward pass computing global pooled representation and spatial patch representations
    pub fn forward_image(&self, img: &Tensor) -> Result<(Vec<f32>, Vec<Vec<f32>>, f64), JepaError> {
        let start = Instant::now();
        let img = img.to_device(&self.device)?;

        let (patch_tokens, pooled) = self.backbone.forward(&img)?;

        // Extract global pooled vector: [1, D] -> Vec<f32>
        let pooled = pooled.squeeze(0)?;
        let pooled_vec: Vec<f32> = pooled.to_vec1()?;

        // Extract spatial patch tokens: [1, N, D] -> Vec<Vec<f32>>
        let (num_patches, dim) = patch_tokens.squeeze(0)?.dims2()?;
        let flattened: Vec<f32> = patch_tokens.flatten_all()?.to_vec1()?;
        let mut patch_vecs = Vec::with_capacity(num_patches);
        for i in 0..num_patches {
            let start_idx = i * dim;
            let end_idx = start_idx + dim;
            patch_vecs.push(flattened[start_idx..end_idx].to_vec());
        }

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok((pooled_vec, patch_vecs, latency_ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::DType;
    use crate::types::ModelModality;

    #[test]
    fn test_load_refuses_missing_checkpoint() {
        let manifest = ModelManifest {
            name: "test/ijepa".to_string(),
            repo_id: "test/repo".to_string(),
            architecture: "vit".to_string(),
            modality: ModelModality::Image,
            patch_size: 16,
            embed_dim: 64,
            num_layers: 1,
            num_heads: 4,
            image_size: 224,
            frames: None,
            parameter_count: "1M".to_string(),
            disk_size_bytes: 0,
            weights_file: "model.safetensors".to_string(),
            created_at: chrono::Utc::now(),
        };
        let err = IJepaModel::load(manifest, Path::new("/nonexistent/model.safetensors"), Device::Cpu)
            .err()
            .expect("must fail without a checkpoint");
        assert!(matches!(err, JepaError::ModelNotFound(_)), "{err}");
    }

    #[test]
    fn test_ijepa_embedding_non_zero_and_similarity() {
        let manifest = ModelManifest {
            name: "test/ijepa".to_string(),
            repo_id: "test/repo".to_string(),
            architecture: "vit".to_string(),
            modality: ModelModality::Image,
            patch_size: 16,
            embed_dim: 64,
            num_layers: 2,
            num_heads: 4,
            image_size: 224,
            frames: None,
            parameter_count: "1M".to_string(),
            disk_size_bytes: 1024,
            weights_file: "model.safetensors".to_string(),
            created_at: chrono::Utc::now(),
        };

        let device = Device::Cpu;
        let model = IJepaModel::load_random(manifest, device).expect("Failed to build model");
        assert_eq!(model.weights.source, "random");

        let img1 = Tensor::ones((1, 3, 224, 224), DType::F32, &Device::Cpu).unwrap();
        let (pooled1, _patches1, _lat1) = model.forward_image(&img1).expect("forward img1 failed");

        let norm_sq: f32 = pooled1.iter().map(|x| x * x).sum();
        assert!(norm_sq > 0.1, "Embedding should be non-zero, but got norm_sq = {}", norm_sq);

        let norm1 = crate::types::normalize_l2(&pooled1);
        let sim_self = crate::types::dot_product(&norm1, &norm1);
        assert!((sim_self - 1.0).abs() < 1e-4, "Self similarity should be 1.0, got {}", sim_self);

        let img2 = Tensor::zeros((1, 3, 224, 224), DType::F32, &Device::Cpu).unwrap();
        let (pooled2, _patches2, _lat2) = model.forward_image(&img2).expect("forward img2 failed");
        let norm2 = crate::types::normalize_l2(&pooled2);
        let sim_diff = crate::types::dot_product(&norm1, &norm2);

        assert!(sim_diff < 0.999, "Different images should have different embeddings, got sim = {}", sim_diff);
    }
}
