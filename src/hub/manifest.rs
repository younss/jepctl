//! Jepafile manifest schema, built-in catalog definitions, and validation logic.

use std::fs;
use std::path::Path;
use chrono::Utc;
use serde::{Deserialize, Serialize};

use crate::types::{JepaError, ModelManifest, ModelModality};

/// Parsable manifest file schema (.jepa or Jepafile)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JepafileConfig {
    pub name: String,
    pub repo_id: String,
    pub architecture: String,
    pub modality: ModelModality,
    pub patch_size: usize,
    pub embed_dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub image_size: usize,
    pub frames: Option<usize>,
    pub parameter_count: Option<String>,
    pub weights_file: Option<String>,
}

impl JepafileConfig {
    /// Convert to canonical ModelManifest with validation
    pub fn to_manifest(&self, disk_size: u64) -> Result<ModelManifest, JepaError> {
        self.validate()?;

        Ok(ModelManifest {
            name: self.name.clone(),
            repo_id: self.repo_id.clone(),
            architecture: self.architecture.clone(),
            modality: self.modality,
            patch_size: self.patch_size,
            embed_dim: self.embed_dim,
            num_layers: self.num_layers,
            num_heads: self.num_heads,
            image_size: self.image_size,
            frames: self.frames,
            parameter_count: self.parameter_count.clone().unwrap_or_else(|| "Unknown".to_string()),
            disk_size_bytes: disk_size,
            weights_file: self.weights_file.clone().unwrap_or_else(|| "model.safetensors".to_string()),
            created_at: Utc::now(),
        })
    }

    /// Strict schema validation
    pub fn validate(&self) -> Result<(), JepaError> {
        if self.name.trim().is_empty() {
            return Err(JepaError::InvalidPayload("Manifest 'name' must not be empty".to_string()));
        }
        if self.repo_id.trim().is_empty() {
            return Err(JepaError::InvalidPayload("Manifest 'repo_id' must not be empty".to_string()));
        }
        if self.patch_size != 14 && self.patch_size != 16 {
            return Err(JepaError::InvalidPayload(format!(
                "Unsupported patch_size {}. Supported: 14 or 16",
                self.patch_size
            )));
        }
        if self.embed_dim == 0 || self.embed_dim % self.num_heads != 0 {
            return Err(JepaError::InvalidPayload(format!(
                "embed_dim ({}) must be divisible by num_heads ({})",
                self.embed_dim, self.num_heads
            )));
        }
        if self.num_layers == 0 {
            return Err(JepaError::InvalidPayload("num_layers must be greater than 0".to_string()));
        }
        if self.image_size == 0 {
            return Err(JepaError::InvalidPayload("image_size must be greater than 0".to_string()));
        }
        Ok(())
    }

    /// Save manifest to path
    pub fn save_to_file(&self, path: &Path) -> Result<(), JepaError> {
        let json = serde_json::to_string_pretty(self)?;
        fs::write(path, json)?;
        Ok(())
    }

    /// Load manifest from path
    pub fn load_from_file(path: &Path) -> Result<Self, JepaError> {
        let content = fs::read_to_string(path)?;
        let manifest = serde_json::from_str::<Self>(&content)?;
        manifest.validate()?;
        Ok(manifest)
    }
}

/// Retrieve verified built-in catalog manifest specifications
pub fn get_verified_manifests() -> Vec<ModelManifest> {
    vec![
        ModelManifest {
            name: "google/vit-base-patch16-224".to_string(),
            repo_id: "google/vit-base-patch16-224".to_string(),
            architecture: "ViT-B/16".to_string(),
            modality: ModelModality::Image,
            patch_size: 16,
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            image_size: 224,
            frames: None,
            parameter_count: "86M".to_string(),
            disk_size_bytes: 343_000_000,
            weights_file: "model.safetensors".to_string(),
            created_at: Utc::now(),
        },
        ModelManifest {
            name: "google/siglip-base-patch16-224".to_string(),
            repo_id: "google/siglip-base-patch16-224".to_string(),
            architecture: "SigLIP-B/16".to_string(),
            modality: ModelModality::Image,
            patch_size: 16,
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            image_size: 224,
            frames: None,
            parameter_count: "86M".to_string(),
            disk_size_bytes: 343_000_000,
            weights_file: "model.safetensors".to_string(),
            created_at: Utc::now(),
        },
        ModelManifest {
            name: "microsoft/beit-base-patch16-224".to_string(),
            repo_id: "microsoft/beit-base-patch16-224".to_string(),
            architecture: "BEiT-B/16".to_string(),
            modality: ModelModality::Image,
            patch_size: 16,
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            image_size: 224,
            frames: None,
            parameter_count: "86M".to_string(),
            disk_size_bytes: 343_000_000,
            weights_file: "model.safetensors".to_string(),
            created_at: Utc::now(),
        },
        ModelManifest {
            name: "timm/vit_base_patch16_224.augreg_in21k".to_string(),
            repo_id: "timm/vit_base_patch16_224.augreg_in21k".to_string(),
            architecture: "ViT-B/16".to_string(),
            modality: ModelModality::Image,
            patch_size: 16,
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            image_size: 224,
            frames: None,
            parameter_count: "86M".to_string(),
            disk_size_bytes: 343_000_000,
            weights_file: "model.safetensors".to_string(),
            created_at: Utc::now(),
        },
        ModelManifest {
            name: "facebook/dinov2-small".to_string(),
            repo_id: "facebook/dinov2-small".to_string(),
            architecture: "ViT-S/14".to_string(),
            modality: ModelModality::Image,
            patch_size: 14,
            embed_dim: 384,
            num_layers: 12,
            num_heads: 6,
            image_size: 224,
            frames: None,
            parameter_count: "22M".to_string(),
            disk_size_bytes: 88_000_000,
            weights_file: "model.safetensors".to_string(),
            created_at: Utc::now(),
        },
        ModelManifest {
            name: "facebook/dinov2-base".to_string(),
            repo_id: "facebook/dinov2-base".to_string(),
            architecture: "ViT-B/14".to_string(),
            modality: ModelModality::Image,
            patch_size: 14,
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            image_size: 224,
            frames: None,
            parameter_count: "86M".to_string(),
            disk_size_bytes: 343_000_000,
            weights_file: "model.safetensors".to_string(),
            created_at: Utc::now(),
        },
        ModelManifest {
            name: "facebook/ijepa_vitb16_1k".to_string(),
            repo_id: "facebook/ijepa_vitb16_1k".to_string(),
            architecture: "ViT-B/16".to_string(),
            modality: ModelModality::Image,
            patch_size: 16,
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            image_size: 224,
            frames: None,
            parameter_count: "86M".to_string(),
            disk_size_bytes: 344_000_000,
            weights_file: "model.safetensors".to_string(),
            created_at: Utc::now(),
        },
        ModelManifest {
            name: "facebook/ijepa_vith14_1k".to_string(),
            repo_id: "facebook/ijepa_vith14_1k".to_string(),
            architecture: "ViT-H/14".to_string(),
            modality: ModelModality::Image,
            patch_size: 14,
            embed_dim: 1280,
            num_layers: 32,
            num_heads: 16,
            image_size: 224,
            frames: None,
            parameter_count: "632M".to_string(),
            disk_size_bytes: 2_528_000_000,
            weights_file: "model.safetensors".to_string(),
            created_at: Utc::now(),
        },
        ModelManifest {
            name: "facebookresearch/jepa:vjepa_vitl16".to_string(),
            repo_id: "facebookresearch/jepa".to_string(),
            architecture: "ViT-L/16".to_string(),
            modality: ModelModality::Video,
            patch_size: 16,
            embed_dim: 1024,
            num_layers: 24,
            num_heads: 16,
            image_size: 224,
            frames: Some(16),
            parameter_count: "307M".to_string(),
            disk_size_bytes: 1_228_000_000,
            weights_file: "model.safetensors".to_string(),
            created_at: Utc::now(),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_verified_manifests_valid() {
        let manifests = get_verified_manifests();
        assert!(!manifests.is_empty());
        assert!(manifests.len() >= 8);

        // Verify multi-organization catalog coverage
        let has_google = manifests.iter().any(|m| m.name.starts_with("google/"));
        let has_microsoft = manifests.iter().any(|m| m.name.starts_with("microsoft/"));
        let has_timm = manifests.iter().any(|m| m.name.starts_with("timm/"));
        let has_facebook = manifests.iter().any(|m| m.name.starts_with("facebook"));

        assert!(has_google, "Catalog must include Google models");
        assert!(has_microsoft, "Catalog must include Microsoft models");
        assert!(has_timm, "Catalog must include TIMM models");
        assert!(has_facebook, "Catalog must include Facebook/Meta models");

        for m in manifests {
            assert!(m.patch_size == 14 || m.patch_size == 16);
            assert!(m.embed_dim % m.num_heads == 0);
            assert!(m.num_layers > 0);
            assert!(m.image_size > 0);
            assert!(!m.weights_file.is_empty());
        }
    }
}
