//! Jepafile manifest schema, built-in catalog definitions, and validation logic.

use chrono::Utc;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::engine::vit::{Pooling, VitVariant};
use crate::types::{AudioSpec, JepaError, ModelManifest, ModelModality, Normalization};

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
    #[serde(default)]
    pub variant: Option<VitVariant>,
    #[serde(default)]
    pub normalization: Option<Normalization>,
    #[serde(default)]
    pub mlp_ratio: Option<f64>,
    #[serde(default)]
    pub tubelet_size: Option<usize>,
    #[serde(default)]
    pub input_width: Option<usize>,
    #[serde(default)]
    pub in_chans: Option<usize>,
    #[serde(default)]
    pub audio: Option<AudioSpec>,
    #[serde(default)]
    pub pooling: Option<Pooling>,
}

impl From<&ModelManifest> for JepafileConfig {
    fn from(m: &ModelManifest) -> Self {
        Self {
            name: m.name.clone(),
            repo_id: m.repo_id.clone(),
            architecture: m.architecture.clone(),
            modality: m.modality,
            patch_size: m.patch_size,
            embed_dim: m.embed_dim,
            num_layers: m.num_layers,
            num_heads: m.num_heads,
            image_size: m.image_size,
            frames: m.frames,
            parameter_count: Some(m.parameter_count.clone()),
            weights_file: Some(m.weights_file.clone()),
            variant: m.variant,
            normalization: m.normalization,
            mlp_ratio: m.mlp_ratio,
            tubelet_size: m.tubelet_size,
            input_width: m.input_width,
            in_chans: m.in_chans,
            audio: m.audio,
            pooling: m.pooling,
        }
    }
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
            variant: self.variant,
            normalization: self.normalization,
            mlp_ratio: self.mlp_ratio,
            tubelet_size: self.tubelet_size,
            input_width: self.input_width,
            in_chans: self.in_chans,
            audio: self.audio,
            pooling: self.pooling,
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
        if let Some(w) = self.input_width {
            if !w.is_multiple_of(self.patch_size) {
                return Err(JepaError::InvalidPayload(format!(
                    "input_width ({}) must be a multiple of patch_size ({})",
                    w, self.patch_size
                )));
            }
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(JepaError::InvalidPayload(format!(
                "image_size ({}) must be a multiple of patch_size ({})",
                self.image_size, self.patch_size
            )));
        }
        if self.embed_dim == 0 || self.num_heads == 0 || !self.embed_dim.is_multiple_of(self.num_heads) {
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

/// A verified catalog entry: a Hugging Face repo whose `model.safetensors` is known
/// to map 100% onto our backbone. Anything not listed here can still be used via a
/// custom `Jepafile.json`, but is not promised to load.
struct Verified {
    name: &'static str,
    modality: ModelModality,
    frames: Option<usize>,
    tubelet_size: Option<usize>,
    input_width: Option<usize>,
    in_chans: Option<usize>,
    audio: Option<AudioSpec>,
    pooling: Pooling,
    architecture: &'static str,
    patch_size: usize,
    embed_dim: usize,
    num_layers: usize,
    num_heads: usize,
    image_size: usize,
    parameter_count: &'static str,
    disk_size_bytes: u64,
    variant: VitVariant,
    normalization: Normalization,
    mlp_ratio: f64,
}

const VERIFIED: &[Verified] = &[
    Verified {
        name: "facebook/vjepa2-vitl-fpc64-256",
        modality: ModelModality::Video,
        frames: Some(16),
        tubelet_size: Some(2),
        input_width: None,
        in_chans: None,
        audio: None,
        pooling: Pooling::Mean,
        architecture: "V-JEPA 2 ViT-L/16 (video, RoPE)",
        patch_size: 16,
        embed_dim: 1024,
        num_layers: 24,
        num_heads: 16,
        image_size: 256,
        parameter_count: "300M (encoder)",
        disk_size_bytes: 1_303_947_864,
        variant: VitVariant::VJepa2,
        normalization: Normalization::ImageNet,
        mlp_ratio: 4.0,
    },
    Verified {
        name: "gaunernst/vit_base_patch16_1024_128.audiomae_as2m",
        modality: ModelModality::Audio,
        frames: None,
        tubelet_size: None,
        input_width: Some(128),
        in_chans: Some(1),
        audio: Some(AudioSpec { sample_rate: 16_000, n_mels: 128, frames: 1024, mean: -4.2677393, std: 4.5689974 }),
        pooling: Pooling::Mean,
        architecture: "AudioMAE ViT-B/16 (log-mel 1024x128, AudioSet-2M)",
        patch_size: 16,
        embed_dim: 768,
        num_layers: 12,
        num_heads: 12,
        image_size: 1024,
        parameter_count: "86M",
        disk_size_bytes: 342_606_472,
        variant: VitVariant::Cls,
        normalization: Normalization::ImageNet,
        mlp_ratio: 4.0,
    },
    Verified {
        name: "facebook/ijepa_vith14_1k",
        modality: ModelModality::Image,
        frames: None,
        tubelet_size: None,
        input_width: None,
        in_chans: None,
        audio: None,
        pooling: Pooling::Mean,
        architecture: "I-JEPA ViT-H/14 (ImageNet-1k)",
        patch_size: 14,
        embed_dim: 1280,
        num_layers: 32,
        num_heads: 16,
        image_size: 224,
        parameter_count: "632M",
        disk_size_bytes: 2_523_108_984,
        variant: VitVariant::Plain,
        normalization: Normalization::ImageNet,
        mlp_ratio: 4.0,
    },
    Verified {
        name: "facebook/ijepa_vith14_22k",
        modality: ModelModality::Image,
        frames: None,
        tubelet_size: None,
        input_width: None,
        in_chans: None,
        audio: None,
        pooling: Pooling::Mean,
        architecture: "I-JEPA ViT-H/14 (ImageNet-22k)",
        patch_size: 14,
        embed_dim: 1280,
        num_layers: 32,
        num_heads: 16,
        image_size: 224,
        parameter_count: "632M",
        disk_size_bytes: 2_523_108_984,
        variant: VitVariant::Plain,
        normalization: Normalization::ImageNet,
        mlp_ratio: 4.0,
    },
    Verified {
        name: "facebook/dinov2-small",
        modality: ModelModality::Image,
        frames: None,
        tubelet_size: None,
        input_width: None,
        in_chans: None,
        audio: None,
        pooling: Pooling::Cls,
        architecture: "DINOv2 ViT-S/14",
        patch_size: 14,
        embed_dim: 384,
        num_layers: 12,
        num_heads: 6,
        image_size: 224,
        parameter_count: "22M",
        disk_size_bytes: 88_249_960,
        variant: VitVariant::DinoV2,
        normalization: Normalization::ImageNet,
        mlp_ratio: 4.0,
    },
    Verified {
        name: "facebook/dinov2-base",
        modality: ModelModality::Image,
        frames: None,
        tubelet_size: None,
        input_width: None,
        in_chans: None,
        audio: None,
        pooling: Pooling::Cls,
        architecture: "DINOv2 ViT-B/14",
        patch_size: 14,
        embed_dim: 768,
        num_layers: 12,
        num_heads: 12,
        image_size: 224,
        parameter_count: "86M",
        disk_size_bytes: 346_345_912,
        variant: VitVariant::DinoV2,
        normalization: Normalization::ImageNet,
        mlp_ratio: 4.0,
    },
    Verified {
        name: "google/vit-base-patch16-224",
        modality: ModelModality::Image,
        frames: None,
        tubelet_size: None,
        input_width: None,
        in_chans: None,
        audio: None,
        pooling: Pooling::Cls,
        architecture: "ViT-B/16 (ImageNet-21k+1k)",
        patch_size: 16,
        embed_dim: 768,
        num_layers: 12,
        num_heads: 12,
        image_size: 224,
        parameter_count: "86M",
        disk_size_bytes: 346_293_852,
        variant: VitVariant::Cls,
        normalization: Normalization::Inception,
        mlp_ratio: 4.0,
    },
    Verified {
        name: "timm/vit_base_patch16_224.augreg_in21k",
        modality: ModelModality::Image,
        frames: None,
        tubelet_size: None,
        input_width: None,
        in_chans: None,
        audio: None,
        pooling: Pooling::Cls,
        architecture: "ViT-B/16 AugReg (ImageNet-21k)",
        patch_size: 16,
        embed_dim: 768,
        num_layers: 12,
        num_heads: 12,
        image_size: 224,
        parameter_count: "86M",
        disk_size_bytes: 410_397_786,
        variant: VitVariant::Cls,
        normalization: Normalization::Inception,
        mlp_ratio: 4.0,
    },
];

/// Retrieve verified built-in catalog manifest specifications
pub fn get_verified_manifests() -> Vec<ModelManifest> {
    VERIFIED
        .iter()
        .map(|v| ModelManifest {
            name: v.name.to_string(),
            repo_id: v.name.to_string(),
            architecture: v.architecture.to_string(),
            modality: v.modality,
            patch_size: v.patch_size,
            embed_dim: v.embed_dim,
            num_layers: v.num_layers,
            num_heads: v.num_heads,
            image_size: v.image_size,
            frames: v.frames,
            parameter_count: v.parameter_count.to_string(),
            disk_size_bytes: v.disk_size_bytes,
            weights_file: "model.safetensors".to_string(),
            created_at: Utc::now(),
            variant: Some(v.variant),
            normalization: Some(v.normalization),
            mlp_ratio: Some(v.mlp_ratio),
            tubelet_size: v.tubelet_size,
            input_width: v.input_width,
            in_chans: v.in_chans,
            audio: v.audio,
            pooling: Some(v.pooling),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_verified_manifests_valid() {
        let manifests = get_verified_manifests();
        assert!(!manifests.is_empty());
        for m in &manifests {
            JepafileConfig::from(m).validate().unwrap_or_else(|e| panic!("{}: {e}", m.name));
            assert!(m.variant.is_some() && m.normalization.is_some(), "{} must be explicit", m.name);
            assert_eq!(m.backbone_variant(), m.variant.unwrap());
        }
        let names: std::collections::HashSet<_> = manifests.iter().map(|m| m.name.as_str()).collect();
        assert_eq!(names.len(), manifests.len(), "duplicate catalog entry");
    }

    #[test]
    fn jepafile_roundtrip_keeps_optional_fields() {
        let m = &get_verified_manifests()[0];
        let cfg = JepafileConfig::from(m);
        let json = serde_json::to_string(&cfg).unwrap();
        let back: JepafileConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(back.variant, m.variant);
        assert_eq!(back.normalization, m.normalization);
        // Old Jepafiles without the new fields still parse.
        let legacy: JepafileConfig = serde_json::from_str(r#"{"name":"x/y","repo_id":"x/y","architecture":"vit","modality":"image","patch_size":16,"embed_dim":64,"num_layers":1,"num_heads":4,"image_size":224,"frames":null,"parameter_count":null,"weights_file":null}"#).unwrap();
        assert!(legacy.variant.is_none());
    }
}
