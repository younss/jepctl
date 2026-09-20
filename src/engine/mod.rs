//! Engine abstraction traits and runtime model execution manager.

pub mod device;
pub mod ijepa;
pub mod vit;
pub mod vjepa;

use std::path::Path;
use std::sync::Arc;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use tokio::sync::RwLock;

use crate::engine::ijepa::IJepaModel;
use crate::engine::vit::{load_safetensors_into_backbone, VitBackbone};
use crate::engine::vjepa::VJepaModel;
use crate::types::{HardwareInfo, JepaError, ModelManifest, ModelModality, WeightReport};

/// MLP hidden size ratio shared by every supported ViT variant.
const MLP_RATIO: f64 = 4.0;

/// Instantiate a randomly initialised backbone matching a manifest.
pub(crate) fn build_backbone(manifest: &ModelManifest, device: &Device) -> Result<(VarMap, VitBackbone), JepaError> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let backbone = VitBackbone::new(
        manifest.image_size,
        manifest.patch_size,
        3,
        manifest.embed_dim,
        manifest.num_layers,
        manifest.num_heads,
        MLP_RATIO,
        vb,
    )?;
    Ok((varmap, backbone))
}

/// Load a checkpoint into a backbone and refuse anything less than full coverage.
///
/// A partially loaded ViT is indistinguishable from a working one at the API level
/// (it still returns vectors of the right size), so the only safe behaviour is to
/// fail loudly and tell the user which parameters were not found.
pub(crate) fn load_checkpoint_strict(
    varmap: &VarMap,
    backbone: &mut VitBackbone,
    path: &Path,
    device: &Device,
    model_name: &str,
) -> Result<WeightReport, JepaError> {
    if !path.is_file() {
        return Err(JepaError::ModelNotFound(format!(
            "Weights for '{}' not found at {}. Run `jepa pull {}` first.",
            model_name,
            path.display(),
            model_name
        )));
    }

    let outcome = load_safetensors_into_backbone(varmap, backbone, path, device)?;
    if outcome.loaded != outcome.expected {
        let preview: Vec<&str> = outcome.missing.iter().take(5).map(String::as_str).collect();
        tracing::error!(
            "Checkpoint {} covers {}/{} parameters of '{}'. Missing (first 5): {:?}. This checkpoint layout is not supported.",
            path.display(),
            outcome.loaded,
            outcome.expected,
            model_name,
            preview
        );
        return Err(JepaError::WeightsIncomplete {
            loaded: outcome.loaded,
            expected: outcome.expected,
        });
    }
    if !outcome.pos_embed_loaded {
        tracing::info!("'{}' uses fixed 2D sin-cos positional embeddings (none found in checkpoint).", model_name);
    }
    tracing::info!("Loaded {}/{} tensors into '{}' from {}", outcome.loaded, outcome.expected, model_name, path.display());
    Ok(WeightReport {
        loaded: outcome.loaded,
        expected: outcome.expected,
        source: "safetensors".to_string(),
    })
}

/// Common trait for all JEPA model variants (I-JEPA, V-JEPA, Audio-JEPA)
pub trait JepaModelTrait: Send + Sync {
    fn name(&self) -> &str;
    fn modality(&self) -> ModelModality;
    fn dimension(&self) -> usize;
    fn weight_report(&self) -> WeightReport;
    fn embed_image(&self, img: &Tensor) -> Result<(Vec<f32>, Option<Vec<Vec<f32>>>, f64), JepaError>;
    fn embed_video(&self, vid: &Tensor) -> Result<(Vec<f32>, f64), JepaError>;
}

impl JepaModelTrait for IJepaModel {
    fn name(&self) -> &str {
        &self.manifest.name
    }

    fn modality(&self) -> ModelModality {
        ModelModality::Image
    }

    fn dimension(&self) -> usize {
        self.manifest.embed_dim
    }

    fn weight_report(&self) -> WeightReport {
        self.weights.clone()
    }

    fn embed_image(&self, img: &Tensor) -> Result<(Vec<f32>, Option<Vec<Vec<f32>>>, f64), JepaError> {
        let (pooled, patches, latency) = self.forward_image(img)?;
        Ok((pooled, Some(patches), latency))
    }

    fn embed_video(&self, vid: &Tensor) -> Result<(Vec<f32>, f64), JepaError> {
        // For video tensors [1, 3, T, H, W], extract the latest temporal frame [1, 3, H, W]
        let (_b, _c, t, _h, _w) = vid.dims5()?;
        let latest_frame = vid.narrow(2, t - 1, 1)?.squeeze(2)?;
        let (pooled, _patches, latency) = self.forward_image(&latest_frame)?;
        Ok((pooled, latency))
    }
}

impl JepaModelTrait for VJepaModel {
    fn name(&self) -> &str {
        &self.manifest.name
    }

    fn modality(&self) -> ModelModality {
        ModelModality::Video
    }

    fn dimension(&self) -> usize {
        self.manifest.embed_dim
    }

    fn weight_report(&self) -> WeightReport {
        self.weights.clone()
    }

    fn embed_image(&self, img: &Tensor) -> Result<(Vec<f32>, Option<Vec<Vec<f32>>>, f64), JepaError> {
        // Broadcast single image across temporal frames for representation extraction
        let (_b, _c, _h, _w) = img.dims4()?;
        let repeated = img.unsqueeze(2)?.repeat((1, 1, self.temporal_frames, 1, 1))?;
        let (emb, latency) = self.forward_video(&repeated)?;
        Ok((emb, None, latency))
    }

    fn embed_video(&self, vid: &Tensor) -> Result<(Vec<f32>, f64), JepaError> {
        self.forward_video(vid)
    }
}

/// Thread-safe active model executor and memory supervisor
pub struct EngineManager {
    active_model: RwLock<Option<Box<dyn JepaModelTrait>>>,
    pub device: Device,
    pub hardware_info: RwLock<HardwareInfo>,
}

impl EngineManager {
    pub fn new(device: Device, hardware_info: HardwareInfo) -> Arc<Self> {
        Arc::new(Self {
            active_model: RwLock::new(None),
            device,
            hardware_info: RwLock::new(hardware_info),
        })
    }

    /// Load a model into active GPU/system memory. `weights_path` must point to a
    /// downloaded safetensors file; a missing or unsupported checkpoint is an error.
    pub async fn load_model(&self, manifest: ModelManifest, weights_path: Option<&Path>) -> Result<WeightReport, JepaError> {
        let name = manifest.name.clone();
        let path = weights_path.ok_or_else(|| {
            JepaError::ModelNotFound(format!("'{}' is not downloaded. Run `jepa pull {}` first.", name, name))
        })?;

        // Model construction is CPU-heavy (mmap + copies); keep it off the async executor.
        let device = self.device.clone();
        let path = path.to_path_buf();
        let model_box: Box<dyn JepaModelTrait> = tokio::task::spawn_blocking(move || -> Result<Box<dyn JepaModelTrait>, JepaError> {
            Ok(match manifest.modality {
                ModelModality::Image => Box::new(IJepaModel::load(manifest, &path, device)?),
                ModelModality::Video | ModelModality::Multimodal | ModelModality::Audio => {
                    Box::new(VJepaModel::load(manifest, &path, device)?)
                }
            })
        })
        .await
        .map_err(|e| JepaError::InferenceError(format!("Model load task failed: {e}")))??;

        let report = model_box.weight_report();
        let mut lock = self.active_model.write().await;
        *lock = Some(model_box);
        tracing::info!("Model '{}' loaded into active memory.", name);
        Ok(report)
    }

    /// Unload currently active model from memory
    pub async fn unload_model(&self) {
        let mut lock = self.active_model.write().await;
        *lock = None;
        tracing::info!("Active model unloaded from memory.");
    }

    /// Get active model name if loaded
    pub async fn get_active_model_name(&self) -> Option<String> {
        let lock = self.active_model.read().await;
        lock.as_ref().map(|m| m.name().to_string())
    }

    /// Modality of the active model, if any.
    pub async fn get_active_modality(&self) -> Option<ModelModality> {
        let lock = self.active_model.read().await;
        lock.as_ref().map(|m| m.modality())
    }

    /// Checkpoint coverage report of the active model, if any.
    pub async fn get_active_weight_report(&self) -> Option<WeightReport> {
        let lock = self.active_model.read().await;
        lock.as_ref().map(|m| m.weight_report())
    }

    /// Embed an image tensor [1, 3, 224, 224]
    pub async fn embed_image(&self, img: &Tensor) -> Result<(String, usize, Vec<f32>, Option<Vec<Vec<f32>>>, f64), JepaError> {
        let lock = self.active_model.read().await;
        let model = lock.as_ref().ok_or_else(|| {
            JepaError::ModelNotFound("No model currently loaded in memory. Load a model first.".to_string())
        })?;

        let name = model.name().to_string();
        let dim = model.dimension();
        let (pooled, patches, latency) = model.embed_image(img)?;
        Ok((name, dim, pooled, patches, latency))
    }

    /// Embed a spatio-temporal video tensor [1, 3, 16, 224, 224]
    pub async fn embed_video(&self, vid: &Tensor) -> Result<(String, usize, Vec<f32>, f64), JepaError> {
        let lock = self.active_model.read().await;
        let model = lock.as_ref().ok_or_else(|| {
            JepaError::ModelNotFound("No model currently loaded in memory. Load a model first.".to_string())
        })?;

        let name = model.name().to_string();
        let dim = model.dimension();
        let (pooled, latency) = model.embed_video(vid)?;
        Ok((name, dim, pooled, latency))
    }

    /// Update telemetry
    pub async fn get_telemetry(&self) -> HardwareInfo {
        let current_backend = self.hardware_info.read().await.backend;
        let updated = device::query_telemetry(current_backend, None);
        let mut lock = self.hardware_info.write().await;
        *lock = updated.clone();
        updated
    }
}
