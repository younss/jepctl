//! Engine abstraction traits and runtime model execution manager.

pub mod audio;
pub mod device;
pub mod ijepa;
pub mod vit;
pub mod vjepa2;

use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::engine::audio::AudioModel;
use crate::engine::ijepa::IJepaModel;
use crate::engine::vit::{load_safetensors_into_backbone, VitBackbone, VitConfig};
use crate::engine::vjepa2::VJepa2Model;
use crate::types::{HardwareInfo, JepaError, ModelManifest, ModelModality, Preprocessing, WeightReport};

/// Instantiate a randomly initialised backbone matching a manifest.
pub(crate) fn build_backbone(manifest: &ModelManifest, device: &Device) -> Result<(VarMap, VitBackbone), JepaError> {
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let cfg = VitConfig {
        img_size: manifest.image_size,
        img_w: manifest.input_width,
        patch_size: manifest.patch_size,
        in_chans: manifest.in_chans.unwrap_or(3),
        embed_dim: manifest.embed_dim,
        depth: manifest.num_layers,
        num_heads: manifest.num_heads,
        mlp_ratio: manifest.mlp_ratio(),
        variant: manifest.backbone_variant(),
        pooling: manifest.pooling(),
    };
    let backbone = VitBackbone::new(&cfg, vb)?;
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
            "Weights for '{}' not found at {}. Run `jepctl pull {}` first.",
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
        return Err(JepaError::WeightsIncomplete { loaded: outcome.loaded, expected: outcome.expected });
    }
    if !outcome.pos_embed_loaded {
        tracing::info!("'{}' uses fixed 2D sin-cos positional embeddings (none found in checkpoint).", model_name);
    }
    tracing::info!(
        "Loaded {}/{} tensors into '{}' from {}",
        outcome.loaded,
        outcome.expected,
        model_name,
        path.display()
    );
    Ok(WeightReport { loaded: outcome.loaded, expected: outcome.expected, source: "safetensors".to_string() })
}

/// `(pooled embedding, per-patch tokens if available, latency in ms)`.
pub type ImageEmbedding = (Vec<f32>, Option<Vec<Vec<f32>>>, f64);

/// Common trait for all JEPA model variants (I-JEPA, V-JEPA, Audio-JEPA)
pub trait JepaModelTrait: Send + Sync {
    fn name(&self) -> &str;
    fn modality(&self) -> ModelModality;
    fn dimension(&self) -> usize;
    fn weight_report(&self) -> WeightReport;
    fn preprocessing(&self) -> Preprocessing;
    /// Number of frames a clip should be sub-sampled to for this model.
    fn clip_frames(&self) -> usize;
    fn embed_image(&self, img: &Tensor) -> Result<ImageEmbedding, JepaError>;
    /// Embed a clip `[1, 3, T, H, W]`. Patch tokens (when returned) describe the last
    /// temporal slot so spatial explanations work for video models too.
    fn embed_video(&self, vid: &Tensor) -> Result<ImageEmbedding, JepaError>;
    /// Embed decoded audio. Only audio models implement this.
    fn embed_audio(&self, _clip: &crate::media::audio::AudioClip) -> Result<ImageEmbedding, JepaError> {
        Err(JepaError::InvalidPayload(format!("'{}' is not an audio model", self.name())))
    }
}

impl JepaModelTrait for AudioModel {
    fn name(&self) -> &str {
        &self.manifest.name
    }

    fn modality(&self) -> ModelModality {
        ModelModality::Audio
    }

    fn dimension(&self) -> usize {
        self.manifest.embed_dim
    }

    fn weight_report(&self) -> WeightReport {
        self.weights.clone()
    }

    fn preprocessing(&self) -> Preprocessing {
        self.manifest.preprocessing()
    }

    fn clip_frames(&self) -> usize {
        1
    }

    fn embed_image(&self, _img: &Tensor) -> Result<ImageEmbedding, JepaError> {
        Err(JepaError::InvalidPayload(format!("'{}' is an audio model: send WAV/MP3/FLAC/OGG", self.manifest.name)))
    }

    fn embed_video(&self, _vid: &Tensor) -> Result<ImageEmbedding, JepaError> {
        Err(JepaError::InvalidPayload(format!("'{}' is an audio model: send WAV/MP3/FLAC/OGG", self.manifest.name)))
    }

    fn embed_audio(&self, clip: &crate::media::audio::AudioClip) -> Result<ImageEmbedding, JepaError> {
        let (pooled, patches, latency) = self.embed_clip(clip)?;
        Ok((pooled, Some(patches), latency))
    }
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

    fn preprocessing(&self) -> Preprocessing {
        self.manifest.preprocessing()
    }

    fn clip_frames(&self) -> usize {
        8
    }

    fn embed_image(&self, img: &Tensor) -> Result<ImageEmbedding, JepaError> {
        let (pooled, patches, latency) = self.forward_image(img)?;
        Ok((pooled, Some(patches), latency))
    }

    fn embed_video(&self, vid: &Tensor) -> Result<ImageEmbedding, JepaError> {
        // Image models embed the latest frame of the clip.
        let (_b, _c, t, _h, _w) = vid.dims5()?;
        let latest_frame = vid.narrow(2, t - 1, 1)?.squeeze(2)?;
        let (pooled, patches, latency) = self.forward_image(&latest_frame)?;
        Ok((pooled, Some(patches), latency))
    }
}

impl JepaModelTrait for VJepa2Model {
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

    fn preprocessing(&self) -> Preprocessing {
        self.manifest.preprocessing()
    }

    fn clip_frames(&self) -> usize {
        self.manifest.frames.unwrap_or(crate::config::VJEPA_TEMPORAL_FRAMES)
    }

    fn embed_image(&self, img: &Tensor) -> Result<ImageEmbedding, JepaError> {
        let (pooled, patches, latency) = self.forward_image(img)?;
        Ok((pooled, Some(patches), latency))
    }

    fn embed_video(&self, vid: &Tensor) -> Result<ImageEmbedding, JepaError> {
        let (pooled, patches, latency) = self.forward_video(vid)?;
        Ok((pooled, Some(patches), latency))
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
        Arc::new(Self { active_model: RwLock::new(None), device, hardware_info: RwLock::new(hardware_info) })
    }

    /// Load a model into active GPU/system memory. `weights_path` must point to a
    /// downloaded safetensors file; a missing or unsupported checkpoint is an error.
    pub async fn load_model(
        &self,
        manifest: ModelManifest,
        weights_path: Option<&Path>,
    ) -> Result<WeightReport, JepaError> {
        let name = manifest.name.clone();
        let path = weights_path.ok_or_else(|| {
            JepaError::ModelNotFound(format!("'{}' is not downloaded. Run `jepctl pull {}` first.", name, name))
        })?;

        // Model construction is CPU-heavy (mmap + copies); keep it off the async executor.
        let device = self.device.clone();
        let path = path.to_path_buf();
        let model_box: Box<dyn JepaModelTrait> =
            tokio::task::spawn_blocking(move || -> Result<Box<dyn JepaModelTrait>, JepaError> {
                Ok(match manifest.modality {
                    ModelModality::Video => Box::new(VJepa2Model::load(manifest, &path, device)?),
                    ModelModality::Audio => Box::new(AudioModel::load(manifest, &path, device)?),
                    ModelModality::Image => Box::new(IJepaModel::load(manifest, &path, device)?),
                    other => {
                        return Err(JepaError::InvalidPayload(format!(
                            "No engine implements the '{}' modality yet",
                            other
                        )))
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

    /// Input preprocessing of the active model (defaults when nothing is loaded).
    pub async fn preprocessing(&self) -> Preprocessing {
        let lock = self.active_model.read().await;
        lock.as_ref().map(|m| m.preprocessing()).unwrap_or_default()
    }

    /// Direct access to the active model slot for tests.
    #[doc(hidden)]
    pub async fn engine_active_model_for_test(
        &self,
    ) -> tokio::sync::RwLockWriteGuard<'_, Option<Box<dyn JepaModelTrait>>> {
        self.active_model.write().await
    }

    /// Build a randomly initialised image model for tests: embeddings are meaningless
    /// but every API path can be exercised without downloading weights.
    #[doc(hidden)]
    pub async fn load_random_for_test(&self, manifest: ModelManifest) -> Result<(), JepaError> {
        let model = IJepaModel::load_random(manifest, self.device.clone())?;
        let mut lock = self.active_model.write().await;
        *lock = Some(Box::new(model));
        Ok(())
    }

    /// Embed an image tensor [1, 3, H, W]
    pub async fn embed_image(
        &self,
        img: &Tensor,
    ) -> Result<(String, usize, Vec<f32>, Option<Vec<Vec<f32>>>, f64), JepaError> {
        let lock = self.active_model.read().await;
        let model = lock.as_ref().ok_or_else(|| {
            JepaError::ModelNotFound("No model currently loaded in memory. Load a model first.".to_string())
        })?;

        let name = model.name().to_string();
        let dim = model.dimension();
        let (pooled, patches, latency) = model.embed_image(img)?;
        Ok((name, dim, pooled, patches, latency))
    }

    /// Embed a spatio-temporal video tensor [1, 3, T, H, W]
    pub async fn embed_video(
        &self,
        vid: &Tensor,
    ) -> Result<(String, usize, Vec<f32>, Option<Vec<Vec<f32>>>, f64), JepaError> {
        let lock = self.active_model.read().await;
        let model = lock.as_ref().ok_or_else(|| {
            JepaError::ModelNotFound("No model currently loaded in memory. Load a model first.".to_string())
        })?;

        let name = model.name().to_string();
        let dim = model.dimension();
        let (pooled, patches, latency) = model.embed_video(vid)?;
        Ok((name, dim, pooled, patches, latency))
    }

    /// Embed decoded audio with the active (audio) model.
    pub async fn embed_audio(
        &self,
        clip: &crate::media::audio::AudioClip,
    ) -> Result<(String, usize, Vec<f32>, Option<Vec<Vec<f32>>>, f64), JepaError> {
        let lock = self.active_model.read().await;
        let model = lock.as_ref().ok_or_else(|| {
            JepaError::ModelNotFound("No model currently loaded in memory. Load a model first.".to_string())
        })?;
        let name = model.name().to_string();
        let dim = model.dimension();
        let (pooled, patches, latency) = model.embed_audio(clip)?;
        Ok((name, dim, pooled, patches, latency))
    }

    /// Embed a decoded clip. Video models get a `[1, 3, T, H, W]` tensor of `T`
    /// uniformly sampled frames; image models embed each sampled frame and return the
    /// mean of the pooled vectors (no spatial tokens in that case).
    pub async fn embed_frames(
        &self,
        frames: &[image::RgbImage],
    ) -> Result<(String, usize, Vec<f32>, Option<Vec<Vec<f32>>>, f64, usize), JepaError> {
        if frames.is_empty() {
            return Err(JepaError::InvalidPayload("Clip has no frames".into()));
        }
        let lock = self.active_model.read().await;
        let model = lock.as_ref().ok_or_else(|| {
            JepaError::ModelNotFound("No model currently loaded in memory. Load a model first.".to_string())
        })?;
        let name = model.name().to_string();
        let dim = model.dimension();
        let prep = model.preprocessing();
        let count = model.clip_frames();
        let sampled = crate::media::video::sample_uniform(frames, count);
        let tensors = sampled
            .iter()
            .map(|f| {
                crate::media::image::preprocess_dynamic_image(
                    &image::DynamicImage::ImageRgb8(f.clone()),
                    &prep,
                    &self.device,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        match model.modality() {
            ModelModality::Video => {
                // [T, 1, 3, H, W] -> [1, 3, T, H, W]
                let clip = Tensor::stack(&tensors, 0)?.squeeze(1)?.transpose(0, 1)?.unsqueeze(0)?.contiguous()?;
                let (pooled, patches, latency) = model.embed_video(&clip)?;
                Ok((name, dim, pooled, patches, latency, sampled.len()))
            }
            _ => {
                let mut sum = vec![0f32; dim];
                let mut latency = 0.0;
                for t in &tensors {
                    let (pooled, _patches, lat) = model.embed_image(t)?;
                    for (s, p) in sum.iter_mut().zip(pooled.iter()) {
                        *s += p;
                    }
                    latency += lat;
                }
                let n = tensors.len() as f32;
                sum.iter_mut().for_each(|s| *s /= n);
                Ok((name, dim, sum, None, latency, sampled.len()))
            }
        }
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
