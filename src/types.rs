//! Shared data transfer objects, schemas, and error definitions for JEPA runtime.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Core error types across the JEPA runtime.
#[derive(Error, Debug)]
pub enum JepaError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Device error: {0}")]
    DeviceError(String),

    #[error("Inference execution failed: {0}")]
    InferenceError(String),

    #[error("Authentication failed: {0}")]
    AuthError(String),

    #[error("Authorization forbidden: {0}")]
    Forbidden(String),

    #[error("Invalid request payload: {0}")]
    InvalidPayload(String),

    #[error("Payload size exceeded limit: {0} bytes")]
    PayloadTooLarge(usize),

    #[error("Rate limit exceeded")]
    RateLimitExceeded,

    #[error("Camera capture error: {0}")]
    CameraError(String),

    #[error("Hugging Face Hub error: {0}")]
    HubError(String),

    #[error("Path traversal detected: {0}")]
    PathTraversal(String),

    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Image processing error: {0}")]
    ImageProcessing(String),

    #[error("Candle framework error: {0}")]
    Candle(#[from] candle_core::Error),

    #[error("Weights incomplete: {loaded}/{expected} tensors loaded from checkpoint")]
    WeightsIncomplete { loaded: usize, expected: usize },
}

/// Supported representation learning modalities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelModality {
    Image,
    Video,
    Multimodal,
    Audio,
}

impl std::fmt::Display for ModelModality {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Image => write!(f, "image"),
            Self::Video => write!(f, "video"),
            Self::Multimodal => write!(f, "multimodal"),
            Self::Audio => write!(f, "audio"),
        }
    }
}

/// Compute acceleration backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum HardwareBackend {
    Metal,
    Cuda,
    Cpu,
}

impl std::fmt::Display for HardwareBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Metal => write!(f, "Metal (Apple Silicon)"),
            Self::Cuda => write!(f, "CUDA (NVIDIA)"),
            Self::Cpu => write!(f, "CPU (Optimized Multi-thread)"),
        }
    }
}

/// System hardware telemetry and compute information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub backend: HardwareBackend,
    pub device_name: String,
    pub memory_used_bytes: u64,
    pub memory_total_bytes: u64,
    pub memory_percent: f32,
    pub temperature_celsius: Option<f32>,
    pub cpu_threads: usize,
}

/// Pixel normalisation applied before the patch embedding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum Normalization {
    /// mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225): I-JEPA, DINOv2, HF ViT.
    #[default]
    ImageNet,
    /// mean 0.5, std 0.5 on every channel: timm "augreg" ViTs, SigLIP.
    Inception,
}

impl Normalization {
    pub fn mean(self) -> [f32; 3] {
        match self {
            Self::ImageNet => [0.485, 0.456, 0.406],
            Self::Inception => [0.5, 0.5, 0.5],
        }
    }

    pub fn std(self) -> [f32; 3] {
        match self {
            Self::ImageNet => [0.229, 0.224, 0.225],
            Self::Inception => [0.5, 0.5, 0.5],
        }
    }
}

/// How a model expects its input prepared. Shared by every entry point (upload,
/// camera, CLI) so that reference and live embeddings are always comparable.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Preprocessing {
    /// Side of the square input (centre crop + resize).
    pub size: u32,
    pub normalization: Normalization,
}

impl Default for Preprocessing {
    fn default() -> Self {
        Self { size: 224, normalization: Normalization::ImageNet }
    }
}

/// Region of interest on the raw camera frame, normalised to `[0, 1]` (x, y = top-left).
/// When set, the camera pipeline crops to it *before* the centre crop and resize, so a
/// hand can fill the model input instead of being a few patches in a wide frame.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Roi {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Roi {
    /// Clamp to the unit square and refuse degenerate boxes.
    pub fn normalized(self) -> Result<Self, JepaError> {
        let x = self.x.clamp(0.0, 1.0);
        let y = self.y.clamp(0.0, 1.0);
        let w = self.w.min(1.0 - x);
        let h = self.h.min(1.0 - y);
        if !(w > 0.02 && h > 0.02) || !x.is_finite() || !y.is_finite() {
            return Err(JepaError::InvalidPayload("ROI must be at least 2% of the frame in both dimensions".into()));
        }
        Ok(Self { x, y, w, h })
    }
}

/// Jepafile manifest definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelManifest {
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
    pub parameter_count: String,
    pub disk_size_bytes: u64,
    pub weights_file: String,
    pub created_at: DateTime<Utc>,
    /// Backbone structure (CLS token, LayerScale). Inferred from `name` when absent.
    #[serde(default)]
    pub variant: Option<crate::engine::vit::VitVariant>,
    /// Pixel normalisation. Inferred from `name` when absent.
    #[serde(default)]
    pub normalization: Option<Normalization>,
    /// MLP hidden ratio (4.0 for every catalog model except ViT-g).
    #[serde(default)]
    pub mlp_ratio: Option<f64>,
    /// Temporal patch size of video encoders (V-JEPA 2: 2).
    #[serde(default)]
    pub tubelet_size: Option<usize>,
    /// Input width when it differs from `image_size` (audio spectrograms: mel bins).
    #[serde(default)]
    pub input_width: Option<usize>,
    /// Input channels (1 for spectrograms, 3 for RGB).
    #[serde(default)]
    pub in_chans: Option<usize>,
    /// Audio front-end parameters for `modality: audio`.
    #[serde(default)]
    pub audio: Option<AudioSpec>,
    /// Pooled-output read-out (`cls` or `mean`). Default: `cls` for CLS backbones, else `mean`.
    #[serde(default)]
    pub pooling: Option<crate::engine::vit::Pooling>,
}

/// Log-mel front-end parameters (Kaldi fbank conventions, as used by AudioMAE / AST).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct AudioSpec {
    pub sample_rate: u32,
    pub n_mels: usize,
    /// Number of 10 ms frames the model expects (time axis, rows).
    pub frames: usize,
    /// Dataset mean/std applied as `(fbank - mean) / (2 * std)`.
    pub mean: f32,
    pub std: f32,
}

impl Default for AudioSpec {
    fn default() -> Self {
        // AudioMAE / AST defaults (AudioSet statistics).
        Self { sample_rate: 16_000, n_mels: 128, frames: 1024, mean: -4.2677393, std: 4.5689974 }
    }
}

impl ModelManifest {
    /// Backbone variant, explicit or inferred from the model family.
    pub fn backbone_variant(&self) -> crate::engine::vit::VitVariant {
        use crate::engine::vit::VitVariant;
        if let Some(v) = self.variant {
            return v;
        }
        let n = self.name.to_ascii_lowercase();
        if n.contains("vjepa2") || self.modality == ModelModality::Video {
            VitVariant::VJepa2
        } else if n.contains("audiomae") || n.contains("ast") && self.modality == ModelModality::Audio {
            VitVariant::Cls
        } else if n.contains("dinov2") {
            VitVariant::DinoV2
        } else if n.contains("jepa") {
            VitVariant::Plain
        } else if n.contains("vit") {
            VitVariant::Cls
        } else {
            VitVariant::Plain
        }
    }

    /// Input preprocessing, explicit or inferred from the model family.
    pub fn preprocessing(&self) -> Preprocessing {
        let normalization = self.normalization.unwrap_or_else(|| {
            let n = self.name.to_ascii_lowercase();
            if (n.starts_with("timm/") && n.contains("augreg")) || n.starts_with("google/vit") || n.contains("siglip") {
                Normalization::Inception
            } else {
                Normalization::ImageNet
            }
        });
        Preprocessing { size: self.image_size as u32, normalization }
    }

    pub fn mlp_ratio(&self) -> f64 {
        self.mlp_ratio.unwrap_or(4.0)
    }

    /// Pooling read-out, explicit or inferred. MAE-style encoders keep a CLS token that
    /// was never trained as a summary, so they must pool by mean.
    pub fn pooling(&self) -> crate::engine::vit::Pooling {
        use crate::engine::vit::Pooling;
        if let Some(p) = self.pooling {
            return p;
        }
        let n = self.name.to_ascii_lowercase();
        if n.contains("mae") {
            Pooling::Mean
        } else if self.backbone_variant().has_cls() {
            Pooling::Cls
        } else {
            Pooling::Mean
        }
    }
}

/// System health and status response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusResponse {
    pub status: String,
    pub version: String,
    pub platform: String,
    pub hardware: HardwareInfo,
    pub active_model: Option<String>,
    /// How the active model's weights were obtained (checkpoint coverage).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub weights: Option<WeightReport>,
    /// Whether the server camera capture thread is running.
    #[serde(default)]
    pub camera_active: bool,
    pub embeddings_computed_total: u64,
    pub uptime_seconds: u64,
}

/// Checkpoint coverage of a loaded model. `loaded == expected` means every
/// parameter came from the safetensors file; anything else means the model is
/// (partially) random and its embeddings are meaningless.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WeightReport {
    pub loaded: usize,
    pub expected: usize,
    pub source: String,
}

impl WeightReport {
    pub fn is_complete(&self) -> bool {
        self.loaded == self.expected && self.expected > 0
    }
}

/// Embed response payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedResponse {
    pub model: String,
    pub dimension: usize,
    pub latency_ms: f64,
    pub embedding: Vec<f32>,
    pub patch_embeddings: Option<Vec<Vec<f32>>>,
}

/// Server-Sent Events stream message schema.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamEvent {
    pub frame_index: usize,
    pub timestamp: f64,
    pub model: String,
    pub latency_ms: f64,
    pub embedding: Vec<f32>,
    pub energy_distance: Option<f32>,
    pub anomaly: bool,
    /// Gesture matching result, present when gestures are registered for the active model.
    #[serde(rename = "gesture_match", skip_serializing_if = "Option::is_none")]
    pub gesture_match: Option<crate::gestures::GestureMatchResult>,
}

/// Energy calculation request payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyRequest {
    pub vector1: Option<Vec<f32>>,
    pub vector2: Option<Vec<f32>>,
    pub frame1_base64: Option<String>,
    pub frame2_base64: Option<String>,
    pub threshold: Option<f32>,
}

/// Energy calculation response payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyResponse {
    pub l2_distance: f32,
    pub cosine_similarity: f32,
    pub cosine_dissimilarity: f32,
    pub anomaly: bool,
    pub threshold: f32,
}

/// API token access scopes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    Admin,
    Inference,
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Admin => write!(f, "admin"),
            Self::Inference => write!(f, "inference"),
        }
    }
}

/// Stored API key record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiKeyRecord {
    pub key_prefix: String,
    pub token_hash: String,
    pub role: Role,
    pub name: String,
    pub created_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
}

/// Key creation request payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateKeyRequest {
    pub name: String,
    pub role: Role,
    pub expire_days: Option<i64>,
}

/// Key creation response payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateKeyResponse {
    pub key_prefix: String,
    pub raw_token: String,
    pub role: Role,
    pub name: String,
    pub created_at: DateTime<Utc>,
}

/// Audit log entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditLogEntry {
    pub timestamp: DateTime<Utc>,
    pub method: String,
    pub path: String,
    pub client_ip: String,
    pub status_code: u16,
    pub latency_ms: f64,
}

/// Hugging Face model pull event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullProgressEvent {
    pub repo_id: String,
    pub status: String,
    pub downloaded_bytes: u64,
    pub total_bytes: u64,
    pub speed_mb_s: f64,
    pub percentage: f32,
    pub finished: bool,
    pub error: Option<String>,
}

/// Camera device info.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CameraDeviceInfo {
    pub index: usize,
    pub name: String,
}

/// Settings update payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SettingsDto {
    pub compute_backend: String,
    pub gpu_memory_high_watermark: f32,
    pub idle_unload_timeout_minutes: i64,
    pub storage_dir: String,
    /// Region of interest applied to camera frames before embedding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub camera_roi: Option<Roi>,
    /// Serial settings for the physical robot arm.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub robot_hardware: Option<crate::robot::hal::HardwareConfig>,
}

/// Compute L2 norm and normalize vector to unit length (norm L2 = 1.0)
pub fn normalize_l2(v: &[f32]) -> Vec<f32> {
    let norm_sq: f32 = v.iter().map(|x| x * x).sum();
    let norm = norm_sq.sqrt();
    if norm > 1e-12 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Compute cosine similarity between two unit vectors via vector dot product
#[inline]
pub fn dot_product(u: &[f32], v: &[f32]) -> f32 {
    u.iter().zip(v.iter()).map(|(a, b)| a * b).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_and_dot_product() {
        let v1 = vec![3.0, 4.0, 0.0];
        let n1 = normalize_l2(&v1);
        assert!((n1[0] - 0.6).abs() < 1e-5);
        assert!((n1[1] - 0.8).abs() < 1e-5);

        let sim_self = dot_product(&n1, &n1);
        assert!((sim_self - 1.0).abs() < 1e-5);

        let v2 = vec![-4.0, 3.0, 0.0];
        let n2 = normalize_l2(&v2);
        let sim_ortho = dot_product(&n1, &n2);
        assert!(sim_ortho.abs() < 1e-5);
    }

    /// Every static `getElementById("...")` in app.js must exist in index.html.
    /// (Template-literal ids such as `slot-img-${i}` are generated and skipped.)
    #[test]
    fn ui_dom_ids_referenced_by_js_exist_in_html() {
        let js = include_str!("ui/app.js");
        let html = include_str!("ui/index.html");
        let mut missing = Vec::new();
        for chunk in js.split("getElementById(\"").skip(1) {
            if let Some(id) = chunk.split('"').next() {
                if !html.contains(&format!("id=\"{id}\"")) {
                    missing.push(id.to_string());
                }
            }
        }
        missing.sort();
        missing.dedup();
        assert!(missing.is_empty(), "ids used in app.js but absent from index.html: {missing:?}");
    }
}
