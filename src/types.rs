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

    #[test]
    fn test_app_js_syntax_balance() {
        let js = include_str!("ui/app.js");
        let chars: Vec<char> = js.chars().collect();
        let mut i = 0;
        let mut line_num = 1;
        let mut in_line_comment = false;
        let mut in_block_comment = false;
        let mut in_single_quote = false;
        let mut in_double_quote = false;
        let mut escaped = false;
        // Stack can contain: '{', '(', '[', '`'
        let mut stack: Vec<(char, usize)> = Vec::new();

        while i < chars.len() {
            let c = chars[i];
            let next_c = if i + 1 < chars.len() { chars[i + 1] } else { '\0' };

            if c == '\n' {
                line_num += 1;
                in_line_comment = false;
                escaped = false;
                i += 1;
                continue;
            }

            if in_line_comment {
                i += 1;
                continue;
            }

            if in_block_comment {
                if c == '*' && next_c == '/' {
                    in_block_comment = false;
                    i += 2;
                } else {
                    i += 1;
                }
                continue;
            }

            if in_single_quote {
                if !escaped && c == '\'' {
                    in_single_quote = false;
                }
                escaped = !escaped && c == '\\';
                i += 1;
                continue;
            }

            if in_double_quote {
                if !escaped && c == '"' {
                    in_double_quote = false;
                }
                escaped = !escaped && c == '\\';
                i += 1;
                continue;
            }

            // Check if we are inside a template literal (stack top is '`')
            if let Some(&('`', _)) = stack.last() {
                if !escaped && c == '`' {
                    stack.pop();
                    i += 1;
                    continue;
                }
                if !escaped && c == '$' && next_c == '{' {
                    stack.push(('{', line_num));
                    i += 2;
                    continue;
                }
                escaped = !escaped && c == '\\';
                i += 1;
                continue;
            }

            // Normal code
            if c == '/' && next_c == '/' {
                in_line_comment = true;
                i += 2;
                continue;
            }
            if c == '/' && next_c == '*' {
                in_block_comment = true;
                i += 2;
                continue;
            }
            if c == '\'' {
                in_single_quote = true;
                escaped = false;
                i += 1;
                continue;
            }
            if c == '"' {
                in_double_quote = true;
                escaped = false;
                i += 1;
                continue;
            }
            if c == '`' {
                stack.push(('`', line_num));
                escaped = false;
                i += 1;
                continue;
            }

            if c == '{' || c == '(' || c == '[' {
                stack.push((c, line_num));
            } else if c == '}' {
                match stack.pop() {
                    Some(('{', _)) => {},
                    other => panic!("Mismatched '}}' at line {}: expected '{{', found {:?}", line_num, other),
                }
            } else if c == ')' {
                match stack.pop() {
                    Some(('(', _)) => {},
                    other => panic!("Mismatched ')' at line {}: expected '(', found {:?}", line_num, other),
                }
            } else if c == ']' {
                match stack.pop() {
                    Some(('[', _)) => {},
                    other => panic!("Mismatched ']' at line {}: expected '[', found {:?}", line_num, other),
                }
            }

            i += 1;
        }

        assert!(stack.is_empty(), "Unclosed tokens remaining on stack: {:?}", stack);
    }
}
