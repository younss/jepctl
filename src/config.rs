//! Cross-platform storage paths, directory layout, and security constants.

use crate::types::{JepaError, SettingsDto};
use std::fs;
use std::path::{Component, Path, PathBuf};

pub const DEFAULT_HOST: &str = "127.0.0.1";

/// Model used by CLI commands when none is given.
pub const DEFAULT_MODEL: &str = "facebook/ijepa_vith14_1k";
pub const DEFAULT_PORT: u16 = 11435;

/// Maximum payload limit for single image embedding requests: 20 Megabytes
pub const MAX_IMAGE_PAYLOAD_SIZE: usize = 20 * 1024 * 1024;

/// Maximum payload limit for video file uploads: 200 Megabytes
pub const MAX_VIDEO_PAYLOAD_SIZE: usize = 200 * 1024 * 1024;

/// Temporal frames required for V-JEPA spatio-temporal representations
pub const VJEPA_TEMPORAL_FRAMES: usize = 16;

/// Default ring buffer capacity
pub const RING_BUFFER_CAPACITY: usize = 16;

/// Canonical runtime settings configuration
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub home_dir: PathBuf,
    pub models_dir: PathBuf,
    pub logs_dir: PathBuf,
    pub auth_token_path: PathBuf,
    pub keys_db_path: PathBuf,
    pub settings_path: PathBuf,
    /// Persisted few-shot gesture registry.
    pub gestures_path: PathBuf,
    pub host: String,
    pub port: u16,
    pub no_auth: bool,
    /// Browser origins allowed to call the API cross-site. Empty = same-origin only.
    pub cors_origins: Vec<String>,
}

impl RuntimeConfig {
    /// Resolve default root storage directory:
    /// - macOS and Linux: ~/.jepa/
    /// - Windows: %USERPROFILE%/.jepa/
    pub fn default_root_dir() -> PathBuf {
        if let Some(home) = dirs::home_dir() {
            home.join(".jepa")
        } else {
            PathBuf::from(".jepa")
        }
    }

    /// Initialize directory structure and create required subfolders
    pub fn init(host: String, port: u16, no_auth: bool) -> Result<Self, JepaError> {
        let home_dir = Self::default_root_dir();
        let models_dir = home_dir.join("models");
        let logs_dir = home_dir.join("logs");
        let auth_token_path = home_dir.join("auth.token");
        let keys_db_path = home_dir.join("keys.json");
        let settings_path = home_dir.join("settings.json");
        let gestures_path = home_dir.join("gestures.json");

        // Ensure directories exist with proper permissions
        fs::create_dir_all(&models_dir)?;
        fs::create_dir_all(&logs_dir)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&home_dir, fs::Permissions::from_mode(0o700));
        }

        Ok(Self {
            home_dir,
            models_dir,
            logs_dir,
            auth_token_path,
            keys_db_path,
            settings_path,
            gestures_path,
            host,
            port,
            no_auth,
            cors_origins: Vec::new(),
        })
    }

    /// Load runtime settings from disk if available, or return defaults.
    pub fn load_settings(&self) -> SettingsDto {
        if self.settings_path.exists() {
            if let Ok(data) = fs::read_to_string(&self.settings_path) {
                if let Ok(settings) = serde_json::from_str::<SettingsDto>(&data) {
                    return settings;
                }
            }
        }
        SettingsDto {
            compute_backend: "auto".to_string(),
            gpu_memory_high_watermark: 0.85,
            idle_unload_timeout_minutes: 15,
            storage_dir: self.models_dir.to_string_lossy().to_string(),
        }
    }

    /// Path traversal prevention helper.
    /// Strictly guarantees that a model name or user-specified path cannot escape models_dir.
    pub fn safe_resolve_model_path(&self, model_name_or_file: &str) -> Result<PathBuf, JepaError> {
        // Disallow path traversal components
        if model_name_or_file.contains("..") {
            return Err(JepaError::PathTraversal("Identifier contains invalid sequence '..'".into()));
        }

        // Normalize colons into safe hierarchical directory paths
        let normalized = model_name_or_file.replace(':', "/");
        let input_path = Path::new(&normalized);

        // Disallow components that go up or root traversal
        for component in input_path.components() {
            match component {
                Component::ParentDir => {
                    return Err(JepaError::PathTraversal(format!(
                        "Path traversal attempt rejected: {}",
                        model_name_or_file
                    )));
                }
                Component::RootDir | Component::Prefix(_) => {
                    return Err(JepaError::PathTraversal(format!(
                        "Absolute path references not allowed in model identifier: {}",
                        model_name_or_file
                    )));
                }
                Component::CurDir | Component::Normal(_) => {}
            }
        }

        // Normalize slashes into safe directory path
        let sanitized = normalized.trim_matches('/');
        let resolved = self.models_dir.join(sanitized);

        // Double check canonicalization if parent exists
        if let Some(parent) = resolved.parent() {
            if parent.exists() {
                if let (Ok(can_models), Ok(can_parent)) = (self.models_dir.canonicalize(), parent.canonicalize()) {
                    if !can_parent.starts_with(&can_models) {
                        return Err(JepaError::PathTraversal(format!(
                            "Resolved path escaped model directory: {}",
                            model_name_or_file
                        )));
                    }
                }
            }
        }

        Ok(resolved)
    }
}
