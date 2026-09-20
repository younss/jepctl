//! Local model catalog registry manager and model lifecycle operations.

pub mod downloader;
pub mod manifest;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::broadcast;

use crate::config::RuntimeConfig;
use crate::hub::downloader::ModelDownloader;
use crate::hub::manifest::{get_verified_manifests, JepafileConfig};
use crate::types::{JepaError, ModelManifest, PullProgressEvent};

pub struct ModelCatalog {
    config: Arc<RuntimeConfig>,
    downloader: ModelDownloader,
}

impl ModelCatalog {
    pub fn new(config: Arc<RuntimeConfig>) -> Self {
        Self { config, downloader: ModelDownloader::new() }
    }

    /// List all locally installed models and registered manifests
    pub fn list_installed(&self) -> Vec<ModelManifest> {
        let mut models = Vec::new();
        let models_dir = &self.config.models_dir;

        if !models_dir.exists() {
            return models;
        }

        // 1. Scan verified models and check if present on disk
        for verified in get_verified_manifests() {
            if let Ok(safe_path) = self.config.safe_resolve_model_path(&verified.name) {
                if safe_path.exists() {
                    let weights_path = safe_path.join("model.safetensors");
                    let mut m = verified.clone();
                    if weights_path.exists() {
                        if let Ok(metadata) = fs::metadata(&weights_path) {
                            m.disk_size_bytes = metadata.len();
                        }
                    } else {
                        m.disk_size_bytes = 0;
                    }
                    models.push(m);
                }
            }
        }

        // 2. Scan custom Jepafile manifests in ~/.jepa/models/ recursively
        let mut manifests = Vec::new();
        scan_manifests_recursive(models_dir, 0, &mut manifests);
        for manifest_file in manifests {
            if let Ok(cfg) = JepafileConfig::load_from_file(&manifest_file) {
                if let Some(parent) = manifest_file.parent() {
                    let weights = parent.join("model.safetensors");
                    let size = fs::metadata(&weights).map(|m| m.len()).unwrap_or(0);
                    if let Ok(m) = cfg.to_manifest(size) {
                        if !models.iter().any(|existing| existing.name == m.name || existing.repo_id == m.repo_id) {
                            models.push(m);
                        }
                    }
                }
            }
        }

        models
    }

    /// Resolve model manifest by name
    pub fn get_manifest(&self, model_name: &str) -> Option<ModelManifest> {
        let clean = model_name.replace("%2F", "/").replace("%2f", "/");
        // First check locally installed
        if let Some(installed) = self.list_installed().into_iter().find(|m| m.name == clean || m.repo_id == clean) {
            return Some(installed);
        }

        // Then check verified catalog
        get_verified_manifests().into_iter().find(|m| m.name == clean || m.repo_id == clean)
    }

    /// Resolve absolute path to weights safetensors file
    pub fn get_weights_path(&self, model_name: &str) -> Option<PathBuf> {
        let clean = model_name.replace("%2F", "/").replace("%2f", "/");
        if let Ok(safe_path) = self.config.safe_resolve_model_path(&clean) {
            let weights = safe_path.join("model.safetensors");
            if weights.exists() {
                return Some(weights);
            }
        }
        let alt = if clean.contains(':') {
            clean.replace(':', "/")
        } else if let Some(last_slash) = clean.rfind('/') {
            let (prefix, suffix) = clean.split_at(last_slash);
            format!("{}:{}", prefix, &suffix[1..])
        } else {
            clean
        };
        if let Ok(alt_path) = self.config.safe_resolve_model_path(&alt) {
            let weights = alt_path.join("model.safetensors");
            if weights.exists() {
                return Some(weights);
            }
        }
        None
    }

    /// Delete an installed model from local storage
    pub fn delete_model(&self, model_name: &str) -> Result<bool, JepaError> {
        let clean_name = model_name.replace("%2F", "/").replace("%2f", "/");
        let safe_path = self.config.safe_resolve_model_path(&clean_name)?;
        let mut deleted = false;

        if safe_path.exists() {
            if safe_path.is_dir() {
                fs::remove_dir_all(&safe_path)?;
            } else {
                fs::remove_file(&safe_path)?;
            }
            tracing::info!("Deleted model storage: {}", safe_path.display());
            deleted = true;
        } else {
            // Also check alternate colon/slash representation
            let alt_name = if clean_name.contains(':') {
                clean_name.replace(':', "/")
            } else if let Some(last_slash) = clean_name.rfind('/') {
                let (prefix, suffix) = clean_name.split_at(last_slash);
                format!("{}:{}", prefix, &suffix[1..])
            } else {
                clean_name.clone()
            };
            if let Ok(alt_path) = self.config.safe_resolve_model_path(&alt_name) {
                if alt_path.exists() {
                    if alt_path.is_dir() {
                        fs::remove_dir_all(&alt_path)?;
                    } else {
                        fs::remove_file(&alt_path)?;
                    }
                    tracing::info!("Deleted model storage via alternate path: {}", alt_path.display());
                    deleted = true;
                }
            }
        }

        if deleted {
            // Prune empty parent directories up to models_dir (e.g. ~/.jepa/models/google/)
            if let Some(parent) = safe_path.parent() {
                if parent != self.config.models_dir && parent.starts_with(&self.config.models_dir) {
                    if let Ok(mut read) = fs::read_dir(parent) {
                        if read.next().is_none() {
                            let _ = fs::remove_dir(parent);
                        }
                    }
                }
            }
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Register a custom Jepafile manifest
    pub fn register_jepafile(&self, cfg: JepafileConfig) -> Result<ModelManifest, JepaError> {
        cfg.validate()?;
        let safe_path = self.config.safe_resolve_model_path(&cfg.name)?;
        fs::create_dir_all(&safe_path)?;

        let manifest_path = safe_path.join("Jepafile.json");
        cfg.save_to_file(&manifest_path)?;

        cfg.to_manifest(0)
    }

    /// Stream download a model from Hugging Face hub
    pub fn start_pull(self: Arc<Self>, repo_id: String) -> broadcast::Receiver<PullProgressEvent> {
        let (tx, rx) = broadcast::channel(128);
        let catalog = self.clone();

        tokio::spawn(async move {
            let target_dir = match catalog.config.safe_resolve_model_path(&repo_id) {
                Ok(p) => p,
                Err(e) => {
                    let _ = tx.send(PullProgressEvent {
                        repo_id: repo_id.clone(),
                        status: "error".to_string(),
                        downloaded_bytes: 0,
                        total_bytes: 0,
                        speed_mb_s: 0.0,
                        percentage: 0.0,
                        finished: true,
                        error: Some(e.to_string()),
                    });
                    return;
                }
            };

            // If verified model, generate manifest file in target directory
            if let Some(m) = get_verified_manifests().into_iter().find(|v| v.name == repo_id || v.repo_id == repo_id) {
                let jepafile = JepafileConfig::from(&m);
                let _ = fs::create_dir_all(&target_dir);
                let _ = jepafile.save_to_file(&target_dir.join("Jepafile.json"));
            }

            let _ = catalog.downloader.download_model(&repo_id, &target_dir, tx).await;
        });

        rx
    }
}

/// Recursively locate custom Jepafile manifests up to max depth
fn scan_manifests_recursive(dir: &Path, depth: usize, out: &mut Vec<PathBuf>) {
    if depth > 3 {
        return;
    }
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_dir() {
                let jepafile = p.join("Jepafile.json");
                if jepafile.exists() {
                    out.push(jepafile);
                }
                scan_manifests_recursive(&p, depth + 1, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelModality;

    #[test]
    fn test_delete_model_and_scan() {
        let test_root = std::env::temp_dir().join(format!("jepa_test_{}", uuid::Uuid::new_v4()));
        let models_dir = test_root.join("models");
        fs::create_dir_all(&models_dir).unwrap();

        let cfg = Arc::new(RuntimeConfig {
            home_dir: test_root.clone(),
            models_dir: models_dir.clone(),
            logs_dir: test_root.join("logs"),
            auth_token_path: test_root.join("auth.token"),
            keys_db_path: test_root.join("keys.json"),
            settings_path: test_root.join("settings.json"),
            gestures_path: test_root.join("gestures.json"),
            world_model_path: test_root.join("robot_world_model.json"),
            host: "127.0.0.1".to_string(),
            port: 11435,
            no_auth: true,
            cors_origins: Vec::new(),
        });

        let catalog = ModelCatalog::new(cfg);

        // Register custom model: google/test-custom-vit
        let custom_jepafile = JepafileConfig {
            name: "google/test-custom-vit".to_string(),
            repo_id: "google/test-custom-vit".to_string(),
            architecture: "ViT-B/16".to_string(),
            modality: ModelModality::Image,
            patch_size: 16,
            embed_dim: 768,
            num_layers: 12,
            num_heads: 12,
            image_size: 224,
            frames: None,
            parameter_count: Some("86M".to_string()),
            weights_file: Some("model.safetensors".to_string()),
            variant: None,
            normalization: None,
            mlp_ratio: None,
            tubelet_size: None,
            input_width: None,
            in_chans: None,
            audio: None,
            pooling: None,
        };

        catalog.register_jepafile(custom_jepafile).unwrap();

        // Check it is listed
        let installed = catalog.list_installed();
        assert!(installed.iter().any(|m| m.name == "google/test-custom-vit"));

        // Delete with slash
        let deleted = catalog.delete_model("google/test-custom-vit").unwrap();
        assert!(deleted);

        // Check it is no longer listed
        let remaining = catalog.list_installed();
        assert!(!remaining.iter().any(|m| m.name == "google/test-custom-vit"));

        // Clean up
        let _ = fs::remove_dir_all(&test_root);
    }
}
