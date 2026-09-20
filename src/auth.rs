//! Authentication, RBAC authorization, and constant-time validation.

use chrono::{Duration, Utc};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use subtle::ConstantTimeEq;
use tokio::sync::RwLock;

use crate::types::{ApiKeyRecord, CreateKeyRequest, CreateKeyResponse, JepaError, Role};

/// In-memory and persisted API Key Store with constant-time lookup
#[derive(Debug)]
pub struct AuthManager {
    /// In-memory map from full token string to ApiKeyRecord
    tokens: RwLock<HashMap<String, ApiKeyRecord>>,
    keys_file: std::path::PathBuf,
    pub no_auth_enabled: bool,
}

impl AuthManager {
    /// Initialize auth manager, load persisted keys, and guarantee initial admin token
    pub fn init(keys_file: &Path, auth_token_path: &Path, no_auth: bool) -> Result<Arc<Self>, JepaError> {
        let mut map = HashMap::new();

        // 1. Load keys from keys.json if it exists
        if keys_file.exists() {
            if let Ok(data) = fs::read_to_string(keys_file) {
                if let Ok(loaded) = serde_json::from_str::<HashMap<String, ApiKeyRecord>>(&data) {
                    map = loaded;
                }
            }
        }

        // 2. Check or create ~/.jepa/auth.token (default admin token)
        let admin_token = if auth_token_path.exists() {
            fs::read_to_string(auth_token_path)
                .map(|s| s.trim().to_string())
                .unwrap_or_else(|_| Self::generate_token_string())
        } else {
            let token = Self::generate_token_string();
            let mut file = File::create(auth_token_path)?;
            file.write_all(token.as_bytes())?;

            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = fs::set_permissions(auth_token_path, fs::Permissions::from_mode(0o600));
            }
            token
        };

        // Ensure default admin token is in map
        let admin_prefix = admin_token.chars().take(12).collect::<String>();
        let admin_record = ApiKeyRecord {
            key_prefix: admin_prefix,
            token_hash: "default_root_key".to_string(),
            role: Role::Admin,
            name: "Default Admin Key".to_string(),
            created_at: Utc::now(),
            expires_at: None,
        };
        map.insert(admin_token.clone(), admin_record);

        let manager =
            Arc::new(Self { tokens: RwLock::new(map), keys_file: keys_file.to_path_buf(), no_auth_enabled: no_auth });

        // Persist back
        let _ = manager.persist_sync();

        Ok(manager)
    }

    /// Generate a cryptographically secure random token string
    pub fn generate_token_string() -> String {
        let bytes = rand::random::<[u8; 32]>();
        format!("jepa_sec_{}", hex::encode(bytes))
    }

    /// Create and register a new scoped Bearer token
    pub async fn create_key(&self, req: CreateKeyRequest) -> Result<CreateKeyResponse, JepaError> {
        let raw_token = Self::generate_token_string();
        let key_prefix = raw_token.chars().take(12).collect::<String>();
        let now = Utc::now();
        let expires_at = req.expire_days.map(|d| now + Duration::days(d));

        let record = ApiKeyRecord {
            key_prefix: key_prefix.clone(),
            token_hash: format!("hash_{}", &raw_token[9..25]),
            role: req.role,
            name: req.name.clone(),
            created_at: now,
            expires_at,
        };

        {
            let mut lock = self.tokens.write().await;
            lock.insert(raw_token.clone(), record);
        }

        self.persist().await?;

        Ok(CreateKeyResponse { key_prefix, raw_token, role: req.role, name: req.name, created_at: now })
    }

    /// List all registered key records
    pub async fn list_keys(&self) -> Vec<ApiKeyRecord> {
        let lock = self.tokens.read().await;
        lock.values().cloned().collect()
    }

    /// Revoke a key by its prefix
    pub async fn revoke_key(&self, prefix: &str) -> Result<bool, JepaError> {
        let mut found = false;
        {
            let mut lock = self.tokens.write().await;
            let to_remove: Vec<String> =
                lock.iter().filter(|(_, rec)| rec.key_prefix == prefix).map(|(k, _)| k.clone()).collect();

            for k in to_remove {
                lock.remove(&k);
                found = true;
            }
        }

        if found {
            self.persist().await?;
        }

        Ok(found)
    }

    /// Constant-time token validation and role authorization check
    pub async fn validate_token(&self, candidate_token: &str, required_role: Role) -> Result<ApiKeyRecord, JepaError> {
        if self.no_auth_enabled {
            return Ok(ApiKeyRecord {
                key_prefix: "no_auth".to_string(),
                token_hash: "no_auth".to_string(),
                role: Role::Admin,
                name: "Dev Loopback Bypass".to_string(),
                created_at: Utc::now(),
                expires_at: None,
            });
        }

        let cand_bytes = candidate_token.as_bytes();
        let lock = self.tokens.read().await;

        let mut matched_record: Option<ApiKeyRecord> = None;

        for (stored_token, record) in lock.iter() {
            let stored_bytes = stored_token.as_bytes();
            if stored_bytes.len() == cand_bytes.len() {
                // Perform constant-time slice comparison to prevent timing attacks
                if stored_bytes.ct_eq(cand_bytes).into() {
                    matched_record = Some(record.clone());
                    break;
                }
            }
        }

        let record = matched_record
            .ok_or_else(|| JepaError::AuthError("Invalid or missing Bearer authorization token".to_string()))?;

        // Verify expiration
        if let Some(exp) = record.expires_at {
            if Utc::now() > exp {
                return Err(JepaError::AuthError("API key has expired".to_string()));
            }
        }

        // Role-based access control (RBAC) validation
        match required_role {
            Role::Inference => {
                // Both Admin and Inference roles can access inference endpoints
                Ok(record)
            }
            Role::Admin => {
                if record.role == Role::Admin {
                    Ok(record)
                } else {
                    Err(JepaError::Forbidden("Endpoint requires administrative privileges".to_string()))
                }
            }
        }
    }

    /// Async persistence of registered keys
    async fn persist(&self) -> Result<(), JepaError> {
        let lock = self.tokens.read().await;
        let data = serde_json::to_string_pretty(&*lock)?;
        fs::write(&self.keys_file, data)?;

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let _ = fs::set_permissions(&self.keys_file, fs::Permissions::from_mode(0o600));
        }

        Ok(())
    }

    /// Synchronous persistence on initialization
    fn persist_sync(&self) -> Result<(), JepaError> {
        if let Ok(lock) = self.tokens.try_read() {
            if let Ok(data) = serde_json::to_string_pretty(&*lock) {
                let _ = fs::write(&self.keys_file, data);
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    let _ = fs::set_permissions(&self.keys_file, fs::Permissions::from_mode(0o600));
                }
            }
        }
        Ok(())
    }
}
