//! Authentication, RBAC authorization, and constant-time validation.

use chrono::{Duration, Utc};
use sha2::{Digest, Sha256};
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
        if keys_file.exists()
            && let Ok(data) = fs::read_to_string(keys_file)
            && let Ok(loaded) = serde_json::from_str::<HashMap<String, ApiKeyRecord>>(&data)
        {
            // Files written before 0.3.1 were keyed by the raw token; hash them on load.
            for (key, mut record) in loaded {
                let digest = if key.starts_with("jepa_sec_") { Self::token_digest(&key) } else { key };
                record.token_hash = digest.clone();
                map.insert(digest, record);
            }
        }

        // 2. Check or create ~/.jepctl/auth.token (default admin token)
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
        let admin_digest = Self::token_digest(&admin_token);
        let admin_record = ApiKeyRecord {
            key_prefix: admin_prefix,
            token_hash: admin_digest.clone(),
            role: Role::Admin,
            name: "Default Admin Key".to_string(),
            created_at: Utc::now(),
            expires_at: None,
        };
        map.insert(admin_digest, admin_record);

        let manager =
            Arc::new(Self { tokens: RwLock::new(map), keys_file: keys_file.to_path_buf(), no_auth_enabled: no_auth });

        // Persist back
        let _ = manager.persist_sync();

        Ok(manager)
    }

    /// SHA-256 of a token, hex encoded. Tokens are never stored in clear: the map and
    /// `keys.json` are keyed by this digest, so a leaked file does not leak the keys.
    pub fn token_digest(raw_token: &str) -> String {
        hex::encode(Sha256::digest(raw_token.as_bytes()))
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

        let digest = Self::token_digest(&raw_token);
        let record = ApiKeyRecord {
            key_prefix: key_prefix.clone(),
            token_hash: digest.clone(),
            role: req.role,
            name: req.name.clone(),
            created_at: now,
            expires_at,
        };

        {
            let mut lock = self.tokens.write().await;
            lock.insert(digest, record);
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

        // Compare digests, in constant time, against every stored digest.
        let candidate = Self::token_digest(candidate_token);
        let cand_bytes = candidate.as_bytes();
        let lock = self.tokens.read().await;

        let mut matched_record: Option<ApiKeyRecord> = None;

        for (stored_digest, record) in lock.iter() {
            let stored_bytes = stored_digest.as_bytes();
            if stored_bytes.len() == cand_bytes.len() && bool::from(stored_bytes.ct_eq(cand_bytes)) {
                matched_record = Some(record.clone());
                break;
            }
        }

        let record = matched_record
            .ok_or_else(|| JepaError::AuthError("Invalid or missing Bearer authorization token".to_string()))?;

        // Verify expiration
        if let Some(exp) = record.expires_at
            && Utc::now() > exp
        {
            return Err(JepaError::AuthError("API key has expired".to_string()));
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
        if let Ok(lock) = self.tokens.try_read()
            && let Ok(data) = serde_json::to_string_pretty(&*lock)
        {
            let _ = fs::write(&self.keys_file, data);
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = fs::set_permissions(&self.keys_file, fs::Permissions::from_mode(0o600));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CreateKeyRequest;

    fn temp_dir() -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!("jepctl_auth_{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[tokio::test]
    async fn tokens_are_stored_as_digests_and_legacy_files_migrate() {
        let dir = temp_dir();
        let keys = dir.join("keys.json");
        let token_file = dir.join("auth.token");
        let auth = AuthManager::init(&keys, &token_file, false).unwrap();
        let admin = fs::read_to_string(&token_file).unwrap();
        let created = auth
            .create_key(CreateKeyRequest { name: "ci".into(), role: Role::Inference, expire_days: None })
            .await
            .unwrap();

        // Both tokens validate; a wrong one does not.
        assert_eq!(auth.validate_token(&admin, Role::Admin).await.unwrap().role, Role::Admin);
        assert_eq!(auth.validate_token(&created.raw_token, Role::Inference).await.unwrap().name, "ci");
        assert!(auth.validate_token(&created.raw_token, Role::Admin).await.is_err());
        assert!(auth.validate_token("jepa_sec_nope", Role::Inference).await.is_err());

        // The file never contains a raw token, only digests.
        let on_disk = fs::read_to_string(&keys).unwrap();
        assert!(!on_disk.contains(&admin) && !on_disk.contains(&created.raw_token), "{on_disk}");
        assert!(on_disk.contains(&AuthManager::token_digest(&created.raw_token)));
        assert!(!on_disk.contains("token_hash"));
        for r in auth.list_keys().await {
            assert!(!serde_json::to_string(&r).unwrap().contains("token_hash"));
        }

        // A pre 0.3.1 file keyed by raw tokens is migrated on load and keeps working.
        let legacy = format!(
            r#"{{"jepa_sec_{}": {{"key_prefix":"jepa_sec_abc","token_hash":"hash_x","role":"inference","name":"old","created_at":"2026-01-01T00:00:00Z","expires_at":null}}}}"#,
            "ab".repeat(32)
        );
        fs::write(&keys, legacy).unwrap();
        let auth2 = AuthManager::init(&keys, &token_file, false).unwrap();
        let old_token = format!("jepa_sec_{}", "ab".repeat(32));
        assert_eq!(auth2.validate_token(&old_token, Role::Inference).await.unwrap().name, "old");
        assert!(!fs::read_to_string(&keys).unwrap().contains(&old_token));
        let _ = fs::remove_dir_all(&dir);
    }
}
