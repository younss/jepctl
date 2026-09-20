//! Security middleware, Bearer token guard, and audit logging.

use axum::extract::Request;
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::Next;
use axum::response::{Json, Response};
use chrono::Utc;
use serde_json::json;
use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::RwLock;

use crate::auth::AuthManager;
use crate::types::{ApiKeyRecord, AuditLogEntry, Role};

/// In-memory rolling buffer for API audit trail
pub type SharedAuditLog = Arc<RwLock<VecDeque<AuditLogEntry>>>;

pub fn create_audit_log() -> SharedAuditLog {
    Arc::new(RwLock::new(VecDeque::with_capacity(100)))
}

/// Record an audit event into the rolling buffer (max 100 entries)
pub async fn record_audit_entry(
    audit_log: &SharedAuditLog,
    method: String,
    path: String,
    client_ip: String,
    status_code: u16,
    latency_ms: f64,
) {
    let mut lock = audit_log.write().await;
    if lock.len() >= 100 {
        lock.pop_back();
    }
    lock.push_front(AuditLogEntry { timestamp: Utc::now(), method, path, client_ip, status_code, latency_ms });
}

/// Extract and authenticate Bearer token with required RBAC role
pub async fn authenticate_request(
    headers: &HeaderMap,
    auth_manager: &AuthManager,
    required_role: Role,
) -> Result<ApiKeyRecord, (StatusCode, Json<serde_json::Value>)> {
    if auth_manager.no_auth_enabled {
        return Ok(ApiKeyRecord {
            key_prefix: "dev_local".to_string(),
            token_hash: "none".to_string(),
            role: Role::Admin,
            name: "Local Dev Bypass".to_string(),
            created_at: Utc::now(),
            expires_at: None,
        });
    }

    let auth_header = headers.get(axum::http::header::AUTHORIZATION).and_then(|h| h.to_str().ok());

    let token = match auth_header {
        Some(val) if val.starts_with("Bearer ") => &val[7..],
        _ => {
            return Err((
                StatusCode::UNAUTHORIZED,
                Json(json!({
                    "error": "Missing or malformed Authorization header. Expected: Bearer <token>"
                })),
            ));
        }
    };

    auth_manager
        .validate_token(token, required_role)
        .await
        .map_err(|e| (StatusCode::FORBIDDEN, Json(json!({ "error": e.to_string() }))))
}

/// Audit logging middleware layer
pub async fn audit_middleware(req: Request, next: Next, audit_log: SharedAuditLog) -> Response {
    let start = Instant::now();
    let method = req.method().to_string();
    let path = req.uri().path().to_string();
    let client_ip =
        req.headers().get("x-forwarded-for").and_then(|v| v.to_str().ok()).unwrap_or("127.0.0.1").to_string();

    let response = next.run(req).await;

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    let status_code = response.status().as_u16();

    // Do not flood audit log with static asset calls
    if path.starts_with("/api/") {
        record_audit_entry(&audit_log, method, path, client_ip, status_code, latency_ms).await;
    }

    response
}
