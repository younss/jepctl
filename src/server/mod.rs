//! Axum server initialization, CORS, payload limits, and graceful shutdown.

pub mod gesture_handlers;
pub mod handlers;
pub mod middleware;
pub mod robot_handlers;
pub mod routes;
#[cfg(test)]
mod tests;
pub mod ui_assets;

use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;

use crate::config::MAX_VIDEO_PAYLOAD_SIZE;
use crate::server::handlers::AppState;
use crate::server::routes::create_router;
use crate::types::JepaError;

/// Initialize and launch Axum HTTP daemon
pub async fn start_daemon(state: AppState, host: &str, port: u16) -> Result<(), JepaError> {
    let addr_str = format!("{}:{}", host, port);
    let addr: SocketAddr =
        addr_str.parse().map_err(|e| JepaError::InvalidPayload(format!("Invalid socket address: {}", e)))?;

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("jepctl daemon listening on http://{}", addr);

    // Same-origin by default: the embedded testbench needs no CORS, and a permissive
    // policy would let any web page in the user's browser read the API (camera frames,
    // embeddings, the session token). Opt in per origin with `--cors-origins`.
    let cors = build_cors(&state.config.cors_origins);

    let app = create_router(state.clone())
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .layer(RequestBodyLimitLayer::new(MAX_VIDEO_PAYLOAD_SIZE))
        .layer(axum::middleware::from_fn({
            let audit_log = state.audit_log.clone();
            move |req, next| {
                let log = audit_log.clone();
                middleware::audit_middleware(req, next, log)
            }
        }));

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| JepaError::Io(std::io::Error::other(e)))?;

    tracing::info!("jepctl daemon shutdown cleanly.");
    Ok(())
}

/// CORS layer allowing only the configured origins (none → no cross-origin access).
fn build_cors(origins: &[String]) -> CorsLayer {
    if origins.is_empty() {
        return CorsLayer::new();
    }
    let parsed: Vec<axum::http::HeaderValue> = origins
        .iter()
        .filter_map(|o| match o.parse() {
            Ok(v) => Some(v),
            Err(_) => {
                tracing::warn!("Ignoring invalid CORS origin '{}'", o);
                None
            }
        })
        .collect();
    tracing::info!("CORS enabled for origins: {:?}", origins);
    CorsLayer::new()
        .allow_origin(parsed)
        .allow_methods(Any)
        .allow_headers([axum::http::header::AUTHORIZATION, axum::http::header::CONTENT_TYPE])
}

/// Wait for OS termination signal (SIGINT / Ctrl-C / SIGTERM)
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c().await.expect("Failed to install Ctrl+C signal handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received Ctrl-C shutdown signal.");
        },
        _ = terminate => {
            tracing::info!("Received SIGTERM shutdown signal.");
        },
    }
}
