//! Axum server initialization, CORS, payload limits, and graceful shutdown.

pub mod gesture_handlers;
pub mod handlers;
pub mod middleware;
pub mod routes;
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
    let addr: SocketAddr = addr_str
        .parse()
        .map_err(|e| JepaError::InvalidPayload(format!("Invalid socket address: {}", e)))?;

    let listener = tokio::net::TcpListener::bind(addr).await?;
    tracing::info!("JEPA daemon listening on http://{}", addr);

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

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
        .map_err(|e| JepaError::Io(std::io::Error::new(std::io::ErrorKind::Other, e)))?;

    tracing::info!("JEPA daemon shutdown cleanly.");
    Ok(())
}

/// Wait for OS termination signal (SIGINT / Ctrl-C / SIGTERM)
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C signal handler");
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
