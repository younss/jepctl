//! Embedded static Web UI assets compiled directly into the single binary distribution.

use axum::http::header;
use axum::response::{IntoResponse, Response};

pub const INDEX_HTML: &str = include_str!("../ui/index.html");
pub const STYLES_CSS: &str = include_str!("../ui/styles.css");
pub const APP_JS: &str = include_str!("../ui/app.js");

/// Serve index.html Single-Page Application
pub async fn serve_index() -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/html; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache, no-store, must-revalidate"),
        ],
        INDEX_HTML,
    )
        .into_response()
}

/// Serve dark-mode CSS stylesheet
pub async fn serve_styles() -> Response {
    (
        [
            (header::CONTENT_TYPE, "text/css; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache, no-store, must-revalidate"),
        ],
        STYLES_CSS,
    )
        .into_response()
}

/// Serve reactive JavaScript client logic
pub async fn serve_app_js() -> Response {
    (
        [
            (header::CONTENT_TYPE, "application/javascript; charset=utf-8"),
            (header::CACHE_CONTROL, "no-cache, no-store, must-revalidate"),
        ],
        APP_JS,
    )
        .into_response()
}
