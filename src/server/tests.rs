//! HTTP integration tests: the full Axum router with an in-memory, randomly
//! initialised model. No weights, camera or network are required.

use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Instant;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use axum::Router;
use serde_json::{json, Value};
use tower::ServiceExt;

use crate::auth::AuthManager;
use crate::config::RuntimeConfig;
use crate::engine::device::select_device;
use crate::engine::EngineManager;
use crate::gestures::GestureStore;
use crate::hub::ModelCatalog;
use crate::media::capture::CameraSupervisor;
use crate::media::ring_buffer::RingBuffer;
use crate::server::handlers::AppState;
use crate::server::middleware::create_audit_log;
use crate::server::routes::create_router;
use crate::types::{ModelManifest, ModelModality};

fn test_manifest() -> ModelManifest {
    ModelManifest {
        name: "test/tiny-vit".into(),
        repo_id: "test/tiny-vit".into(),
        architecture: "tiny".into(),
        modality: ModelModality::Image,
        patch_size: 16,
        embed_dim: 32,
        num_layers: 1,
        num_heads: 4,
        image_size: 32,
        frames: None,
        parameter_count: "tiny".into(),
        disk_size_bytes: 0,
        weights_file: "model.safetensors".into(),
        created_at: chrono::Utc::now(),
        variant: None,
        normalization: None,
        mlp_ratio: None,
    }
}

struct TestApp {
    router: Router,
    state: AppState,
    _root: tempdir::Dir,
}

mod tempdir {
    pub struct Dir(pub std::path::PathBuf);
    impl Dir {
        pub fn new() -> Self {
            let p = std::env::temp_dir().join(format!("jepa_http_{}", uuid::Uuid::new_v4()));
            std::fs::create_dir_all(&p).unwrap();
            Self(p)
        }
    }
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }
}

async fn app(with_model: bool) -> TestApp {
    app_with(with_model, true).await
}

async fn app_with(with_model: bool, no_auth: bool) -> TestApp {
    let root = tempdir::Dir::new();
    let models_dir = root.0.join("models");
    std::fs::create_dir_all(&models_dir).unwrap();
    let config = Arc::new(RuntimeConfig {
        home_dir: root.0.clone(),
        models_dir,
        logs_dir: root.0.join("logs"),
        auth_token_path: root.0.join("auth.token"),
        keys_db_path: root.0.join("keys.json"),
        settings_path: root.0.join("settings.json"),
        gestures_path: root.0.join("gestures.json"),
        host: "127.0.0.1".into(),
        port: 0,
        no_auth,
        cors_origins: Vec::new(),
    });

    let (dev, hw) = select_device(Some("cpu"));
    let engine = EngineManager::new(dev, hw);
    if with_model {
        engine.load_random_for_test(test_manifest()).await.unwrap();
    }
    let ring_buffer = Arc::new(tokio::sync::RwLock::new(RingBuffer::new(4)));
    let state = AppState {
        engine,
        catalog: Arc::new(ModelCatalog::new(config.clone())),
        auth: AuthManager::init(&config.keys_db_path, &config.auth_token_path, no_auth).unwrap(),
        audit_log: create_audit_log(),
        camera_supervisor: Arc::new(CameraSupervisor::new(ring_buffer.clone())),
        ring_buffer,
        embeddings_total: Arc::new(AtomicU64::new(0)),
        gestures: Arc::new(tokio::sync::RwLock::new(GestureStore::load(&config.gestures_path))),
        start_time: Instant::now(),
        config,
    };
    TestApp { router: create_router(state.clone()), state, _root: root }
}

async fn call(router: &Router, method: &str, uri: &str, body: Option<Value>) -> (StatusCode, Value) {
    let mut req = Request::builder().method(method).uri(uri);
    let body = match body {
        Some(v) => {
            req = req.header("content-type", "application/json");
            Body::from(v.to_string())
        }
        None => Body::empty(),
    };
    let res = router.clone().oneshot(req.body(body).unwrap()).await.unwrap();
    let status = res.status();
    let bytes = to_bytes(res.into_body(), 1 << 20).await.unwrap();
    let json = if bytes.is_empty() { Value::Null } else { serde_json::from_slice(&bytes).unwrap_or(Value::Null) };
    (status, json)
}

fn unit(i: usize, dim: usize) -> Vec<f32> {
    let mut v = vec![0.0; dim];
    v[i] = 1.0;
    v
}

#[tokio::test]
async fn status_reports_model_and_weights() {
    let t = app(true).await;
    let (status, body) = call(&t.router, "GET", "/api/status", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["active_model"], "test/tiny-vit");
    assert_eq!(body["weights"]["source"], "random");
    assert_eq!(body["weights"]["loaded"], 0);
}

#[tokio::test]
async fn inference_without_a_model_is_a_clear_409() {
    let t = app(false).await;
    let (status, body) = call(&t.router, "POST", "/api/gestures/match", Some(json!({ "embedding": [1.0, 0.0] }))).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert!(body["error"].as_str().unwrap().contains("No model loaded"));
}

#[tokio::test]
async fn gesture_lifecycle_register_match_delete() {
    let t = app(true).await;
    let r = &t.router;

    // Register two gestures and a neutral pose from raw embeddings (two samples for the first).
    for (name, emb, neutral) in [
        ("open", unit(0, 32), false),
        ("open", unit(0, 32), false),
        ("fist", unit(1, 32), false),
        ("rest", unit(2, 32), true),
    ] {
        let (status, body) =
            call(r, "POST", "/api/gestures", Some(json!({ "name": name, "embedding": emb, "is_neutral": neutral })))
                .await;
        assert_eq!(status, StatusCode::CREATED, "{body}");
        assert_eq!(body["model_name"], "test/tiny-vit");
    }

    let (status, list) = call(r, "GET", "/api/gestures", None).await;
    assert_eq!(status, StatusCode::OK);
    let items = list.as_array().unwrap();
    assert_eq!(items.len(), 3);
    let open = items.iter().find(|g| g["name"] == "open").unwrap();
    assert_eq!(open["sample_count"], 2);
    assert_eq!(open["dimension"], 32);
    assert!(items.iter().any(|g| g["name"] == "rest" && g["is_neutral"] == true));

    // Registry is persisted.
    assert!(t.state.config.gestures_path.exists());

    // A clean "open" input is detected with a full decision trace.
    let (status, m) =
        call(r, "POST", "/api/gestures/match", Some(json!({ "embedding": unit(0, 32), "threshold": 0.5 }))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(m["matched"], "open", "{m}");
    assert_eq!(m["detected"], true);
    assert_eq!(m["method"], "contrastive");
    assert_eq!(m["scores"][0]["name"], "open");
    assert!(m["scores"][0]["raw_cosine"].as_f64().unwrap() > 0.99);
    assert!(m["reason"].as_str().unwrap().contains("open"));

    // The neutral pose wins → never a detection.
    let (_, m) =
        call(r, "POST", "/api/gestures/match", Some(json!({ "embedding": unit(2, 32), "threshold": 0.5 }))).await;
    assert_eq!(m["detected"], false);
    assert!(m["matched"].is_null());
    assert!(m["reason"].as_str().unwrap().contains("neutral"));

    // Wrong dimension is rejected, not silently ignored.
    let (status, body) =
        call(r, "POST", "/api/gestures", Some(json!({ "name": "open", "embedding": [1.0, 2.0] }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    // Delete one, then clear the rest.
    let (status, _) = call(r, "DELETE", "/api/gestures/fist", None).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = call(r, "DELETE", "/api/gestures/fist", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, body) = call(r, "DELETE", "/api/gestures", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["removed"], 2);
    let (_, list) = call(r, "GET", "/api/gestures", None).await;
    assert!(list.as_array().unwrap().is_empty());
}

#[tokio::test]
async fn gesture_validation_errors() {
    let t = app(true).await;
    let (status, _) = call(&t.router, "POST", "/api/gestures", Some(json!({ "name": "  ", "embedding": [1.0] }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, body) = call(&t.router, "POST", "/api/gestures", Some(json!({ "name": "x" }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("from_camera"));
    let (status, _) =
        call(&t.router, "POST", "/api/gestures", Some(json!({ "name": "x", "image_base64": "!!!" }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn camera_endpoints_without_frames() {
    let t = app(true).await;
    let (status, _) = call(&t.router, "GET", "/api/camera/frame", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, body) =
        call(&t.router, "POST", "/api/gestures", Some(json!({ "name": "x", "from_camera": true }))).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
}

#[tokio::test]
async fn from_camera_uses_the_ring_buffer_frame() {
    let t = app(true).await;
    t.state.ring_buffer.write().await.push_frame(image::RgbImage::from_pixel(64, 48, image::Rgb([200, 30, 30])), 1);

    let res = t.router.clone().oneshot(Request::get("/api/camera/frame").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert_eq!(res.headers()["content-type"], "image/jpeg");
    assert_eq!(res.headers()["x-frame-sequence"], "1");

    let (status, body) =
        call(&t.router, "POST", "/api/gestures", Some(json!({ "name": "cam", "from_camera": true }))).await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["has_patch_prototype"], true);
    assert!(body["thumbnail"].as_str().unwrap().starts_with("data:image/jpeg;base64,"));

    let (_, m) =
        call(&t.router, "POST", "/api/gestures/match", Some(json!({ "from_camera": true, "threshold": 0.5 }))).await;
    assert_eq!(m["matched"], "cam", "{m}");
    assert_eq!(m["grid_size"], 2);
    assert_eq!(m["patch_diff"].as_array().unwrap().len(), 4);
}

#[tokio::test]
async fn embed_rejects_video_and_garbage() {
    let t = app(true).await;
    let boundary = "XBOUNDARY";
    let body = format!(
        "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.bin\"\r\n\r\nthis is not an image at all\r\n--{boundary}--\r\n"
    );
    let req = Request::post("/api/embed")
        .header("content-type", format!("multipart/form-data; boundary={boundary}"))
        .body(Body::from(body))
        .unwrap();
    let res = t.router.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::UNSUPPORTED_MEDIA_TYPE);
}

#[tokio::test]
async fn unknown_model_load_is_404_and_missing_weights_409() {
    let t = app(false).await;
    let (status, _) = call(&t.router, "POST", "/api/models/load", Some(json!({ "model_name": "nope/none" }))).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, body) =
        call(&t.router, "POST", "/api/models/load", Some(json!({ "model_name": "facebook/dinov2-small" }))).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    assert!(body["error"].as_str().unwrap().contains("jepa pull"));
}

#[tokio::test]
async fn auth_mode_protects_camera_stream_and_token_bootstrap() {
    let t = app_with(true, false).await;
    let r = &t.router;
    let admin_token = std::fs::read_to_string(&t.state.config.auth_token_path).unwrap().trim().to_string();

    // Camera control, frames and gestures need a bearer token.
    for (method, uri) in [
        ("POST", "/api/camera/start?device=0"),
        ("POST", "/api/camera/stop"),
        ("GET", "/api/ring-buffer"),
        ("GET", "/api/camera/frame"),
        ("GET", "/api/gestures"),
        ("GET", "/api/embed/stream"),
    ] {
        let (status, _) = call(r, method, uri, None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED, "{method} {uri}");
    }

    // The SSE endpoint accepts the token as a query parameter (EventSource cannot set headers).
    let res = r
        .clone()
        .oneshot(Request::get(format!("/api/embed/stream?token={admin_token}")).body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    assert!(res.headers()["content-type"].to_str().unwrap().starts_with("text/event-stream"));

    // A bearer header works everywhere else.
    let res = r
        .clone()
        .oneshot(
            Request::get("/api/gestures")
                .header("authorization", format!("Bearer {admin_token}"))
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);

    // Session bootstrap: same-origin only.
    let res = r
        .clone()
        .oneshot(Request::get("/api/auth/token").header("sec-fetch-site", "cross-site").body(Body::empty()).unwrap())
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::FORBIDDEN);
    let res = r
        .clone()
        .oneshot(
            Request::get("/api/auth/token")
                .header("host", "127.0.0.1:11435")
                .header("sec-fetch-site", "same-origin")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(res.status(), StatusCode::OK);
}
