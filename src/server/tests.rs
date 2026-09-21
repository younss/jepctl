//! HTTP integration tests: the full Axum router with an in-memory, randomly
//! initialised model. No weights, camera or network are required.

use std::sync::atomic::AtomicU64;
use std::sync::Arc;
use std::time::Instant;

use axum::body::{to_bytes, Body};
use axum::http::{Request, StatusCode};
use axum::Router;
use base64::Engine;
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
        tubelet_size: None,
        input_width: None,
        in_chans: None,
        audio: None,
        pooling: None,
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

fn audio_manifest() -> ModelManifest {
    ModelManifest {
        name: "test/audio-tiny".into(),
        modality: ModelModality::Audio,
        image_size: 64,
        input_width: Some(32),
        in_chans: Some(1),
        audio: Some(crate::types::AudioSpec { sample_rate: 16_000, n_mels: 32, frames: 64, mean: 0.0, std: 1.0 }),
        variant: Some(crate::engine::vit::VitVariant::Cls),
        pooling: Some(crate::engine::vit::Pooling::Mean),
        ..test_manifest()
    }
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
        world_model_path: root.0.join("robot_world_model.json"),
        sounds_path: root.0.join("sounds.json"),
        companion_path: root.0.join("companion.json"),
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
        sounds: Arc::new(tokio::sync::RwLock::new(GestureStore::load(&config.sounds_path))),
        mic: Arc::new(crate::media::mic::MicSupervisor::new()),
        companion: crate::companion::CompanionHandle::new(),
        camera_roi: Arc::new(tokio::sync::RwLock::new(None)),
        robot: {
            let r = crate::robot::RobotHandle::new(Default::default());
            r.core.lock().await.backend.connect().unwrap();
            r
        },
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
    assert!(body["error"].as_str().unwrap().contains("jepctl pull"));
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

#[tokio::test]
async fn gesture_bundle_export_import_over_http() {
    let t = app(true).await;
    let r = &t.router;
    for (name, i) in [("a", 0), ("b", 1)] {
        let (status, _) =
            call(r, "POST", "/api/gestures", Some(json!({ "name": name, "embedding": unit(i, 32) }))).await;
        assert_eq!(status, StatusCode::CREATED);
    }

    let (status, bundle) = call(r, "GET", "/api/gestures/export?threshold=0.6&thumbnails=false", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(bundle["version"], 1);
    assert_eq!(bundle["threshold"], 0.6);
    assert_eq!(bundle["gestures"].as_array().unwrap().len(), 2);
    assert!(bundle["gestures"][0]["thumbnail"].is_null());

    // Fresh instance, import with replace, then match works immediately.
    let t2 = app(true).await;
    let (status, report) = call(&t2.router, "POST", "/api/gestures/import?replace=true", Some(bundle.clone())).await;
    assert_eq!(status, StatusCode::OK, "{report}");
    assert_eq!(report["imported"], 2);
    let (_, m) =
        call(&t2.router, "POST", "/api/gestures/match", Some(json!({ "embedding": unit(1, 32), "threshold": 0.5 })))
            .await;
    assert_eq!(m["matched"], "b", "{m}");

    // Corrupt bundle → 400
    let mut bad = bundle;
    bad["gestures"][0]["dimension"] = json!(3);
    let (status, _) = call(&t2.router, "POST", "/api/gestures/import", Some(bad)).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn embed_accepts_gif_clips() {
    let t = app(true).await;
    // 6-frame 32x32 GIF built in memory (same helper as media::video tests).
    let gif = {
        use image::codecs::gif::GifEncoder;
        use image::{Delay, Frame, Rgba, RgbaImage};
        let mut buf = Vec::new();
        let mut enc = GifEncoder::new(&mut buf);
        for i in 0..6u32 {
            let img = RgbaImage::from_fn(32, 32, |x, _| {
                if (x / 8 + i) % 2 == 0 {
                    Rgba([255, 255, 255, 255])
                } else {
                    Rgba([0, 0, 0, 255])
                }
            });
            enc.encode_frame(Frame::from_parts(img, 0, 0, Delay::from_numer_denom_ms(100, 1))).unwrap();
        }
        drop(enc);
        buf
    };
    let boundary = "XBOUNDARY";
    let mut body = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.gif\"\r\nContent-Type: image/gif\r\n\r\n").into_bytes();
    body.extend_from_slice(&gif);
    body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
    let req = Request::post("/api/embed")
        .header("content-type", format!("multipart/form-data; boundary={boundary}"))
        .body(Body::from(body))
        .unwrap();
    let res = t.router.clone().oneshot(req).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = to_bytes(res.into_body(), 1 << 20).await.unwrap();
    let json: Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["dimension"], 32);
    assert_eq!(json["embedding"].as_array().unwrap().len(), 32);
    // Image model over a clip: mean of frames, no spatial tokens.
    assert!(json["patch_embeddings"].is_null());
}

#[tokio::test]
async fn roi_changes_what_the_model_sees_and_travels_in_bundles() {
    let t = app(true).await;
    let r = &t.router;
    // Left half red, right half green.
    let img =
        image::RgbImage::from_fn(
            200,
            100,
            |x, _| if x < 100 { image::Rgb([220, 20, 20]) } else { image::Rgb([20, 220, 20]) },
        );
    t.state.ring_buffer.write().await.push_frame(img, 1);

    let (status, body) = call(r, "GET", "/api/camera/roi", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["roi"].is_null());

    let (status, _) = call(r, "PUT", "/api/camera/roi", Some(json!({ "x": 0.5, "y": 0.0, "w": 0.5, "h": 1.0 }))).await;
    assert_eq!(status, StatusCode::OK);
    let (status, body) =
        call(r, "PUT", "/api/camera/roi", Some(json!({ "x": 0.5, "y": 0.5, "w": 0.001, "h": 0.001 }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");

    // The model view is now the green half.
    let res = r.clone().oneshot(Request::get("/api/camera/frame").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let bytes = to_bytes(res.into_body(), 1 << 20).await.unwrap();
    let decoded = image::load_from_memory(&bytes).unwrap().to_rgb8();
    assert!(decoded.pixels().all(|p| p[1] > 150 && p[0] < 80));

    // Full frame for the editor keeps its aspect ratio and reports the source size.
    let res =
        r.clone().oneshot(Request::get("/api/camera/frame?full=true").body(Body::empty()).unwrap()).await.unwrap();
    assert_eq!(res.headers()["x-frame-width"], "200");
    assert_eq!(res.headers()["x-frame-height"], "100");

    // ROI is recorded in the bundle and restored on import; it is kept across settings saves.
    let (_, bundle) = call(r, "GET", "/api/gestures/export?thumbnails=false", None).await;
    assert_eq!(bundle["roi"]["x"], 0.5);
    let (_, settings) = call(r, "GET", "/api/settings", None).await;
    let (status, _) = call(r, "POST", "/api/settings", Some(settings)).await;
    assert_eq!(status, StatusCode::OK);
    assert!(t.state.config.load_settings().camera_roi.is_some());

    let (status, _) = call(r, "DELETE", "/api/camera/roi", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(t.state.camera_roi.read().await.is_none());
    let (status, _) = call(r, "POST", "/api/gestures/import", Some(bundle)).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(t.state.camera_roi.read().await.map(|r| r.w), Some(0.5));
}

#[tokio::test]
async fn embed_accepts_wav_with_an_audio_model_and_rejects_images() {
    let t = app(false).await;
    t.state.engine.load_random_audio_for_test(audio_manifest()).await.unwrap();

    // 0.5 s of a 440 Hz tone, PCM16 mono 16 kHz.
    let mut data = Vec::new();
    for i in 0..8000u32 {
        let v = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin() * 0.5;
        data.extend_from_slice(&((v * 32767.0) as i16).to_le_bytes());
    }
    let mut wav = Vec::new();
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&((36 + data.len()) as u32).to_le_bytes());
    wav.extend_from_slice(b"WAVEfmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes());
    wav.extend_from_slice(&16_000u32.to_le_bytes());
    wav.extend_from_slice(&32_000u32.to_le_bytes());
    wav.extend_from_slice(&2u16.to_le_bytes());
    wav.extend_from_slice(&16u16.to_le_bytes());
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&(data.len() as u32).to_le_bytes());
    wav.extend_from_slice(&data);

    let multipart = |name: &str, ctype: &str, bytes: &[u8]| {
        let boundary = "XBOUNDARY";
        let mut body = format!("--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{name}\"\r\nContent-Type: {ctype}\r\n\r\n").into_bytes();
        body.extend_from_slice(bytes);
        body.extend_from_slice(format!("\r\n--{boundary}--\r\n").as_bytes());
        Request::post("/api/embed")
            .header("content-type", format!("multipart/form-data; boundary={boundary}"))
            .body(Body::from(body))
            .unwrap()
    };

    let res = t.router.clone().oneshot(multipart("tone.wav", "audio/wav", &wav)).await.unwrap();
    assert_eq!(res.status(), StatusCode::OK);
    let json: Value = serde_json::from_slice(&to_bytes(res.into_body(), 1 << 20).await.unwrap()).unwrap();
    assert_eq!(json["model"], "test/audio-tiny");
    assert_eq!(json["embedding"].as_array().unwrap().len(), 32);
    assert_eq!(json["patch_embeddings"].as_array().unwrap().len(), (64 / 16) * (32 / 16));

    // The audio model lives in its own slot: an image still needs a vision model.
    let png = [0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0, 0, 0, 0x0D];
    let res = t.router.clone().oneshot(multipart("a.png", "image/png", &png)).await.unwrap();
    assert_ne!(res.status(), StatusCode::OK);
    let (status, st) = call(&t.router, "GET", "/api/status", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(st["audio_model"], "test/audio-tiny");
    assert!(st["active_model"].is_null());
}

#[tokio::test]
async fn sounds_and_companion_learn_from_the_microphone() {
    let t = app(true).await;
    let r = &t.router;
    t.state.engine.load_random_audio_for_test(audio_manifest()).await.unwrap();

    // Nothing captured yet: registering a sound is a clear 409.
    let (status, body) = call(r, "POST", "/api/sounds", Some(json!({ "name": "clap" }))).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");

    // Feed the microphone ring directly (no device in CI) and mark it active.
    let tone: Vec<f32> = (0..16_000).map(|i| ((i as f32) * 0.2).sin() * 0.4).collect();
    t.state.mic.push_samples(&tone, 16_000);
    let (status, mic) = call(r, "GET", "/api/mic/status", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(mic["buffered_seconds"].as_f64().unwrap() > 0.9);
    assert!(mic["level"].as_f64().unwrap() > 0.1);
    t.state.mic.force_active_for_test(true);

    let (status, body) = call(r, "POST", "/api/sounds", Some(json!({ "name": "clap", "seconds": 0.5 }))).await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["model_name"], "test/audio-tiny");
    let (status, list) = call(r, "GET", "/api/sounds", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(list.as_array().unwrap().len(), 1);
    let (status, m) = call(r, "POST", "/api/sounds/match", Some(json!({ "seconds": 0.5 }))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(m["scores"].as_array().unwrap().len(), 1);

    // Companion: teach a sound cue and a pose cue, then check the memory.
    let (status, c) = call(r, "GET", "/api/companion/status", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(c["mode"], "manual");
    let (status, body) =
        call(r, "POST", "/api/companion/teach", Some(json!({ "kind": "sound", "name": "clap", "behaviour": "nod" })))
            .await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["cue"]["behaviour"], "nod");
    assert_eq!(body["sample_count"], 2);
    assert_eq!(body["waveform"].as_array().unwrap().len(), 120);
    assert!(body["level"].as_f64().unwrap() > 0.3);
    let (status, wf) = call(r, "GET", "/api/mic/waveform?points=40", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(wf["points"].as_array().unwrap().len(), 40);
    let (_, c) = call(r, "GET", "/api/companion/status", None).await;
    assert_eq!(c["animation"], "acknowledge");
    let (status, body) = call(
        r,
        "POST",
        "/api/companion/pose",
        Some(
            json!({ "head_pan": 0.0, "head_tilt": 0.0, "left_arm": 1.0, "right_arm": -0.5, "lean": 0.0, "mood": 0.5 }),
        ),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    // A pose cue needs a camera frame (the registry pipeline is the gesture one).
    let (status, body) =
        call(r, "POST", "/api/companion/teach", Some(json!({ "kind": "gesture", "name": "arm up" }))).await;
    assert_eq!(status, StatusCode::CONFLICT, "{body}");
    t.state.ring_buffer.write().await.push_frame(image::RgbImage::from_pixel(64, 48, image::Rgb([40, 120, 200])), 1);
    let (status, body) =
        call(r, "POST", "/api/companion/teach", Some(json!({ "kind": "gesture", "name": "arm up" }))).await;
    assert_eq!(status, StatusCode::CREATED, "{body}");
    assert_eq!(body["cue"]["behaviour"], "pose");
    assert_eq!(body["cue"]["pose"]["left_arm"], 1.0);
    assert!(body["thumbnail"].as_str().unwrap().starts_with("data:image/jpeg;base64,"));
    let (status, cues) = call(r, "GET", "/api/companion/cues", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(cues.as_array().unwrap().len(), 2);
    assert!(t.state.config.companion_path.is_file());

    let (status, _) = call(r, "POST", "/api/companion/mode", Some(json!({ "mode": "interactive" }))).await;
    assert_eq!(status, StatusCode::OK);
    let (status, body) = call(r, "POST", "/api/companion/behaviour", Some(json!({ "behaviour": "dance" }))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["animation"], "dance");
    let (status, _) = call(r, "DELETE", "/api/companion/cues/sound/clap", None).await;
    assert_eq!(status, StatusCode::OK);
    let (status, _) = call(r, "DELETE", "/api/companion/cues/sound/clap", None).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    let (status, _) = call(r, "DELETE", "/api/sounds/clap", None).await;
    assert_eq!(status, StatusCode::OK);
}

#[tokio::test]
async fn robot_api_manual_mode_gate_and_estop() {
    let t = app(true).await;
    let r = &t.router;

    let (status, st) = call(r, "GET", "/api/robot/status", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(st["backend"], "virtual");
    assert_eq!(st["connected"], true);
    assert_eq!(st["estop"], false);

    // Out of range joint is a 400 with the offending joint named.
    let (status, body) =
        call(r, "POST", "/api/robot/joints", Some(json!({ "joints": [5.0, 0, 0, 0, 0, 0], "gripper": 0.5 }))).await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert!(body["error"].as_str().unwrap().contains("Joint 1"));

    // Valid command executes immediately on the virtual arm (no gate).
    let (status, body) =
        call(r, "POST", "/api/robot/joints", Some(json!({ "joints": [0.5, 0, 0, 0, 0, 0], "gripper": 1.0 }))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["executed"], true);

    // Ramp: a few ticks move the joint by at most 1.5 rad/s.
    {
        let mut core = t.state.robot.core.lock().await;
        for _ in 0..3 {
            core.tick(1.0 / 30.0);
        }
        assert!((core.joints[0] - 0.15).abs() < 1e-5, "{:?}", core.joints);
    }

    // Physical backend needs the serial feature.
    let (status, body) = call(r, "POST", "/api/robot/target", Some(json!({ "backend": "physical" }))).await;
    if cfg!(feature = "serial") {
        assert!(status == StatusCode::SERVICE_UNAVAILABLE || status == StatusCode::OK, "{body}");
    } else {
        assert_eq!(status, StatusCode::NOT_IMPLEMENTED, "{body}");
    }

    // E-stop locks commands; reset re-enables them.
    let (status, st) = call(r, "POST", "/api/robot/e-stop", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(st["estop"], true);
    let (status, _) =
        call(r, "POST", "/api/robot/joints", Some(json!({ "joints": [0, 0, 0, 0, 0, 0], "gripper": 0.5 }))).await;
    assert_eq!(status, StatusCode::LOCKED);
    let (status, st) = call(r, "POST", "/api/robot/reset-safety", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(st["estop"], false);

    // Mode A: a detection maps to the gripper through the shared gesture map.
    let (status, _) = call(r, "POST", "/api/robot/mode", Some(json!({ "mode": "shadowing" }))).await;
    assert_eq!(status, StatusCode::OK);
    t.state.robot.core.lock().await.apply_gesture("Fist").unwrap();
    let (_, st) = call(r, "GET", "/api/robot/status", None).await;
    assert_eq!(st["gripper_target"], 0.0);
    assert_eq!(st["last_gesture"], "Fist");

    // Gesture map validation.
    let (status, _) = call(
        r,
        "PUT",
        "/api/robot/gesture-map",
        Some(json!({ "Wave": { "action": "joint_delta", "joint": 9, "delta": 0.1 } })),
    )
    .await;
    assert_eq!(status, StatusCode::BAD_REQUEST);
    let (status, _) =
        call(r, "PUT", "/api/robot/gesture-map", Some(json!({ "Wave": { "action": "open_gripper" } }))).await;
    assert_eq!(status, StatusCode::OK);

    // Mode C: the goal is one image, observations are another (different embeddings).
    let png_of = |rgb: [u8; 3]| {
        let img = image::RgbImage::from_pixel(32, 32, image::Rgb(rgb));
        let mut buf = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img).write_to(&mut buf, image::ImageFormat::Png).unwrap();
        base64::prelude::BASE64_STANDARD.encode(buf.into_inner())
    };
    let goal_png = png_of([200, 30, 30]);
    let png_1x1 = png_of([30, 30, 200]);
    let (status, st) = call(r, "POST", "/api/robot/goal", Some(json!({ "image_base64": goal_png }))).await;
    assert_eq!(status, StatusCode::OK, "{st}");
    assert_eq!(st["goal"]["has_goal"], true);
    let (status, _) = call(r, "POST", "/api/robot/mode", Some(json!({ "mode": "goal_seeking" }))).await;
    assert_eq!(status, StatusCode::OK);
    // Settle the arm at the explorer's first pose, then observe.
    {
        let mut core = t.state.robot.core.lock().await;
        for _ in 0..200 {
            core.tick(1.0 / 30.0);
        }
        assert!(core.telemetry().goal.awaiting_observation);
    }
    let (status, body) = call(r, "POST", "/api/robot/observe", Some(json!({ "image_base64": png_1x1 }))).await;
    assert_eq!(status, StatusCode::OK, "{body}");
    assert_eq!(body["accepted"], true);
    assert_eq!(body["goal"]["steps"], 1);
    assert_eq!(body["goal"]["policy"], "random");
    assert!(body["command"]["joints"].is_array());
    // After the move settles, the next observation completes a transition and the
    // world model learns from it.
    {
        let mut core = t.state.robot.core.lock().await;
        for _ in 0..200 {
            core.tick(1.0 / 30.0);
        }
    }
    let (_, body) = call(r, "POST", "/api/robot/observe", Some(json!({ "image_base64": png_1x1 }))).await;
    assert_eq!(body["accepted"], true, "{body}");
    assert_eq!(body["goal"]["world"]["transitions"], 1);
    assert_eq!(body["goal"]["steps"], 2);
    assert!(body["goal"]["current_energy"].as_f64().unwrap() > 0.0);

    // Exploring mode keeps learning without a goal.
    let (status, _) = call(r, "DELETE", "/api/robot/goal", None).await;
    assert_eq!(status, StatusCode::OK);
    let (status, st) = call(r, "POST", "/api/robot/mode", Some(json!({ "mode": "exploring" }))).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(st["connected"], true, "a failed backend switch must not disconnect the current one");
    {
        let mut core = t.state.robot.core.lock().await;
        for _ in 0..200 {
            core.tick(1.0 / 30.0);
        }
        assert!(core.telemetry().goal.awaiting_observation);
    }
    let (_, body) = call(r, "GET", "/api/robot/world-model", None).await;
    assert_eq!(body["stats"]["transitions"], 1);
}
