//! Axum HTTP and SSE route handlers for embedding, streaming, and energy scoring.

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use axum::extract::{Multipart, Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{Json, Response};
use futures_util::stream::Stream;
use serde::Deserialize;
use serde_json::json;
use tokio_stream::StreamExt;
use tokio_stream::wrappers::BroadcastStream;

use crate::auth::AuthManager;
use crate::config::{MAX_IMAGE_PAYLOAD_SIZE, MAX_VIDEO_PAYLOAD_SIZE, RuntimeConfig};
use crate::engine::EngineManager;
use crate::gestures::{DEFAULT_MARGIN, DEFAULT_THRESHOLD, GestureStore, match_gestures};
use crate::hub::ModelCatalog;
use crate::hub::manifest::JepafileConfig;
use crate::media::capture::{CameraSupervisor, list_camera_devices};
use crate::media::image::{preprocess_image_bytes, sniff_media_format};
use crate::media::ring_buffer::{MODEL_VIEW_SIZE, SharedRingBuffer};
use crate::server::middleware::{SharedAuditLog, authenticate_request};
use crate::types::{
    CreateKeyRequest, EmbedResponse, EnergyRequest, EnergyResponse, JepaError, ModelModality, Roi, Role, SettingsDto,
    StatusResponse, StreamEvent,
};

/// Global application state shared across HTTP route handlers
#[derive(Clone)]
pub struct AppState {
    pub engine: Arc<EngineManager>,
    pub catalog: Arc<ModelCatalog>,
    pub auth: Arc<AuthManager>,
    pub audit_log: SharedAuditLog,
    pub camera_supervisor: Arc<CameraSupervisor>,
    pub ring_buffer: SharedRingBuffer,
    pub embeddings_total: Arc<AtomicU64>,
    pub gestures: Arc<tokio::sync::RwLock<GestureStore>>,
    /// Registered sounds: same prototype store, embedded by the audio model.
    pub sounds: Arc<tokio::sync::RwLock<GestureStore>>,
    pub mic: Arc<crate::media::mic::MicSupervisor>,
    /// Companion robot (watches and listens).
    pub companion: crate::companion::CompanionHandle,
    /// Online latent world model for the World tab (predict-next and surprise).
    pub scene: std::sync::Arc<tokio::sync::Mutex<crate::engine::scene_predictor::WorldScenePredictor>>,
    /// Region of interest applied to camera frames (see `types::Roi`).
    pub camera_roi: Arc<tokio::sync::RwLock<Option<Roi>>>,
    /// Robot arm control and digital twin.
    pub robot: crate::robot::RobotHandle,
    pub start_time: Instant,
    pub config: Arc<RuntimeConfig>,
}

/// Persist the current ROI into settings.json without touching other settings.
pub fn persist_roi(config: &RuntimeConfig, roi: Option<Roi>) {
    let mut settings = config.load_settings();
    settings.camera_roi = roi;
    match serde_json::to_string_pretty(&settings) {
        Ok(data) => {
            if let Err(e) = std::fs::write(&config.settings_path, data) {
                tracing::warn!("Could not persist ROI: {}", e);
            }
        }
        Err(e) => tracing::warn!("Could not serialise settings: {}", e),
    }
}

/// JSON error tuple used by every handler.
pub type ApiError = (StatusCode, Json<serde_json::Value>);

pub fn api_error(status: StatusCode, msg: impl Into<String>) -> ApiError {
    (status, Json(json!({ "error": msg.into() })))
}

/// Map an engine error onto an HTTP status.
pub fn engine_error(e: JepaError) -> ApiError {
    let status = match &e {
        JepaError::ModelNotFound(_) => StatusCode::CONFLICT,
        JepaError::InvalidPayload(_) | JepaError::ImageProcessing(_) => StatusCode::BAD_REQUEST,
        JepaError::WeightsIncomplete { .. } => StatusCode::UNPROCESSABLE_ENTITY,
        _ => StatusCode::INTERNAL_SERVER_ERROR,
    };
    api_error(status, e.to_string())
}

/// Return the active model name, auto-loading the first *downloaded* model when
/// nothing is loaded. Never falls back to an uninitialised model.
pub async fn ensure_model_loaded(state: &AppState) -> Result<String, ApiError> {
    if let Some(name) = state.engine.get_active_model_name().await {
        return Ok(name);
    }
    let candidate = state
        .catalog
        .list_installed()
        .into_iter()
        .find_map(|m| state.catalog.get_weights_path(&m.name).map(|p| (m, p)));

    match candidate {
        Some((manifest, path)) => {
            let name = manifest.name.clone();
            tracing::info!("No model loaded; auto-loading '{}'", name);
            state.engine.load_model(manifest, Some(&path)).await.map_err(engine_error)?;
            Ok(name)
        }
        None => Err(api_error(
            StatusCode::CONFLICT,
            "No model loaded and no downloaded model available. Pull a model first (e.g. `jepctl pull facebook/ijepa_vith14_1k`).",
        )),
    }
}

/// Embedding of the frame currently in front of the camera, computed the same way
/// for gesture registration and live matching.
pub struct CurrentViewEmbedding {
    pub model: String,
    pub embedding: Vec<f32>,
    pub patches: Option<Vec<Vec<f32>>>,
    pub latency_ms: f64,
    pub frame_sequence: u64,
    pub frame_jpeg: Arc<Vec<u8>>,
}

/// Embed the latest camera frame with the active model.
///
/// Image models embed the latest frame only; video models embed the full temporal
/// window of the ring buffer. Both paths are used for registration *and* matching
/// so reference prototypes and live inputs are always comparable.
pub async fn embed_current_view(state: &AppState) -> Result<CurrentViewEmbedding, ApiError> {
    let model = ensure_model_loaded(state).await?;
    let modality = state.engine.get_active_modality().await.unwrap_or(ModelModality::Image);
    let prep = state.engine.preprocessing().await;
    let roi = *state.camera_roi.read().await;

    let (tensor, frame_sequence, frame_jpeg) = {
        let rb = state.ring_buffer.read().await;
        let (jpeg, seq) = rb
            .latest_model_view_jpeg(MODEL_VIEW_SIZE, roi.as_ref())
            .ok_or_else(|| api_error(StatusCode::CONFLICT, "Camera is not running or no frame captured yet"))?;
        let tensor = match modality {
            ModelModality::Image => rb.latest_image_tensor(&prep, roi.as_ref(), &state.engine.device),
            _ => rb.to_video_tensor(&prep, roi.as_ref(), &state.engine.device),
        }
        .map_err(engine_error)?;
        (tensor, seq, Arc::new(jpeg))
    };

    let (embedding, patches, latency_ms) = match modality {
        ModelModality::Image => {
            let (_m, _d, emb, patches, lat) = state.engine.embed_image(&tensor).await.map_err(engine_error)?;
            (emb, patches, lat)
        }
        _ => {
            let (_m, _d, emb, patches, lat) = state.engine.embed_video(&tensor).await.map_err(engine_error)?;
            (emb, patches, lat)
        }
    };
    state.embeddings_total.fetch_add(1, Ordering::Relaxed);

    Ok(CurrentViewEmbedding { model, embedding, patches, latency_ms, frame_sequence, frame_jpeg })
}

/// GET /api/status
pub async fn handle_status(State(state): State<AppState>) -> Json<StatusResponse> {
    let hw = state.engine.get_telemetry().await;
    let active_model = state.engine.get_active_model_name().await;
    let weights = state.engine.get_active_weight_report().await;
    let embeddings = state.embeddings_total.load(Ordering::Relaxed);
    let uptime = state.start_time.elapsed().as_secs();

    Json(StatusResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        platform: std::env::consts::OS.to_string(),
        hardware: hw,
        active_model,
        weights,
        audio_model: state.engine.get_audio_model_name().await,
        audio_weights: state.engine.get_audio_weight_report().await,
        camera_active: state.camera_supervisor.is_active(),
        camera: Some(state.camera_supervisor.health()),
        mic: Some(state.mic.health()),
        embeddings_computed_total: embeddings,
        uptime_seconds: uptime,
    })
}

/// GET /api/catalog - Verified catalog (models known to load with full checkpoint coverage)
pub async fn handle_catalog() -> Json<serde_json::Value> {
    Json(json!(crate::hub::manifest::get_verified_manifests()))
}

/// GET /api/tags - List locally installed models
pub async fn handle_tags(State(state): State<AppState>) -> Json<serde_json::Value> {
    let models = state.catalog.list_installed();
    Json(json!(models))
}

/// POST /api/models/load - Load model into active memory
#[derive(Deserialize)]
pub struct LoadModelPayload {
    pub model_name: String,
}

pub async fn handle_load_model(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<LoadModelPayload>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let _ = authenticate_request(&headers, &state.auth, Role::Admin).await?;

    let manifest = state.catalog.get_manifest(&payload.model_name).ok_or_else(|| {
        (StatusCode::NOT_FOUND, Json(json!({ "error": format!("Model not found: {}", payload.model_name) })))
    })?;

    let weights_path = state.catalog.get_weights_path(&payload.model_name);

    let report = state.engine.load_model(manifest, weights_path.as_deref()).await.map_err(engine_error)?;

    Ok(Json(json!({
        "status": "loaded",
        "model": payload.model_name,
        "weights": report
    })))
}

/// POST /api/models/unload - Unload active model from memory
#[derive(Deserialize, Default)]
pub struct UnloadModelPayload {
    /// Unload this model wherever it is loaded (vision or audio slot). Without it
    /// the vision model is unloaded.
    pub model_name: Option<String>,
}

pub async fn handle_unload_model(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Option<Json<UnloadModelPayload>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let _ = authenticate_request(&headers, &state.auth, Role::Admin).await?;
    match payload.and_then(|p| p.model_name.clone()) {
        Some(name) => state.engine.unload_named(&name).await,
        None => state.engine.unload_model().await,
    }
    Ok(Json(json!({ "status": "unloaded" })))
}

#[derive(Deserialize)]
pub struct DeleteModelQuery {
    pub name: Option<String>,
}

#[derive(Deserialize)]
pub struct DeleteModelPayload {
    pub name: Option<String>,
    pub model: Option<String>,
}

/// Helper executing model deletion from active memory and disk storage
async fn execute_delete_model(
    state: AppState,
    headers: HeaderMap,
    raw_name: String,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let _ = authenticate_request(&headers, &state.auth, Role::Admin).await?;

    let name = raw_name.trim().trim_start_matches('/').replace("%2F", "/").replace("%2f", "/");

    // If deleting a loaded model, unload it first.
    state.engine.unload_named(&name).await;
    state.engine.unload_named(&name.replace(':', "/")).await;

    match state.catalog.delete_model(&name) {
        Ok(true) => Ok(Json(json!({ "status": "deleted", "model": name }))),
        Ok(false) => {
            Err((StatusCode::NOT_FOUND, Json(json!({ "error": format!("Model '{}' not found on disk", name) }))))
        }
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))),
    }
}

/// DELETE /api/models/{*name} - Delete an installed model via path
pub async fn handle_delete_model(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    execute_delete_model(state, headers, name).await
}

/// DELETE /api/models - Delete an installed model via query param (?name=...) or JSON payload
pub async fn handle_delete_model_root(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<DeleteModelQuery>,
    body: Option<Json<DeleteModelPayload>>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let name = query.name.or_else(|| body.and_then(|Json(b)| b.name.or(b.model))).ok_or_else(|| {
        (StatusCode::BAD_REQUEST, Json(json!({ "error": "Missing 'name' or 'model' parameter for deletion" })))
    })?;
    execute_delete_model(state, headers, name).await
}

/// POST /api/pull - Stream progress downloading a model from Hugging Face
#[derive(Deserialize)]
pub struct PullRequest {
    pub repo_id: String,
}

pub async fn handle_pull(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<PullRequest>,
) -> Result<Response, (StatusCode, Json<serde_json::Value>)> {
    let _ = authenticate_request(&headers, &state.auth, Role::Admin).await?;
    let rx = state.catalog.clone().start_pull(payload.repo_id);
    let stream = BroadcastStream::new(rx).filter_map(|res| match res {
        Ok(event) => {
            let serialized = serde_json::to_string(&event).unwrap_or_default();
            Some(Ok::<_, Infallible>(format!("{}\n", serialized)))
        }
        Err(_) => None,
    });

    Ok(Response::builder()
        .header("Content-Type", "application/x-ndjson")
        .body(axum::body::Body::from_stream(stream))
        .unwrap())
}

/// POST /api/embed - Multipart file upload returning embedding representation
pub async fn handle_embed(
    State(state): State<AppState>,
    headers: HeaderMap,
    mut multipart: Multipart,
) -> Result<Json<EmbedResponse>, (StatusCode, Json<serde_json::Value>)> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;

    let mut file_bytes: Option<Vec<u8>> = None;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" || name == "image" || name == "video" {
            let data = field.bytes().await.map_err(|e| {
                (StatusCode::BAD_REQUEST, Json(json!({ "error": format!("Failed reading upload field: {}", e) })))
            })?;
            file_bytes = Some(data.to_vec());
            break;
        }
    }

    let buffer = file_bytes
        .ok_or_else(|| (StatusCode::BAD_REQUEST, Json(json!({ "error": "No 'file' multipart field provided" }))))?;

    // Check payload limits
    if buffer.len() > MAX_VIDEO_PAYLOAD_SIZE {
        return Err((StatusCode::PAYLOAD_TOO_LARGE, Json(json!({ "error": "Payload exceeds 200 MB maximum limit" }))));
    }

    // Audio first (WAV/MP3/FLAC/OGG), then images and clips.
    if let Some(audio_fmt) = crate::media::audio::sniff_audio_format(&buffer) {
        let clip = tokio::task::spawn_blocking({
            let buffer = buffer.clone();
            move || crate::media::audio::decode_audio_bytes(&buffer)
        })
        .await
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Decode task failed: {e}")))?
        .map_err(engine_error)?;
        let (model, dim, embedding, patches, latency_ms) =
            state.engine.embed_audio(&clip).await.map_err(engine_error)?;
        state.embeddings_total.fetch_add(1, Ordering::Relaxed);
        tracing::debug!(
            "Embedded {} audio ({} samples @ {} Hz) with {}",
            audio_fmt,
            clip.samples.len(),
            clip.sample_rate,
            model
        );
        return Ok(Json(EmbedResponse { model, dimension: dim, latency_ms, embedding, patch_embeddings: patches }));
    }

    // Sniff media format
    let format = sniff_media_format(&buffer)
        .map_err(|e| (StatusCode::UNSUPPORTED_MEDIA_TYPE, Json(json!({ "error": e.to_string() }))))?;

    // Animated WebP is a clip; a still WebP is an image.
    let animated_webp = format == "webp" && crate::media::video::decode_clip_bytes(&buffer, 1, 1.0).is_ok();

    // If image format: PNG, JPEG, still WebP
    if matches!(format, "png" | "jpeg") || (format == "webp" && !animated_webp) {
        if buffer.len() > MAX_IMAGE_PAYLOAD_SIZE {
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                Json(json!({ "error": "Image payload exceeds 20 MB maximum limit" })),
            ));
        }

        ensure_model_loaded(&state).await?;
        let prep = state.engine.preprocessing().await;
        let tensor = preprocess_image_bytes(&buffer, &prep, &state.engine.device)
            .map_err(|e| api_error(StatusCode::BAD_REQUEST, format!("Image preprocessing failed: {e}")))?;

        let (model, dim, embedding, patches, latency_ms) =
            state.engine.embed_image(&tensor).await.map_err(engine_error)?;

        state.embeddings_total.fetch_add(1, Ordering::Relaxed);

        Ok(Json(EmbedResponse { model, dimension: dim, latency_ms, embedding, patch_embeddings: patches }))
    } else {
        // Video: GIF/WebP natively, MP4/WebM through ffmpeg (see media::video).
        ensure_model_loaded(&state).await?;
        let frames = tokio::task::spawn_blocking({
            let buffer = buffer.clone();
            move || crate::media::video::decode_clip_bytes(&buffer, crate::media::video::MAX_DECODED_FRAMES, 8.0)
        })
        .await
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Decode task failed: {e}")))?
        .map_err(engine_error)?;

        let (model, dim, embedding, patches, latency_ms, used) =
            state.engine.embed_frames(&frames).await.map_err(engine_error)?;
        state.embeddings_total.fetch_add(1, Ordering::Relaxed);
        tracing::debug!("Embedded a {}-frame clip ({} sampled) with {}", frames.len(), used, model);

        Ok(Json(EmbedResponse { model, dimension: dim, latency_ms, embedding, patch_embeddings: patches }))
    }
}

/// Query parameters for streaming endpoint
#[derive(Deserialize)]
pub struct StreamQuery {
    pub fps: Option<u64>,
    /// Blended-score threshold for gesture detection (default 0.70).
    pub threshold: Option<f32>,
    /// Required lead over the runner-up (default 0.04).
    pub margin: Option<f32>,
    /// Bearer token (query fallback for `EventSource`, which cannot set headers).
    pub token: Option<String>,
}

/// GET /api/embed/stream - Server-Sent Events (SSE) stream of live embeddings.
///
/// Each event carries the embedding of the current camera view, and, when gestures
/// are registered for the active model, a full [`crate::gestures::GestureMatchResult`].
/// Errors (no model, no camera) are sent as `event: error` messages instead of
/// silently dropping frames.
pub async fn handle_embed_stream(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(params): Query<StreamQuery>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, ApiError> {
    // `EventSource` cannot set headers, so the token may also come as `?token=`.
    let mut headers = headers;
    if let Some(t) = params.token.as_deref()
        && let Ok(v) = format!("Bearer {t}").parse()
    {
        headers.insert(axum::http::header::AUTHORIZATION, v);
    }
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;

    let mut frame_rx = state.camera_supervisor.subscribe();
    let fps = params.fps.unwrap_or(10).clamp(1, 30);
    let threshold = params.threshold.unwrap_or(DEFAULT_THRESHOLD).clamp(0.0, 1.0);
    let margin = params.margin.unwrap_or(DEFAULT_MARGIN).clamp(0.0, 1.0);
    let interval_duration = Duration::from_millis(1000 / fps);

    let stream = async_stream::stream! {
        let mut frame_idx: usize = 0;
        let mut last_sequence: u64 = 0;
        let mut last_error: Option<String> = None;
        let start = Instant::now();

        loop {
            tokio::select! {
                _ = frame_rx.recv() => {},
                _ = tokio::time::sleep(interval_duration) => {}
            }

            // Skip work when the camera has not produced a new frame.
            let current_seq = state.ring_buffer.read().await.latest().map(|f| f.sequence).unwrap_or(0);
            if current_seq == last_sequence {
                continue;
            }
            last_sequence = current_seq;

            match embed_current_view(&state).await {
                Ok(view) => {
                    last_error = None;
                    frame_idx += 1;

                    let gesture_match = {
                        let store = state.gestures.read().await;
                        let registered = store.for_model(&view.model);
                        (!registered.is_empty()).then(|| {
                            match_gestures(&view.embedding, view.patches.as_deref(), &registered, threshold, margin)
                        })
                    };
                    // Mode A: shadowing follows the same detections the UI shows;
                    // mirror mode blends every mapped pose by score.
                    if let Some(m) = gesture_match.as_ref() {
                        let mut core = state.robot.core.lock().await;
                        if let Some(name) = m.matched.as_deref()
                            && let Err(e) = core.apply_gesture(name) {
                                core.last_error = Some(e.to_string());
                            }
                        if let Err(e) = core.apply_mirror(&m.scores) {
                            core.last_error = Some(e.to_string());
                        }
                    }

                    let event_data = StreamEvent {
                        frame_index: frame_idx,
                        timestamp: start.elapsed().as_secs_f64(),
                        model: view.model,
                        latency_ms: view.latency_ms,
                        embedding: view.embedding,
                        energy_distance: None,
                        anomaly: false,
                        gesture_match,
                    };
                    if let Ok(json_str) = serde_json::to_string(&event_data) {
                        yield Ok(Event::default().id(view.frame_sequence.to_string()).data(json_str));
                    }
                }
                Err((status, Json(body))) => {
                    let msg = body.get("error").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                    // Emit each distinct error once, not once per frame.
                    if last_error.as_deref() != Some(&msg) {
                        tracing::warn!("Stream embedding error ({}): {}", status, msg);
                        last_error = Some(msg.clone());
                        yield Ok(Event::default().event("error").data(json!({ "error": msg, "status": status.as_u16() }).to_string()));
                    }
                }
            }
        }
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

/// POST /api/energy - Compute latent energy distance and anomaly metrics
pub async fn handle_energy(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<EnergyRequest>,
) -> Result<Json<EnergyResponse>, (StatusCode, Json<serde_json::Value>)> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;

    let threshold = req.threshold.unwrap_or(0.45);

    let (v1, v2) = match (req.vector1, req.vector2) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": "Must provide two vectors (vector1 and vector2)" })),
            ));
        }
    };

    if v1.len() != v2.len() || v1.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Vectors must be non-empty and possess identical dimensions" })),
        ));
    }

    let dim = v1.len() as f32;
    let mut sum_sq = 0.0f32;
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..v1.len() {
        let diff = v1[i] - v2[i];
        sum_sq += diff * diff;
        dot += v1[i] * v2[i];
        norm_a += v1[i] * v1[i];
        norm_b += v2[i] * v2[i];
    }

    let l2_distance = (sum_sq / dim).sqrt();
    let norm_product = (norm_a * norm_b).sqrt();
    let cosine_similarity = if norm_product > 1e-8 { dot / norm_product } else { 1.0 };
    let cosine_dissimilarity = (1.0 - cosine_similarity).max(0.0);

    let anomaly = l2_distance > threshold;

    Ok(Json(EnergyResponse { l2_distance, cosine_similarity, cosine_dissimilarity, anomaly, threshold }))
}

/// POST /api/keys - Generate scoped API key
pub async fn handle_create_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<CreateKeyRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let _ = authenticate_request(&headers, &state.auth, Role::Admin).await?;

    match state.auth.create_key(req).await {
        Ok(res) => Ok(Json(json!(res))),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))),
    }
}

/// GET /api/keys - List active API keys
pub async fn handle_list_keys(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let _ = authenticate_request(&headers, &state.auth, Role::Admin).await?;
    let keys = state.auth.list_keys().await;
    Ok(Json(json!(keys)))
}

/// DELETE /api/keys/:prefix - Revoke API key
pub async fn handle_revoke_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(prefix): Path<String>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let _ = authenticate_request(&headers, &state.auth, Role::Admin).await?;

    match state.auth.revoke_key(&prefix).await {
        Ok(true) => Ok(Json(json!({ "status": "revoked", "prefix": prefix }))),
        Ok(false) => Err((StatusCode::NOT_FOUND, Json(json!({ "error": "Key prefix not found" })))),
        Err(e) => Err((StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() })))),
    }
}

/// GET /api/audit - Get rolling audit log entries
pub async fn handle_audit(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let _ = authenticate_request(&headers, &state.auth, Role::Admin).await?;
    let lock = state.audit_log.read().await;
    let list: Vec<_> = lock.iter().cloned().collect();
    Ok(Json(json!(list)))
}

/// GET /api/cameras - List detected camera devices
pub async fn handle_cameras() -> Json<serde_json::Value> {
    let devices = list_camera_devices();
    Json(json!(devices))
}

/// Query for camera controls
#[derive(Deserialize)]
pub struct CameraControlQuery {
    pub device: Option<usize>,
    pub fps: Option<u64>,
}

/// POST /api/camera/start
pub async fn handle_camera_start(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<CameraControlQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let dev = q.device.unwrap_or(0);
    let fps = q.fps.unwrap_or(10).clamp(1, 30);
    state.camera_supervisor.start(dev, fps).map_err(engine_error)?;
    Ok(Json(json!({ "status": "started", "device": dev, "fps": fps })))
}

/// POST /api/camera/stop
pub async fn handle_camera_stop(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    state.camera_supervisor.stop();
    Ok(Json(json!({ "status": "stopped" })))
}

/// GET /api/ring-buffer - Retrieve thumbnails of frames currently in sliding window
pub async fn handle_ring_buffer(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let lock = state.ring_buffer.read().await;
    let count = lock.len();
    let thumbnails = lock.get_thumbnails();
    Ok(Json(json!({ "count": count, "thumbnails": thumbnails })))
}

/// POST /api/manifests - Register custom Jepafile
pub async fn handle_register_manifest(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(cfg): Json<JepafileConfig>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let _ = authenticate_request(&headers, &state.auth, Role::Admin).await?;

    match state.catalog.register_jepafile(cfg) {
        Ok(m) => Ok(Json(json!(m))),
        Err(e) => Err((StatusCode::BAD_REQUEST, Json(json!({ "error": e.to_string() })))),
    }
}

/// GET /api/settings - Retrieve runtime settings
pub async fn handle_get_settings(State(state): State<AppState>) -> Json<SettingsDto> {
    let settings = state.config.load_settings();
    Json(settings)
}

/// POST /api/settings - Save runtime settings
pub async fn handle_save_settings(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(dto): Json<SettingsDto>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let _ = authenticate_request(&headers, &state.auth, Role::Admin).await?;
    let mut dto = dto;
    if dto.camera_roi.is_none() {
        dto.camera_roi = *state.camera_roi.read().await;
    }
    let data = serde_json::to_string_pretty(&dto)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, Json(json!({ "error": e.to_string() }))))?;
    std::fs::write(&state.config.settings_path, data)
        .map_err(|e| api_error(StatusCode::INTERNAL_SERVER_ERROR, format!("Could not write settings: {e}")))?;
    Ok(Json(json!({ "status": "saved" })))
}

/// GET /api/auth/token - Retrieve session admin token for local loopback testbench
pub async fn handle_get_session_token(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    if state.auth.no_auth_enabled {
        return Ok(Json(json!({ "token": "no_auth", "no_auth": true })));
    }

    let host_hdr = headers.get("host").and_then(|h| h.to_str().ok()).unwrap_or("");
    let is_loopback =
        host_hdr.starts_with("127.0.0.1") || host_hdr.starts_with("localhost") || host_hdr.starts_with("[::1]");

    if !is_loopback && state.config.host != "127.0.0.1" {
        return Err((
            StatusCode::FORBIDDEN,
            Json(json!({ "error": "Session token retrieval only permitted from local loopback" })),
        ));
    }

    // Only the embedded testbench (same origin) may bootstrap a session this way.
    // Browsers send `Sec-Fetch-Site` on every fetch; a cross-site page gets refused
    // even if CORS were misconfigured. Non-browser clients must use `jepctl key`.
    let fetch_site = headers.get("sec-fetch-site").and_then(|h| h.to_str().ok()).unwrap_or("same-origin");
    if !matches!(fetch_site, "same-origin" | "none") {
        return Err((
            StatusCode::FORBIDDEN,
            Json(json!({ "error": "Session token is only issued to the embedded testbench (same-origin)" })),
        ));
    }

    let token =
        std::fs::read_to_string(&state.config.auth_token_path).map(|s| s.trim().to_string()).unwrap_or_default();

    Ok(Json(json!({ "token": token, "no_auth": false })))
}
