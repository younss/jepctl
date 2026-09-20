//! HTTP handlers for the few-shot gesture registry and the camera "model view".

use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use base64::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::gestures::{match_gestures, GestureMatchResult, RegisteredGesture, DEFAULT_MARGIN, DEFAULT_THRESHOLD};
use crate::media::image::preprocess_image_bytes;
use crate::media::ring_buffer::MODEL_VIEW_SIZE;
use crate::server::handlers::{api_error, embed_current_view, engine_error, ensure_model_loaded, ApiError, AppState};
use crate::server::middleware::authenticate_request;
use crate::types::Role;

/// Registration payload. Exactly one input source must be given:
/// `from_camera`, `embedding` or `image_base64`.
#[derive(Debug, Deserialize)]
pub struct CreateGesturePayload {
    pub name: String,
    #[serde(default)]
    pub is_neutral: bool,
    /// Embed the latest server camera frame (recommended: identical pipeline to live matching).
    #[serde(default)]
    pub from_camera: bool,
    pub embedding: Option<Vec<f32>>,
    pub image_base64: Option<String>,
    /// Data URI shown in the UI. Ignored with `from_camera` (the camera frame is used).
    pub thumbnail: Option<String>,
}

/// Matching payload. Same input sources as registration.
#[derive(Debug, Deserialize)]
pub struct MatchGesturePayload {
    #[serde(default)]
    pub from_camera: bool,
    pub embedding: Option<Vec<f32>>,
    pub image_base64: Option<String>,
    pub threshold: Option<f32>,
    pub margin: Option<f32>,
}

/// Public view of a registered gesture (prototype vectors are omitted).
#[derive(Debug, Serialize)]
pub struct GestureItem {
    pub name: String,
    pub model_name: String,
    pub dimension: usize,
    pub sample_count: usize,
    pub is_neutral: bool,
    pub has_patch_prototype: bool,
    pub created_at: u64,
    pub updated_at: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumbnail: Option<String>,
}

impl From<&RegisteredGesture> for GestureItem {
    fn from(g: &RegisteredGesture) -> Self {
        Self {
            name: g.name.clone(),
            model_name: g.model_name.clone(),
            dimension: g.dimension,
            sample_count: g.samples.len(),
            is_neutral: g.is_neutral,
            has_patch_prototype: g.patch_prototype.is_some(),
            created_at: g.created_at,
            updated_at: g.updated_at,
            thumbnail: g.thumbnail.clone(),
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct ListGesturesQuery {
    /// Restrict to a model (defaults to the active model).
    pub model: Option<String>,
    /// List gestures of every model.
    #[serde(default)]
    pub all: bool,
}

fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

fn decode_data_uri(b64: &str) -> Result<Vec<u8>, ApiError> {
    let payload = b64.rsplit(',').next().unwrap_or(b64);
    BASE64_STANDARD
        .decode(payload.trim())
        .map_err(|e| api_error(StatusCode::BAD_REQUEST, format!("Invalid base64 image: {e}")))
}

/// Resolved input for registration or matching.
struct ResolvedInput {
    model: String,
    embedding: Vec<f32>,
    patches: Option<Vec<Vec<f32>>>,
    thumbnail: Option<String>,
}

async fn resolve_input(
    state: &AppState,
    from_camera: bool,
    embedding: Option<Vec<f32>>,
    image_base64: Option<String>,
    thumbnail: Option<String>,
) -> Result<ResolvedInput, ApiError> {
    if from_camera {
        let view = embed_current_view(state).await?;
        let thumb = format!("data:image/jpeg;base64,{}", BASE64_STANDARD.encode(view.frame_jpeg.as_slice()));
        return Ok(ResolvedInput {
            model: view.model,
            embedding: view.embedding,
            patches: view.patches,
            thumbnail: Some(thumb),
        });
    }

    if let Some(emb) = embedding {
        if emb.is_empty() {
            return Err(api_error(StatusCode::BAD_REQUEST, "'embedding' must not be empty"));
        }
        let model = ensure_model_loaded(state).await?;
        return Ok(ResolvedInput { model, embedding: emb, patches: None, thumbnail });
    }

    if let Some(b64) = image_base64 {
        let bytes = decode_data_uri(&b64)?;
        let tensor = preprocess_image_bytes(&bytes, MODEL_VIEW_SIZE, MODEL_VIEW_SIZE, &state.engine.device)
            .map_err(|e| api_error(StatusCode::BAD_REQUEST, format!("Image preprocessing failed: {e}")))?;
        ensure_model_loaded(state).await?;
        let (model, _dim, emb, patches, _lat) = state.engine.embed_image(&tensor).await.map_err(engine_error)?;
        let thumb = thumbnail.or_else(|| Some(format!("data:image/jpeg;base64,{}", BASE64_STANDARD.encode(&bytes))));
        return Ok(ResolvedInput { model, embedding: emb, patches, thumbnail: thumb });
    }

    Err(api_error(
        StatusCode::BAD_REQUEST,
        "Provide one input: 'from_camera': true, 'embedding' or 'image_base64'",
    ))
}

/// POST /api/gestures - Register (or add a sample to) a reference gesture.
pub async fn handle_create_gesture(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<CreateGesturePayload>,
) -> Result<(StatusCode, Json<GestureItem>), ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;

    let name = payload.name.trim().to_string();
    if name.is_empty() || name.len() > 64 {
        return Err(api_error(StatusCode::BAD_REQUEST, "'name' must be 1-64 characters"));
    }

    let input = resolve_input(&state, payload.from_camera, payload.embedding, payload.image_base64, payload.thumbnail).await?;

    let now = now_secs();
    let item = {
        let mut store = state.gestures.write().await;
        let gesture = store.get_mut_or_insert(&name, &input.model, payload.is_neutral, now);
        gesture
            .add_sample(&input.embedding, input.patches.as_deref(), now)
            .map_err(engine_error)?;
        if input.thumbnail.is_some() {
            gesture.thumbnail = input.thumbnail;
        }
        let item = GestureItem::from(&*gesture);
        if let Err(e) = store.save() {
            tracing::warn!("Could not persist gesture registry: {}", e);
        }
        item
    };

    tracing::info!(
        "Gesture '{}' now has {} sample(s) for model '{}' (dim {})",
        item.name, item.sample_count, item.model_name, item.dimension
    );
    Ok((StatusCode::CREATED, Json(item)))
}

/// GET /api/gestures - List registered gestures (active model by default).
pub async fn handle_list_gestures(
    State(state): State<AppState>,
    Query(q): Query<ListGesturesQuery>,
) -> Json<Vec<GestureItem>> {
    let store = state.gestures.read().await;
    let items: Vec<GestureItem> = if q.all {
        let mut all: Vec<&RegisteredGesture> = store.gestures.values().collect();
        all.sort_by_key(|g| g.created_at);
        all.into_iter().map(GestureItem::from).collect()
    } else {
        let model = match q.model {
            Some(m) => Some(m),
            None => state.engine.get_active_model_name().await,
        };
        match model {
            Some(m) => store.for_model(&m).into_iter().map(GestureItem::from).collect(),
            None => Vec::new(),
        }
    };
    Json(items)
}

/// DELETE /api/gestures/{name} - Remove a gesture and all its samples.
pub async fn handle_delete_gesture(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let mut store = state.gestures.write().await;
    if store.remove(&name).is_none() {
        return Err(api_error(StatusCode::NOT_FOUND, format!("Gesture '{name}' not found")));
    }
    if let Err(e) = store.save() {
        tracing::warn!("Could not persist gesture registry: {}", e);
    }
    tracing::info!("Deleted gesture '{}'", name);
    Ok(Json(json!({ "status": "deleted", "name": name })))
}

/// DELETE /api/gestures - Remove every gesture of the active model (or `?all=true`).
pub async fn handle_clear_gestures(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<ListGesturesQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let model = match (&q.model, q.all) {
        (_, true) => None,
        (Some(m), false) => Some(m.clone()),
        (None, false) => state.engine.get_active_model_name().await,
    };
    let mut store = state.gestures.write().await;
    let before = store.gestures.len();
    match &model {
        Some(m) => store.gestures.retain(|_, g| &g.model_name != m),
        None => store.gestures.clear(),
    }
    let removed = before - store.gestures.len();
    if let Err(e) = store.save() {
        tracing::warn!("Could not persist gesture registry: {}", e);
    }
    Ok(Json(json!({ "status": "cleared", "removed": removed, "model": model })))
}

/// POST /api/gestures/match - Match an input against the registered gestures.
pub async fn handle_match_gesture(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<MatchGesturePayload>,
) -> Result<Json<GestureMatchResult>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;

    let threshold = payload.threshold.unwrap_or(DEFAULT_THRESHOLD).clamp(0.0, 1.0);
    let margin = payload.margin.unwrap_or(DEFAULT_MARGIN).clamp(0.0, 1.0);
    let input = resolve_input(&state, payload.from_camera, payload.embedding, payload.image_base64, None).await?;

    let store = state.gestures.read().await;
    let registered = store.for_model(&input.model);
    if registered.is_empty() {
        return Ok(Json(GestureMatchResult::empty(
            threshold,
            &format!("No gestures registered for model '{}'", input.model),
        )));
    }
    Ok(Json(match_gestures(&input.embedding, input.patches.as_deref(), &registered, threshold, margin)))
}

/// GET /api/camera/frame - JPEG of exactly what the model receives (224x224 centre crop).
pub async fn handle_camera_frame(State(state): State<AppState>, headers: HeaderMap) -> Result<Response, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let (bytes, seq) = {
        let rb = state.ring_buffer.read().await;
        let latest = rb
            .latest()
            .ok_or_else(|| api_error(StatusCode::NOT_FOUND, "Camera is not running or no frame captured yet"))?;
        (latest.model_view_jpeg.clone(), latest.sequence)
    };
    Ok((
        [
            (header::CONTENT_TYPE, "image/jpeg".to_string()),
            (header::CACHE_CONTROL, "no-store".to_string()),
            (header::HeaderName::from_static("x-frame-sequence"), seq.to_string()),
        ],
        bytes.as_slice().to_vec(),
    )
        .into_response())
}
