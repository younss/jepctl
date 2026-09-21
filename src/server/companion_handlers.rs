//! HTTP handlers for the microphone, the sound registry and the companion robot.

use axum::extract::ws::{Message, WebSocket};
use axum::extract::{Path, Query, State, WebSocketUpgrade};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{Json, Response};
use serde::{Deserialize, Serialize};
use serde_json::json;

use crate::companion::{motion_centroid, Behaviour, CompanionMode, CompanionPose, CompanionTelemetry, Cue, CueKind};
use crate::gestures::{match_gestures, GestureMatchResult, DEFAULT_MARGIN, DEFAULT_THRESHOLD};
use crate::media::mic::{list_mic_devices, MicHealth};
use crate::server::gesture_handlers::GestureItem;
use crate::server::handlers::{api_error, embed_current_view, engine_error, ApiError, AppState};
use crate::server::middleware::authenticate_request;
use crate::types::Role;

/// Seconds of microphone audio embedded per observation and per registration.
/// Short cues (a clap, a word) sit inside the window; AudioMAE zero-pads the rest.
pub const SOUND_WINDOW_SECONDS: f32 = 1.5;

/// Registered sounds score lower than gestures (a short cue in a padded window),
/// so the default threshold is more permissive.
pub const SOUND_THRESHOLD: f32 = 0.55;

fn now_secs_f64() -> f64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs_f64()
}

fn now_secs() -> u64 {
    now_secs_f64() as u64
}

// ---------------------------------------------------------------------------
// Microphone
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, Default)]
pub struct MicStartPayload {
    pub device: Option<usize>,
}

/// GET /api/mics - Input devices.
pub async fn handle_mics(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let devices = tokio::task::spawn_blocking(list_mic_devices).await.unwrap_or_default();
    Ok(Json(json!(devices)))
}

/// POST /api/mic/start
pub async fn handle_mic_start(
    State(state): State<AppState>,
    headers: HeaderMap,
    payload: Option<Json<MicStartPayload>>,
) -> Result<Json<MicHealth>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let device = payload.and_then(|p| p.device);
    state.mic.start(device).map_err(engine_error)?;
    // Give the worker a moment so the answer reflects an opened device or its error.
    tokio::time::sleep(std::time::Duration::from_millis(250)).await;
    Ok(Json(state.mic.health()))
}

/// POST /api/mic/stop
pub async fn handle_mic_stop(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<MicHealth>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    state.mic.stop();
    Ok(Json(state.mic.health()))
}

/// GET /api/mic/status
pub async fn handle_mic_status(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<MicHealth>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    Ok(Json(state.mic.health()))
}

#[derive(Debug, Deserialize)]
pub struct WaveformQuery {
    pub seconds: Option<f32>,
    pub points: Option<usize>,
}

/// GET /api/mic/waveform - Envelope of the last seconds of audio for the UI meter.
pub async fn handle_mic_waveform(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<WaveformQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let seconds = q.seconds.unwrap_or(SOUND_WINDOW_SECONDS).clamp(0.1, 10.0);
    let points = q.points.unwrap_or(120).clamp(8, 1000);
    let h = state.mic.health();
    Ok(Json(
        json!({ "active": h.active, "level": h.level, "seconds": seconds, "points": state.mic.waveform(seconds, points) }),
    ))
}

/// Embedding of the last `seconds` of microphone audio with the audio model.
pub struct CurrentAudioEmbedding {
    pub model: String,
    pub embedding: Vec<f32>,
    pub patches: Option<Vec<Vec<f32>>>,
    pub level: f32,
    pub latency_ms: f64,
    /// Envelope of the embedded clip (120 points) for display.
    pub waveform: Vec<f32>,
}

pub async fn embed_current_audio(state: &AppState, seconds: f32) -> Result<CurrentAudioEmbedding, ApiError> {
    if !state.mic.is_active() {
        return Err(api_error(
            StatusCode::CONFLICT,
            "Microphone is not running. Start it first (POST /api/mic/start).",
        ));
    }
    let Some(clip) = state.mic.latest_clip(seconds) else {
        return Err(api_error(StatusCode::CONFLICT, "Microphone has not captured audio yet"));
    };
    let level = state.mic.health().level;
    let waveform = crate::media::mic::envelope(&clip.samples, 120);
    let (model, _dim, embedding, patches, latency_ms) = state.engine.embed_audio(&clip).await.map_err(engine_error)?;
    state.embeddings_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Ok(CurrentAudioEmbedding { model, embedding, patches, level, latency_ms, waveform })
}

// ---------------------------------------------------------------------------
// Sounds (audio prototypes)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CreateSoundPayload {
    pub name: String,
    #[serde(default)]
    pub is_neutral: bool,
    /// Seconds of microphone audio to embed (default [`SOUND_WINDOW_SECONDS`]).
    pub seconds: Option<f32>,
    /// A precomputed embedding instead of the microphone.
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Deserialize)]
pub struct MatchSoundPayload {
    pub seconds: Option<f32>,
    pub embedding: Option<Vec<f32>>,
    pub threshold: Option<f32>,
    pub margin: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct ListSoundsQuery {
    pub model: Option<String>,
    #[serde(default)]
    pub all: bool,
}

async fn resolve_audio_input(
    state: &AppState,
    seconds: Option<f32>,
    embedding: Option<Vec<f32>>,
) -> Result<(String, Vec<f32>, Option<Vec<Vec<f32>>>), ApiError> {
    if let Some(emb) = embedding {
        if emb.is_empty() {
            return Err(api_error(StatusCode::BAD_REQUEST, "'embedding' must not be empty"));
        }
        let model = state
            .engine
            .get_audio_model_name()
            .await
            .ok_or_else(|| api_error(StatusCode::CONFLICT, "No audio model loaded"))?;
        return Ok((model, emb, None));
    }
    let seconds = seconds.unwrap_or(SOUND_WINDOW_SECONDS).clamp(0.3, 10.0);
    let a = embed_current_audio(state, seconds).await?;
    Ok((a.model, a.embedding, a.patches))
}

/// POST /api/sounds - Register (or add a sample to) a sound from the microphone.
pub async fn handle_create_sound(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<CreateSoundPayload>,
) -> Result<(StatusCode, Json<GestureItem>), ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let name = payload.name.trim().to_string();
    if name.is_empty() || name.len() > 64 {
        return Err(api_error(StatusCode::BAD_REQUEST, "'name' must be 1-64 characters"));
    }
    let (model, embedding, patches) = resolve_audio_input(&state, payload.seconds, payload.embedding).await?;
    let now = now_secs();
    let item = {
        let mut store = state.sounds.write().await;
        let sound = store.get_mut_or_insert(&name, &model, payload.is_neutral, now);
        sound.add_sample(&embedding, patches.as_deref(), now).map_err(engine_error)?;
        let item = GestureItem::from(&*sound);
        if let Err(e) = store.save() {
            tracing::warn!("Could not persist sound registry: {}", e);
        }
        item
    };
    tracing::info!("Sound '{}' now has {} sample(s) for model '{}'", item.name, item.sample_count, item.model_name);
    Ok((StatusCode::CREATED, Json(item)))
}

/// GET /api/sounds
pub async fn handle_list_sounds(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<ListSoundsQuery>,
) -> Result<Json<Vec<GestureItem>>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let store = state.sounds.read().await;
    let items = if q.all {
        let mut all: Vec<_> = store.gestures.values().collect();
        all.sort_by_key(|g| g.created_at);
        all.into_iter().map(GestureItem::from).collect()
    } else {
        let model = match q.model {
            Some(m) => Some(m),
            None => state.engine.get_audio_model_name().await,
        };
        match model {
            Some(m) => store.for_model(&m).into_iter().map(GestureItem::from).collect(),
            None => Vec::new(),
        }
    };
    Ok(Json(items))
}

/// DELETE /api/sounds/{name}
pub async fn handle_delete_sound(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(name): Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let mut store = state.sounds.write().await;
    if store.remove(&name).is_none() {
        return Err(api_error(StatusCode::NOT_FOUND, format!("Sound '{name}' not found")));
    }
    if let Err(e) = store.save() {
        tracing::warn!("Could not persist sound registry: {}", e);
    }
    Ok(Json(json!({ "status": "deleted", "name": name })))
}

/// DELETE /api/sounds - Remove every sound of the audio model (or `?all=true`).
pub async fn handle_clear_sounds(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<ListSoundsQuery>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let model = match (&q.model, q.all) {
        (_, true) => None,
        (Some(m), false) => Some(m.clone()),
        (None, false) => state.engine.get_audio_model_name().await,
    };
    let mut store = state.sounds.write().await;
    let before = store.gestures.len();
    match &model {
        Some(m) => store.gestures.retain(|_, g| &g.model_name != m),
        None => store.gestures.clear(),
    }
    let removed = before - store.gestures.len();
    if let Err(e) = store.save() {
        tracing::warn!("Could not persist sound registry: {}", e);
    }
    Ok(Json(json!({ "status": "cleared", "removed": removed, "model": model })))
}

/// POST /api/sounds/match - Match the microphone (or an embedding) against the sounds.
pub async fn handle_match_sound(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(payload): Json<MatchSoundPayload>,
) -> Result<Json<GestureMatchResult>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let (model, embedding, patches) = resolve_audio_input(&state, payload.seconds, payload.embedding).await?;
    let threshold = payload.threshold.unwrap_or(SOUND_THRESHOLD).clamp(0.0, 1.0);
    let margin = payload.margin.unwrap_or(DEFAULT_MARGIN).clamp(0.0, 1.0);
    let store = state.sounds.read().await;
    let registered = store.for_model(&model);
    if registered.is_empty() {
        return Ok(Json(GestureMatchResult::empty(threshold, "No sounds registered for the audio model")));
    }
    Ok(Json(match_gestures(&embedding, patches.as_deref(), &registered, threshold, margin)))
}

// ---------------------------------------------------------------------------
// Companion
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct ModePayload {
    pub mode: CompanionMode,
}

#[derive(Debug, Deserialize)]
pub struct PosePayload {
    #[serde(flatten)]
    pub pose: CompanionPose,
}

/// A behaviour is written flattened into the payload (`"behaviour": "nod"`, or
/// `"behaviour": "pose", "pose": {...}`), or nested under `behaviour` as an object.
fn behaviour_from_value(v: &serde_json::Value) -> Result<Option<Behaviour>, ApiError> {
    let source = match v.get("behaviour") {
        None | Some(serde_json::Value::Null) => return Ok(None),
        Some(serde_json::Value::Object(_)) => v["behaviour"].clone(),
        Some(_) => v.clone(),
    };
    serde_json::from_value::<Behaviour>(source)
        .map(Some)
        .map_err(|e| api_error(StatusCode::BAD_REQUEST, format!("Invalid behaviour: {e}")))
}

fn required_behaviour(v: &serde_json::Value) -> Result<Behaviour, ApiError> {
    behaviour_from_value(v)?.ok_or_else(|| api_error(StatusCode::BAD_REQUEST, "'behaviour' is required"))
}

#[derive(Debug, Deserialize)]
pub struct CueFields {
    pub kind: CueKind,
    pub name: String,
    /// Sound cues: seconds of audio to embed.
    pub seconds: Option<f32>,
}

fn cue_fields(v: &serde_json::Value) -> Result<CueFields, ApiError> {
    serde_json::from_value::<CueFields>(v.clone())
        .map_err(|e| api_error(StatusCode::BAD_REQUEST, format!("Invalid cue: {e}")))
}

#[derive(Debug, Serialize)]
pub struct TeachResponse {
    pub cue: Cue,
    pub sample_count: usize,
    pub model: String,
    /// Gesture cues: JPEG data URI of exactly what the model saw.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thumbnail: Option<String>,
    /// Sound cues: envelope of the clip that was embedded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub waveform: Option<Vec<f32>>,
    /// Peak level of that clip (0 to 1); near zero means the microphone heard nothing.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub level: Option<f32>,
}

#[derive(Debug, Deserialize)]
pub struct WsQuery {
    pub token: Option<String>,
}

fn persist_memory(state: &AppState, core: &mut crate::companion::CompanionCore) {
    if !core.dirty {
        return;
    }
    match serde_json::to_string_pretty(&core.memory()) {
        Ok(text) => {
            if let Err(e) = std::fs::write(&state.config.companion_path, text) {
                tracing::warn!("Could not persist companion memory: {}", e);
            }
        }
        Err(e) => tracing::warn!("Could not serialize companion memory: {}", e),
    }
    core.dirty = false;
}

/// GET /api/companion/status
pub async fn handle_companion_status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<CompanionTelemetry>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    Ok(Json(state.companion.core.lock().await.telemetry()))
}

/// POST /api/companion/mode
pub async fn handle_companion_mode(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(p): Json<ModePayload>,
) -> Result<Json<CompanionTelemetry>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let mut core = state.companion.core.lock().await;
    core.set_mode(p.mode);
    Ok(Json(core.telemetry()))
}

/// POST /api/companion/pose - Manual target pose.
pub async fn handle_companion_pose(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(p): Json<PosePayload>,
) -> Result<Json<CompanionTelemetry>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let mut core = state.companion.core.lock().await;
    core.set_target(p.pose);
    Ok(Json(core.telemetry()))
}

/// POST /api/companion/behaviour - Perform a behaviour now.
pub async fn handle_companion_behaviour(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(p): Json<serde_json::Value>,
) -> Result<Json<CompanionTelemetry>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let behaviour = required_behaviour(&p)?;
    let mut core = state.companion.core.lock().await;
    core.perform(behaviour);
    Ok(Json(core.telemetry()))
}

/// GET /api/companion/cues
pub async fn handle_companion_cues(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<Cue>>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    Ok(Json(state.companion.core.lock().await.memory().cues))
}

/// PUT /api/companion/cues - Map a registered gesture or sound to a behaviour.
pub async fn handle_companion_set_cue(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(p): Json<serde_json::Value>,
) -> Result<Json<Vec<Cue>>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let fields = cue_fields(&p)?;
    let behaviour = required_behaviour(&p)?;
    let name = fields.name.trim().to_string();
    if name.is_empty() {
        return Err(api_error(StatusCode::BAD_REQUEST, "'name' must not be empty"));
    }
    let mut core = state.companion.core.lock().await;
    core.set_cue(Cue { kind: fields.kind, name, behaviour });
    persist_memory(&state, &mut core);
    Ok(Json(core.memory().cues))
}

/// DELETE /api/companion/cues/{kind}/{name}
pub async fn handle_companion_delete_cue(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((kind, name)): Path<(String, String)>,
) -> Result<Json<Vec<Cue>>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let kind = match kind.as_str() {
        "gesture" => CueKind::Gesture,
        "sound" => CueKind::Sound,
        other => return Err(api_error(StatusCode::BAD_REQUEST, format!("Unknown cue kind '{other}'"))),
    };
    let mut core = state.companion.core.lock().await;
    if !core.remove_cue(kind, &name) {
        return Err(api_error(StatusCode::NOT_FOUND, format!("No cue '{name}' of kind {kind:?}")));
    }
    persist_memory(&state, &mut core);
    Ok(Json(core.memory().cues))
}

/// POST /api/companion/teach - Register what the camera or microphone captures now
/// under `name` and map it to a behaviour (default: hold the current pose).
pub async fn handle_companion_teach(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(p): Json<serde_json::Value>,
) -> Result<(StatusCode, Json<TeachResponse>), ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let behaviour = behaviour_from_value(&p)?;
    let p = cue_fields(&p)?;
    let name = p.name.trim().to_string();
    if name.is_empty() || name.len() > 64 {
        return Err(api_error(StatusCode::BAD_REQUEST, "'name' must be 1-64 characters"));
    }
    let now = now_secs();
    let mut thumbnail = None;
    let mut waveform = None;
    let mut level = None;
    let (model, sample_count) = match p.kind {
        CueKind::Gesture => {
            let view = embed_current_view(&state).await?;
            let thumb = format!(
                "data:image/jpeg;base64,{}",
                base64::Engine::encode(&base64::prelude::BASE64_STANDARD, view.frame_jpeg.as_slice())
            );
            let mut store = state.gestures.write().await;
            let g = store.get_mut_or_insert(&name, &view.model, false, now);
            g.add_sample(&view.embedding, view.patches.as_deref(), now).map_err(engine_error)?;
            g.thumbnail = Some(thumb.clone());
            thumbnail = Some(thumb);
            let n = g.samples.len();
            if let Err(e) = store.save() {
                tracing::warn!("Could not persist gesture registry: {}", e);
            }
            (view.model, n)
        }
        CueKind::Sound => {
            let seconds = p.seconds.unwrap_or(SOUND_WINDOW_SECONDS).clamp(0.3, 10.0);
            let a = embed_current_audio(&state, seconds).await?;
            let peak = state.mic.latest_clip(seconds).map(|c| c.samples.iter().fold(0.0f32, |m, s| m.max(s.abs())));
            level = peak;
            waveform = Some(a.waveform.clone());
            let mut store = state.sounds.write().await;
            let g = store.get_mut_or_insert(&name, &a.model, false, now);
            g.add_sample(&a.embedding, a.patches.as_deref(), now).map_err(engine_error)?;
            let n = g.samples.len();
            if let Err(e) = store.save() {
                tracing::warn!("Could not persist sound registry: {}", e);
            }
            (a.model, n)
        }
    };
    let mut core = state.companion.core.lock().await;
    let behaviour = behaviour.unwrap_or(Behaviour::Pose { pose: core.target });
    let cue = Cue { kind: p.kind, name, behaviour };
    core.set_cue(cue.clone());
    persist_memory(&state, &mut core);
    // Visible "got it": the companion reacts to every lesson.
    core.perform(Behaviour::Acknowledge);
    tracing::info!("Companion learned {:?} cue '{}' -> {}", cue.kind, cue.name, cue.behaviour.label());
    Ok((StatusCode::CREATED, Json(TeachResponse { cue, sample_count, model, thumbnail, waveform, level })))
}

/// GET /api/companion/ws - Telemetry at the control rate.
pub async fn handle_companion_ws(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<WsQuery>,
    ws: WebSocketUpgrade,
) -> Result<Response, ApiError> {
    let mut headers = headers;
    if let Some(t) = q.token.as_deref() {
        if let Ok(v) = format!("Bearer {t}").parse() {
            headers.insert(axum::http::header::AUTHORIZATION, v);
        }
    }
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    Ok(ws.on_upgrade(move |socket| companion_ws_session(socket, state)))
}

async fn companion_ws_session(mut socket: WebSocket, state: AppState) {
    let mut rx = state.companion.subscribe();
    let first = rx.borrow().clone();
    if let Ok(text) = serde_json::to_string(&first) {
        if socket.send(Message::Text(text.into())).await.is_err() {
            return;
        }
    }
    loop {
        tokio::select! {
            changed = rx.changed() => {
                if changed.is_err() {
                    break;
                }
                let t = rx.borrow_and_update().clone();
                let Ok(text) = serde_json::to_string(&t) else { continue };
                if socket.send(Message::Text(text.into())).await.is_err() {
                    break;
                }
            }
            incoming = socket.recv() => {
                match incoming {
                    Some(Ok(Message::Text(text))) => {
                        if let Ok(p) = serde_json::from_str::<PosePayload>(&text) {
                            state.companion.core.lock().await.set_target(p.pose);
                        }
                    }
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                    _ => {}
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Observers: the companion's eyes and ears
// ---------------------------------------------------------------------------

/// Grayscale thumbnail of the latest camera frame for motion tracking.
async fn latest_thumbnail(state: &AppState, w: u32, h: u32) -> Option<(u64, Vec<u8>)> {
    let rb = state.ring_buffer.read().await;
    let f = rb.latest()?;
    let small = image::imageops::resize(&f.rgb_image, w, h, image::imageops::FilterType::Triangle);
    let gray: Vec<u8> =
        small.pixels().map(|p| ((p[0] as u32 * 30 + p[1] as u32 * 59 + p[2] as u32 * 11) / 100) as u8).collect();
    Some((f.sequence, gray))
}

/// Watching: motion attention at ~10 Hz and gesture recognition at ~3 Hz.
/// Listening: sound recognition at ~1.5 Hz. All only in interactive mode.
pub fn spawn_observers(state: AppState) {
    // Eyes
    {
        let state = state.clone();
        tokio::spawn(async move {
            const W: u32 = 32;
            const H: u32 = 24;
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(100));
            let mut prev: Option<(u64, Vec<u8>)> = None;
            let mut last_embed = std::time::Instant::now();
            loop {
                interval.tick().await;
                let interactive = state.companion.core.lock().await.mode == CompanionMode::Interactive;
                if !interactive || !state.camera_supervisor.is_active() {
                    prev = None;
                    continue;
                }
                if let Some((seq, cur)) = latest_thumbnail(&state, W, H).await {
                    if let Some((pseq, p)) = &prev {
                        if *pseq != seq {
                            let (x, y, m) = motion_centroid(p, &cur, W as usize, H as usize);
                            state.companion.core.lock().await.observe_motion(x, y, m);
                        }
                    }
                    prev = Some((seq, cur));
                }
                if last_embed.elapsed() < std::time::Duration::from_millis(300) {
                    continue;
                }
                last_embed = std::time::Instant::now();
                let has_gestures = {
                    let store = state.gestures.read().await;
                    match state.engine.get_active_model_name().await {
                        Some(m) => !store.for_model(&m).is_empty(),
                        None => false,
                    }
                };
                if !has_gestures {
                    continue;
                }
                match embed_current_view(&state).await {
                    Ok(view) => {
                        let result = {
                            let store = state.gestures.read().await;
                            let registered = store.for_model(&view.model);
                            match_gestures(
                                &view.embedding,
                                view.patches.as_deref(),
                                &registered,
                                DEFAULT_THRESHOLD,
                                DEFAULT_MARGIN,
                            )
                        };
                        let mut core = state.companion.core.lock().await;
                        core.observe_gestures(
                            &result.scores,
                            result.matched.as_deref(),
                            result.confidence,
                            now_secs_f64(),
                        );
                        if core.last_error.as_deref().is_some_and(|e| e.starts_with("Watching")) {
                            core.last_error = None;
                        }
                    }
                    Err((_, Json(body))) => {
                        let msg = body.get("error").and_then(|v| v.as_str()).unwrap_or("observation failed");
                        state.companion.core.lock().await.last_error = Some(format!("Watching: {msg}"));
                    }
                }
            }
        });
    }
    // Ears
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(700));
        loop {
            interval.tick().await;
            let interactive = state.companion.core.lock().await.mode == CompanionMode::Interactive;
            if !interactive || !state.mic.is_active() {
                continue;
            }
            let level = state.mic.health().level;
            let audio_model = state.engine.get_audio_model_name().await;
            let has_sounds = match &audio_model {
                Some(m) => !state.sounds.read().await.for_model(m).is_empty(),
                None => false,
            };
            if !has_sounds {
                // Nothing taught yet: only the level (startle on loud noise).
                state.companion.core.lock().await.observe_sound(level, None, 0.0, now_secs_f64());
                continue;
            }
            match embed_current_audio(&state, SOUND_WINDOW_SECONDS).await {
                Ok(a) => {
                    let result = {
                        let store = state.sounds.read().await;
                        let registered = store.for_model(&a.model);
                        match_gestures(&a.embedding, a.patches.as_deref(), &registered, SOUND_THRESHOLD, DEFAULT_MARGIN)
                    };
                    let mut core = state.companion.core.lock().await;
                    core.observe_sound(a.level, result.matched.as_deref(), result.confidence, now_secs_f64());
                    if core.last_error.as_deref().is_some_and(|e| e.starts_with("Listening")) {
                        core.last_error = None;
                    }
                }
                Err((_, Json(body))) => {
                    let msg = body.get("error").and_then(|v| v.as_str()).unwrap_or("listening failed");
                    state.companion.core.lock().await.last_error = Some(format!("Listening: {msg}"));
                }
            }
        }
    });
}
