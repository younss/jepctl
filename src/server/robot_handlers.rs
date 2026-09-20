//! HTTP and WebSocket handlers for the robot subsystem.

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{Json, Response};
use base64::prelude::*;
use serde::Deserialize;
use serde_json::json;

use crate::media::image::preprocess_image_bytes;
use crate::robot::hal::BackendKind;
use crate::robot::{GestureAction, JointCommand, RobotError, RobotMode, RobotTelemetry, DOF};
use crate::server::handlers::{api_error, embed_current_view, engine_error, ensure_model_loaded, ApiError, AppState};
use crate::server::middleware::authenticate_request;
use crate::types::Role;

fn robot_error(e: RobotError) -> ApiError {
    let status = match &e {
        RobotError::EStopEngaged => StatusCode::LOCKED,
        RobotError::JointOutOfRange { .. } | RobotError::GripperOutOfRange(_) | RobotError::Invalid(_) => {
            StatusCode::BAD_REQUEST
        }
        RobotError::FeatureMissing(_) => StatusCode::NOT_IMPLEMENTED,
        RobotError::NotConnected(_) | RobotError::Serial(_) => StatusCode::SERVICE_UNAVAILABLE,
    };
    api_error(status, e.to_string())
}

/// GET /api/robot/status
pub async fn handle_robot_status(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<RobotTelemetry>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    Ok(Json(state.robot.core.lock().await.telemetry()))
}

#[derive(Debug, Deserialize)]
pub struct TargetPayload {
    pub backend: BackendKind,
}

/// POST /api/robot/target - Select the execution backend.
pub async fn handle_robot_target(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(p): Json<TargetPayload>,
) -> Result<Json<RobotTelemetry>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Admin).await?;
    let mut core = state.robot.core.lock().await;
    core.switch_backend(p.backend).map_err(robot_error)?;
    tracing::info!("Robot backend switched to {:?}", p.backend);
    Ok(Json(core.telemetry()))
}

#[derive(Debug, Deserialize)]
pub struct JointsPayload {
    pub joints: [f32; DOF],
    pub gripper: f32,
    #[serde(default)]
    pub approved: bool,
}

/// POST /api/robot/joints - Direct joint command (held by the safety gate when needed).
pub async fn handle_robot_joints(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(p): Json<JointsPayload>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let mut core = state.robot.core.lock().await;
    let executed =
        core.submit(JointCommand { joints: p.joints, gripper: p.gripper }, p.approved).map_err(robot_error)?;
    Ok(Json(
        json!({ "executed": executed, "pending": core.pending, "targets": core.targets, "gripper_target": core.gripper_target }),
    ))
}

/// POST /api/robot/approve - Execute the command held by the safety gate.
pub async fn handle_robot_approve(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let mut core = state.robot.core.lock().await;
    let executed = core.approve_pending().map_err(robot_error)?;
    Ok(Json(json!({ "executed": executed })))
}

#[derive(Debug, Deserialize)]
pub struct ModePayload {
    pub mode: RobotMode,
    /// Mode B toggle; cannot be disabled while the physical backend is active.
    pub safety_gate: Option<bool>,
}

/// POST /api/robot/mode
pub async fn handle_robot_mode(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(p): Json<ModePayload>,
) -> Result<Json<RobotTelemetry>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let mut core = state.robot.core.lock().await;
    core.mode = p.mode;
    if let Some(gate) = p.safety_gate {
        core.safety_gate = gate || core.backend.kind() == BackendKind::Physical;
    }
    if core.learning_mode() {
        core.agent.start();
    } else {
        core.agent.stop();
    }
    Ok(Json(core.telemetry()))
}

#[derive(Debug, Deserialize, Default)]
pub struct ObservationPayload {
    /// Optional image instead of the server camera (replay, tests, external cameras).
    pub image_base64: Option<String>,
}

async fn embed_observation(state: &AppState, image_base64: Option<String>) -> Result<Vec<f32>, ApiError> {
    match image_base64 {
        Some(b64) => {
            let payload = b64.rsplit(',').next().unwrap_or(&b64);
            let bytes = BASE64_STANDARD
                .decode(payload.trim())
                .map_err(|e| api_error(StatusCode::BAD_REQUEST, format!("Invalid base64 image: {e}")))?;
            ensure_model_loaded(state).await?;
            let prep = state.engine.preprocessing().await;
            let tensor = preprocess_image_bytes(&bytes, &prep, &state.engine.device)
                .map_err(|e| api_error(StatusCode::BAD_REQUEST, format!("Image preprocessing failed: {e}")))?;
            let (_m, _d, emb, _p, _lat) = state.engine.embed_image(&tensor).await.map_err(engine_error)?;
            Ok(emb)
        }
        None => Ok(embed_current_view(state).await?.embedding),
    }
}

fn now_secs() -> u64 {
    std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs()
}

/// POST /api/robot/goal - Capture what the camera sees now as the latent goal.
pub async fn handle_robot_goal(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Option<Json<ObservationPayload>>,
) -> Result<Json<RobotTelemetry>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let z = embed_observation(&state, body.and_then(|b| b.0.image_base64)).await?;
    let mut core = state.robot.core.lock().await;
    let dims = z.len();
    core.agent.set_goal(z);
    tracing::info!("Latent goal captured from the camera ({} dims)", dims);
    Ok(Json(core.telemetry()))
}

/// DELETE /api/robot/goal - Forget the goal (the learned world model is kept).
pub async fn handle_robot_clear_goal(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<RobotTelemetry>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let mut core = state.robot.core.lock().await;
    core.agent.clear_goal();
    Ok(Json(core.telemetry()))
}

/// POST /api/robot/observe - Feed one observation to the agent. The server camera does
/// this automatically while a learning mode is active; this endpoint exists for
/// replays and external cameras. Accepted only when the arm is settled.
pub async fn handle_robot_observe(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Option<Json<ObservationPayload>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    {
        let core = state.robot.core.lock().await;
        let t = core.telemetry();
        if !t.goal.awaiting_observation {
            return Ok(Json(json!({ "accepted": false, "reason": "not awaiting an observation", "goal": t.goal })));
        }
    }
    let z = embed_observation(&state, body.and_then(|b| b.0.image_base64)).await?;
    let mut core = state.robot.core.lock().await;
    let next = core.observe(&z, now_secs());
    let goal = core.telemetry().goal;
    persist_world_model_if_due(&state, &core);
    Ok(Json(json!({ "accepted": true, "command": next, "goal": goal })))
}

/// GET /api/robot/world-model - Learned transitions summary.
pub async fn handle_robot_world_model(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let core = state.robot.core.lock().await;
    let w = &core.agent.world;
    Ok(Json(json!({
        "stats": w.stats(),
        "recent": w.transitions.iter().rev().take(20).map(|t| json!({ "action": t.action, "joints": t.joints, "timestamp": t.timestamp })).collect::<Vec<_>>(),
        "path": state.config.world_model_path,
    })))
}

/// DELETE /api/robot/world-model - Forget everything learned.
pub async fn handle_robot_world_model_clear(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Admin).await?;
    let mut core = state.robot.core.lock().await;
    core.agent.world = crate::robot::world_model::LatentWorldModel::default();
    let _ = std::fs::remove_file(&state.config.world_model_path);
    Ok(Json(json!({ "status": "cleared" })))
}

fn persist_world_model_if_due(state: &AppState, core: &crate::robot::RobotCore) {
    let n = core.agent.world.transitions.len();
    if n == 0 || !n.is_multiple_of(10) {
        return;
    }
    match serde_json::to_string(&core.agent.world) {
        Ok(json) => {
            if let Err(e) = std::fs::write(&state.config.world_model_path, json) {
                tracing::warn!("Could not persist world model: {}", e);
            }
        }
        Err(e) => tracing::warn!("Could not serialise world model: {}", e),
    }
}

/// Camera observer: while a learning mode is active, embeds the camera frame each
/// time the arm settles and feeds it to the agent. This is the loop that makes the
/// robot learn from its environment; nothing here looks at the WebGL twin.
pub fn spawn_camera_observer(state: AppState) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_millis(150));
        let mut warned_camera = false;
        loop {
            interval.tick().await;
            let awaiting = {
                let core = state.robot.core.lock().await;
                core.telemetry().goal.awaiting_observation
            };
            if !awaiting {
                continue;
            }
            if !state.camera_supervisor.is_active() {
                if !warned_camera {
                    warned_camera = true;
                    let mut core = state.robot.core.lock().await;
                    core.last_error = Some("Learning needs the camera: start it in Live or Gestures".into());
                }
                continue;
            }
            warned_camera = false;
            match embed_current_view(&state).await {
                Ok(view) => {
                    let mut core = state.robot.core.lock().await;
                    if core.last_error.as_deref().is_some_and(|e| e.starts_with("Learning needs the camera")) {
                        core.last_error = None;
                    }
                    let _ = core.observe(&view.embedding, now_secs());
                    persist_world_model_if_due(&state, &core);
                }
                Err((_, Json(body))) => {
                    let msg = body.get("error").and_then(|v| v.as_str()).unwrap_or("observation failed").to_string();
                    let mut core = state.robot.core.lock().await;
                    core.last_error = Some(msg);
                }
            }
        }
    });
}

/// POST /api/robot/e-stop
pub async fn handle_robot_estop(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<RobotTelemetry>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let mut core = state.robot.core.lock().await;
    core.emergency_stop();
    tracing::warn!("Robot emergency stop engaged");
    Ok(Json(core.telemetry()))
}

/// POST /api/robot/reset-safety
pub async fn handle_robot_reset_safety(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<RobotTelemetry>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Admin).await?;
    let mut core = state.robot.core.lock().await;
    core.reset_safety();
    Ok(Json(core.telemetry()))
}

/// GET /api/robot/gesture-map
pub async fn handle_robot_gesture_map_get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<std::collections::HashMap<String, GestureAction>>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    Ok(Json(state.robot.core.lock().await.gesture_map.clone()))
}

/// PUT /api/robot/gesture-map - Replace the gesture to action mapping.
pub async fn handle_robot_gesture_map_put(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(map): Json<std::collections::HashMap<String, GestureAction>>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    for (name, action) in &map {
        if let GestureAction::JointDelta { joint, .. } = action {
            if *joint >= DOF {
                return Err(api_error(
                    StatusCode::BAD_REQUEST,
                    format!("'{name}': joint index {joint} out of range (0..{DOF})"),
                ));
            }
        }
    }
    let n = map.len();
    state.robot.core.lock().await.gesture_map = map;
    Ok(Json(json!({ "status": "saved", "entries": n })))
}

#[derive(Debug, Deserialize)]
pub struct WsQuery {
    pub token: Option<String>,
}

/// GET /api/robot/ws - Telemetry at the control rate; accepts joint commands back.
pub async fn handle_robot_ws(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(q): Query<WsQuery>,
    ws: WebSocketUpgrade,
) -> Result<Response, ApiError> {
    // Browsers cannot set headers on a WebSocket: accept `?token=` like the SSE stream.
    let mut headers = headers;
    if let Some(t) = q.token.as_deref() {
        if let Ok(v) = format!("Bearer {t}").parse() {
            headers.insert(axum::http::header::AUTHORIZATION, v);
        }
    }
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    Ok(ws.on_upgrade(move |socket| robot_ws_session(socket, state)))
}

async fn robot_ws_session(mut socket: WebSocket, state: AppState) {
    let mut rx = state.robot.subscribe();
    // Send the current state immediately, then every change.
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
                        if let Ok(cmd) = serde_json::from_str::<JointsPayload>(&text) {
                            let mut core = state.robot.core.lock().await;
                            if let Err(e) = core.submit(JointCommand { joints: cmd.joints, gripper: cmd.gripper }, cmd.approved) {
                                core.last_error = Some(e.to_string());
                            }
                        }
                    }
                    Some(Ok(Message::Close(_))) | None | Some(Err(_)) => break,
                    _ => {}
                }
            }
        }
    }
}
