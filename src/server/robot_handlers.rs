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
    if p.mode == RobotMode::GoalSeeking {
        core.explorer.resume();
    }
    Ok(Json(core.telemetry()))
}

#[derive(Debug, Deserialize, Default)]
pub struct ObservationPayload {
    /// Snapshot of the WebGL twin (or any image). Without it the server camera is used.
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

/// POST /api/robot/goal - Capture the current visual state as the latent goal.
pub async fn handle_robot_goal(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Option<Json<ObservationPayload>>,
) -> Result<Json<RobotTelemetry>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let z = embed_observation(&state, body.and_then(|b| b.0.image_base64)).await?;
    let mut core = state.robot.core.lock().await;
    let pose = core.joints;
    core.explorer.set_goal(z, pose);
    tracing::info!("Latent goal captured ({} dims)", core.explorer.goal.as_ref().map_or(0, |g| g.len()));
    Ok(Json(core.telemetry()))
}

/// DELETE /api/robot/goal - Forget the goal.
pub async fn handle_robot_clear_goal(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<RobotTelemetry>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let mut core = state.robot.core.lock().await;
    core.explorer.clear();
    Ok(Json(core.telemetry()))
}

/// POST /api/robot/observe - Feed one observation to the goal seeker (Mode C).
/// Accepted only when the arm is settled at the explorer's pose.
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
    let next = core.observe(&z);
    Ok(Json(json!({ "accepted": true, "next_pose": next, "goal": core.telemetry().goal })))
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
