//! Axum router configuration binding REST endpoints, SSE streams, and embedded UI.

use axum::routing::{delete, get, post};
use axum::Router;

use crate::server::gesture_handlers::{
    handle_camera_frame, handle_clear_gestures, handle_clear_roi, handle_create_gesture, handle_delete_gesture,
    handle_export_gestures, handle_get_roi, handle_import_gestures, handle_list_gestures, handle_match_gesture,
    handle_set_roi,
};
use crate::server::handlers::{
    handle_audit, handle_camera_start, handle_camera_stop, handle_cameras, handle_catalog, handle_create_key,
    handle_delete_model, handle_delete_model_root, handle_embed, handle_embed_stream, handle_energy,
    handle_get_session_token, handle_get_settings, handle_list_keys, handle_load_model, handle_pull,
    handle_register_manifest, handle_revoke_key, handle_ring_buffer, handle_save_settings, handle_status, handle_tags,
    handle_unload_model, AppState,
};
use crate::server::robot_handlers::{
    handle_robot_approve, handle_robot_clear_goal, handle_robot_estop, handle_robot_gesture_map_get,
    handle_robot_gesture_map_put, handle_robot_goal, handle_robot_joints, handle_robot_mode, handle_robot_observe,
    handle_robot_reset_safety, handle_robot_status, handle_robot_target, handle_robot_ws,
};
use crate::server::ui_assets::{serve_app_js, serve_index, serve_styles};

/// Assemble all application routes and static assets into an Axum Router
pub fn create_router(state: AppState) -> Router {
    Router::new()
        // Embedded Single-Page Application
        .route("/", get(serve_index))
        .route("/index.html", get(serve_index))
        .route("/styles.css", get(serve_styles))
        .route("/app.js", get(serve_app_js))
        // API: System & Catalog
        .route("/api/status", get(handle_status))
        .route("/api/tags", get(handle_tags))
        .route("/api/catalog", get(handle_catalog))
        .route("/api/auth/token", get(handle_get_session_token))
        .route("/api/pull", post(handle_pull))
        .route("/api/models", delete(handle_delete_model_root))
        .route("/api/models/load", post(handle_load_model))
        .route("/api/models/unload", post(handle_unload_model))
        .route("/api/models/{*name}", delete(handle_delete_model))
        .route("/api/manifests", post(handle_register_manifest))
        // API: Inference & Streaming
        .route("/api/embed", post(handle_embed))
        .route("/api/embed/stream", get(handle_embed_stream).post(handle_embed_stream))
        .route("/api/energy", post(handle_energy))
        // API: Camera & Ring Buffer
        .route("/api/cameras", get(handle_cameras))
        .route("/api/camera/start", post(handle_camera_start))
        .route("/api/camera/stop", post(handle_camera_stop))
        .route("/api/ring-buffer", get(handle_ring_buffer))
        .route("/api/camera/frame", get(handle_camera_frame))
        .route("/api/camera/roi", get(handle_get_roi).put(handle_set_roi).delete(handle_clear_roi))
        // API: Security, Keys & Audit
        .route("/api/keys", post(handle_create_key).get(handle_list_keys))
        .route("/api/keys/{prefix}", delete(handle_revoke_key))
        .route("/api/audit", get(handle_audit))
        // API: Settings
        .route("/api/settings", get(handle_get_settings).post(handle_save_settings))
        // API: Gestures (Few-Shot Latent Matching)
        .route("/api/gestures", post(handle_create_gesture).get(handle_list_gestures).delete(handle_clear_gestures))
        .route("/api/gestures/match", post(handle_match_gesture))
        .route("/api/gestures/export", get(handle_export_gestures))
        .route("/api/gestures/import", post(handle_import_gestures))
        .route("/api/gestures/{name}", delete(handle_delete_gesture))
        // API: Robot control and digital twin
        .route("/api/robot/status", get(handle_robot_status))
        .route("/api/robot/target", post(handle_robot_target))
        .route("/api/robot/joints", post(handle_robot_joints))
        .route("/api/robot/approve", post(handle_robot_approve))
        .route("/api/robot/mode", post(handle_robot_mode))
        .route("/api/robot/goal", post(handle_robot_goal).delete(handle_robot_clear_goal))
        .route("/api/robot/observe", post(handle_robot_observe))
        .route("/api/robot/e-stop", post(handle_robot_estop))
        .route("/api/robot/reset-safety", post(handle_robot_reset_safety))
        .route("/api/robot/gesture-map", get(handle_robot_gesture_map_get).put(handle_robot_gesture_map_put))
        .route("/api/robot/ws", get(handle_robot_ws))
        // Fallback for SPA routing
        .fallback(serve_index)
        .with_state(state)
}
