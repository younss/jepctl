//! Live latent reconstruction of the camera scene.
//!
//! The vision model embeds the current camera frame into one vector per patch (the
//! ViT grid). This turns that grid into a small field the UI renders as a live WebGL
//! terrain: one column per patch, its **height** the salience of the patch (how far
//! it stands out from the frame's average patch) and its **colour** a fixed
//! projection of the patch embedding. The result is not a photograph and not a
//! metric 3D scan: it is what the model perceives, rebuilt in space as you move
//! things in front of the camera. The real per-patch pixel colour is also returned
//! so the UI can offer a recognisable tint.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::Json;
use base64::prelude::*;
use serde::Serialize;

use crate::server::handlers::{ApiError, AppState, api_error, embed_current_view};
use crate::server::middleware::authenticate_request;
use crate::types::{Role, normalize_l2};

/// Cache of the fixed `dim x 3` colour projection, one per embedding dimension.
fn projection(dim: usize) -> &'static Vec<f32> {
    static CACHE: OnceLock<Mutex<HashMap<usize, &'static Vec<f32>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(m) = guard.get(&dim) {
        return m;
    }
    // Deterministic so colours are stable across restarts and machines.
    let mut state: u64 = 0x9E3779B97F4A7C15 ^ (dim as u64);
    let mut next = || {
        state = state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^= z >> 31;
        (z as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
    };
    let matrix: Vec<f32> = (0..dim * 3).map(|_| next()).collect();
    let leaked: &'static Vec<f32> = Box::leak(Box::new(matrix));
    guard.insert(dim, leaked);
    leaked
}

/// One patch mapped to a colour and a height.
#[derive(Serialize)]
pub struct WorldFrame {
    pub model: String,
    pub grid_w: usize,
    pub grid_h: usize,
    /// Latent colour per patch, row major, `grid_w * grid_h * 3` values in `[0, 1]`.
    pub colors: Vec<f32>,
    /// Actual average pixel colour per patch, same layout (recognisable tint).
    pub pixels: Vec<f32>,
    /// Foreground relief per patch, `grid_w * grid_h` values in `[0, 1]`: how far the
    /// patch is from the background prototype, smoothed. This is the geometry.
    pub heights: Vec<f32>,
    /// The camera frame as a JPEG data URI, aligned with the grid: the UI drapes it
    /// over the relief as a texture, so the surface shows the real scene.
    pub image: String,
    /// Texture aspect ratio (width / height) so the UI mesh matches the real field of
    /// view instead of forcing a square.
    pub aspect: f32,
    // --- Online world model (predict-next + surprise), see engine::scene_predictor ---
    /// Prediction error of the frame the model expected last tick, 0 to 1.
    pub surprise: f32,
    /// How well the current scene matches something already seen, 0 to 1.
    pub recognition: f32,
    /// Distinct states the model has learned so far.
    pub known_states: usize,
    /// `learning`, `recognized` or `surprised`.
    pub state: String,
    /// Per-patch prediction error (surprise map), row major, 0 to 1.
    pub surprise_map: Vec<f32>,
    /// Predicted next per-patch relief (what the model expects next), 0 to 1.
    pub predicted_heights: Vec<f32>,
    pub steps: u64,
    /// Nearest saved named state for the current view, if close enough.
    pub recognized_label: Option<String>,
    pub recognized_conf: f32,
    /// Names of the saved states.
    pub snapshots: Vec<String>,
    pub latency_ms: f64,
}

/// GET /api/world/frame - the current camera scene as a latent field.
pub async fn handle_world_frame(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<WorldFrame>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let started = std::time::Instant::now();

    // Full field of view: for image models embed the whole frame (stretched to the
    // model input), not the centre square crop, and texture with the full frame at
    // its real aspect ratio. Video models keep the centre-crop path.
    let modality = state.engine.get_active_modality().await;
    let model_name = crate::server::handlers::ensure_model_loaded(&state).await?;
    let prep = state.engine.preprocessing().await;
    let (raw_patches, model, full_jpeg, aspect) = if modality == Some(crate::types::ModelModality::Image) {
        let tensor = {
            let rb = state.ring_buffer.read().await;
            rb.latest_full_image_tensor(&prep, &state.engine.device).map_err(|_| {
                api_error(axum::http::StatusCode::CONFLICT, "Camera is not running or no frame captured yet")
            })?
        };
        let (_m, _dim, _emb, patches, _lat) =
            state.engine.embed_image(&tensor).await.map_err(crate::server::handlers::engine_error)?;
        let (jpeg, _seq, w, h) = {
            let rb = state.ring_buffer.read().await;
            rb.latest_full_frame_jpeg(720).ok_or_else(|| {
                api_error(axum::http::StatusCode::CONFLICT, "Camera is not running or no frame captured yet")
            })?
        };
        state.embeddings_total.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let aspect = if h > 0 { w as f32 / h as f32 } else { 1.0 };
        (patches, model_name, jpeg, aspect)
    } else {
        let view = embed_current_view(&state).await?;
        (view.patches, view.model, view.frame_jpeg.as_ref().clone(), 1.0)
    };
    let patches = raw_patches.filter(|p| !p.is_empty()).ok_or_else(|| {
        api_error(axum::http::StatusCode::CONFLICT, "The active model does not expose per-patch tokens")
    })?;

    let n = patches.len();
    let dim = patches[0].len();
    // Assume a square grid (every catalogued vision model has square input); fall
    // back to a single row when the count is not a perfect square.
    let side = (n as f64).sqrt().round() as usize;
    let (grid_w, grid_h) = if side * side == n { (side, side) } else { (n, 1) };

    // Background prototype: the mean of the patches that look most alike (the bulk of
    // the frame is background). Distance from it is how much a patch belongs to the
    // foreground, which is where JEPA earns its keep: a person or an object sits far
    // from a flat wall in embedding space even when their pixels are not that
    // different. This gives geometry that tracks the real object, not just edges.
    let mut centroid = vec![0.0f32; dim];
    for p in &patches {
        for (c, v) in centroid.iter_mut().zip(p.iter()) {
            *c += v;
        }
    }
    for c in centroid.iter_mut() {
        *c /= n as f32;
    }
    let mut cdist: Vec<f32> =
        patches.iter().map(|p| p.iter().zip(&centroid).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt()).collect();
    let mut sorted = cdist.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2].max(1e-6);
    let mut bg = vec![0.0f32; dim];
    let mut bg_count = 0usize;
    for (p, &d) in patches.iter().zip(&cdist) {
        if d <= median {
            for (b, v) in bg.iter_mut().zip(p.iter()) {
                *b += v;
            }
            bg_count += 1;
        }
    }
    for b in bg.iter_mut() {
        *b /= bg_count.max(1) as f32;
    }

    let proj = projection(dim);
    let mut colors = Vec::with_capacity(n * 3);
    let mut raw = Vec::with_capacity(n);
    let mut max_dist = 1e-6f32;
    for p in &patches {
        let dist: f32 = p.iter().zip(&bg).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
        max_dist = max_dist.max(dist);
        raw.push(dist);
        let unit = normalize_l2(p);
        for k in 0..3 {
            let mut acc = 0.0f32;
            for (i, &u) in unit.iter().enumerate() {
                acc += u * proj[i * 3 + k];
            }
            colors.push((acc * 2.0).tanh() * 0.5 + 0.5);
        }
    }
    for r in raw.iter_mut() {
        *r /= max_dist;
    }
    let _ = &mut cdist;
    // Smooth the relief so the draped surface is continuous, not stepped.
    let heights = smooth_grid(&raw, grid_w, grid_h);
    // Online world model: pooled embedding is the mean patch; the predictor learns to
    // forecast the next embedding and reports its surprise and recognition.
    let mut pooled = vec![0.0f32; dim];
    for p in &patches {
        for (c, v) in pooled.iter_mut().zip(p.iter()) {
            *c += v;
        }
    }
    for c in pooled.iter_mut() {
        *c /= n as f32;
    }
    let report = {
        let mut scene = state.scene.lock().await;
        scene.step(&pooled, &patches, grid_w, grid_h)
    };

    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;
    let pixels = average_patch_colors(&full_jpeg, grid_w, grid_h);
    let image = format!("data:image/jpeg;base64,{}", BASE64_STANDARD.encode(&full_jpeg));

    Ok(Json(WorldFrame {
        model,
        grid_w,
        grid_h,
        colors,
        pixels,
        heights,
        image,
        aspect,
        surprise: report.surprise,
        recognition: report.recognition,
        known_states: report.known_states,
        state: report.state.to_string(),
        surprise_map: report.surprise_map,
        predicted_heights: report.predicted_heights,
        steps: report.steps,
        recognized_label: report.recognized_label,
        recognized_conf: report.recognized_conf,
        snapshots: report.snapshots,
        latency_ms,
    }))
}

/// 3x3 box blur over a `w x h` grid (edges clamp).
fn smooth_grid(v: &[f32], w: usize, h: usize) -> Vec<f32> {
    if w == 0 || h == 0 || v.len() != w * h {
        return v.to_vec();
    }
    let mut out = vec![0.0f32; v.len()];
    for y in 0..h {
        for x in 0..w {
            let mut sum = 0.0f32;
            let mut cnt = 0.0f32;
            for dy in -1i32..=1 {
                for dx in -1i32..=1 {
                    let nx = x as i32 + dx;
                    let ny = y as i32 + dy;
                    if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                        sum += v[ny as usize * w + nx as usize];
                        cnt += 1.0;
                    }
                }
            }
            out[y * w + x] = sum / cnt;
        }
    }
    out
}

/// Average RGB of the model-view frame in each grid cell, `grid_w * grid_h * 3` in `[0, 1]`.
fn average_patch_colors(jpeg: &[u8], grid_w: usize, grid_h: usize) -> Vec<f32> {
    let fallback = vec![0.5f32; grid_w * grid_h * 3];
    let Ok(img) = image::load_from_memory(jpeg) else {
        return fallback;
    };
    let small =
        image::imageops::resize(&img.to_rgb8(), grid_w as u32, grid_h as u32, image::imageops::FilterType::Triangle);
    let mut out = Vec::with_capacity(grid_w * grid_h * 3);
    for p in small.pixels() {
        out.push(p[0] as f32 / 255.0);
        out.push(p[1] as f32 / 255.0);
        out.push(p[2] as f32 / 255.0);
    }
    if out.len() == grid_w * grid_h * 3 { out } else { fallback }
}

/// POST /api/world/snapshot - save the current view under a name (learning snapshot).
#[derive(serde::Deserialize)]
pub struct SnapshotPayload {
    pub name: String,
}

pub async fn handle_world_snapshot(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(p): Json<SnapshotPayload>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let name = p.name.trim().to_string();
    if name.is_empty() || name.len() > 48 {
        return Err(api_error(axum::http::StatusCode::BAD_REQUEST, "'name' must be 1-48 characters"));
    }
    let mut scene = state.scene.lock().await;
    if !scene.save_snapshot(&name) {
        return Err(api_error(axum::http::StatusCode::CONFLICT, "No frame observed yet; start the camera first"));
    }
    Ok(Json(serde_json::json!({ "status": "saved", "name": name, "snapshots": scene.snapshot_names() })))
}

/// DELETE /api/world/snapshot/{name}
pub async fn handle_world_snapshot_delete(
    State(state): State<AppState>,
    headers: HeaderMap,
    axum::extract::Path(name): axum::extract::Path<String>,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    let mut scene = state.scene.lock().await;
    let removed = scene.delete_snapshot(&name);
    Ok(Json(
        serde_json::json!({ "status": if removed { "deleted" } else { "not_found" }, "snapshots": scene.snapshot_names() }),
    ))
}

/// POST /api/world/reset - forget the learned dynamics and recognition memory.
pub async fn handle_world_reset(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<serde_json::Value>, ApiError> {
    let _ = authenticate_request(&headers, &state.auth, Role::Inference).await?;
    state.scene.lock().await.reset();
    Ok(Json(serde_json::json!({ "status": "reset" })))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_is_stable_and_sized() {
        let a = projection(384);
        let b = projection(384);
        assert_eq!(a.len(), 384 * 3);
        assert!(std::ptr::eq(a.as_slice(), b.as_slice()), "same dim returns the cached matrix");
        assert_eq!(projection(768).len(), 768 * 3);
    }

    #[test]
    fn patch_colors_average_the_frame() {
        // A 2x2 red/green/blue/white image, resized to 2x2, keeps the corners.
        let mut img = image::RgbImage::new(2, 2);
        img.put_pixel(0, 0, image::Rgb([255, 0, 0]));
        img.put_pixel(1, 0, image::Rgb([0, 255, 0]));
        img.put_pixel(0, 1, image::Rgb([0, 0, 255]));
        img.put_pixel(1, 1, image::Rgb([255, 255, 255]));
        let mut jpeg = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut jpeg), image::ImageFormat::Png)
            .unwrap();
        let cells = average_patch_colors(&jpeg, 2, 2);
        assert_eq!(cells.len(), 12);
        assert!(cells[0] > 0.5 && cells[1] < 0.5, "top-left is reddish");
    }
}
