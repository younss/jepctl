//! Online latent world model for the live camera, in the spirit of LeWorldModel
//! (Maes, Le Lidec, Scieur, LeCun, Balestriero, arXiv:2603.19312).
//!
//! LeWM freezes nothing: it trains an encoder and a predictor end to end from pixels
//! with a next-embedding MSE loss and the SIGReg anti-collapse regulariser. Here the
//! encoder is the frozen JEPA vision model already loaded in jepctl (DINOv2, I-JEPA,
//! V-JEPA 2); on top of its embeddings we fit a *predictor online*, from the live
//! stream, with the same idea: predict the next embedding, and read the prediction
//! error as **surprise** (LeWM's surprise evaluation, which flags implausible events).
//! We also keep a small memory of seen states so the model can say whether it
//! **recognises** the current scene. No training, no labels, no checkpoint: it learns
//! the dynamics of whatever the camera shows, on its own, as it watches.

/// One patch grid (row major) of embedding vectors.
type Grid = Vec<Vec<f32>>;

/// Number of random projection axes the global predictor works in. The pooled
/// embedding is projected to this many dimensions (Johnson-Lindenstrauss: distances
/// are preserved), which keeps the online fit cheap and stable.
const PROJ_DIM: usize = 48;

/// Recognition memory: a state is remembered when it is at least this different from
/// everything already known (cosine similarity below the threshold).
const NOVELTY_COS: f32 = 0.86;
const MEMORY_CAP: usize = 96;

/// What the predictor reports for one observed frame.
#[derive(Debug, Clone, Default)]
pub struct SceneReport {
    /// Global prediction error of the frame the model expected last tick, 0 to 1.
    pub surprise: f32,
    /// How well the current scene matches something already seen, 0 to 1.
    pub recognition: f32,
    /// Distinct states remembered so far.
    pub known_states: usize,
    /// `learning`, `recognized` or `surprised`.
    pub state: &'static str,
    /// Per-patch surprise (prediction error map), row major, 0 to 1. Empty until two
    /// frames have been seen.
    pub surprise_map: Vec<f32>,
    /// Predicted next per-patch relief (what the model expects to see next), 0 to 1.
    pub predicted_heights: Vec<f32>,
    /// Frames observed since the last reset.
    pub steps: u64,
    /// Best matching *named* snapshot for the current view, if any is close enough.
    pub recognized_label: Option<String>,
    /// Confidence (cosine similarity) of that match, 0 to 1.
    pub recognized_conf: f32,
    /// Names of the saved snapshots, in save order.
    pub snapshots: Vec<String>,
    /// The current pooled state under a fixed random projection to three dimensions.
    /// This is an honest view of the latent space itself: unlike a per-patch relief it
    /// claims no geometry, it is the same Johnson-Lindenstrauss projection the
    /// predictor already works in, truncated to what a screen can show.
    pub latent: [f32; 3],
    /// Where the predictor expects the next state to land, same projection.
    pub predicted_latent: [f32; 3],
    /// Remembered states, same projection, in the order they were learned.
    pub memory_latent: Vec<[f32; 3]>,
    /// Named snapshots, same projection, aligned with `snapshots`.
    pub snapshot_latent: Vec<[f32; 3]>,
}

/// Per-dimension online AR(1) predictor `x_next = a * x + c`, fit with decayed least
/// squares. One instance covers the whole projected vector.
struct Ar1 {
    /// Decayed sufficient statistics per dimension: n, sx, sy, sxx, sxy.
    n: f32,
    sx: Vec<f32>,
    sy: Vec<f32>,
    sxx: Vec<f32>,
    sxy: Vec<f32>,
    decay: f32,
}

impl Ar1 {
    fn new(k: usize) -> Self {
        Self { n: 0.0, sx: vec![0.0; k], sy: vec![0.0; k], sxx: vec![0.0; k], sxy: vec![0.0; k], decay: 0.97 }
    }

    /// Record a transition x -> y (both length k).
    fn observe(&mut self, x: &[f32], y: &[f32]) {
        self.n = self.n * self.decay + 1.0;
        for i in 0..x.len() {
            self.sx[i] = self.sx[i] * self.decay + x[i];
            self.sy[i] = self.sy[i] * self.decay + y[i];
            self.sxx[i] = self.sxx[i] * self.decay + x[i] * x[i];
            self.sxy[i] = self.sxy[i] * self.decay + x[i] * y[i];
        }
    }

    /// Predict y from x per dimension. Falls back to persistence until enough data.
    fn predict(&self, x: &[f32]) -> Vec<f32> {
        if self.n < 4.0 {
            return x.to_vec();
        }
        let n = self.n;
        (0..x.len())
            .map(|i| {
                let var = (self.sxx[i] - self.sx[i] * self.sx[i] / n).max(1e-6);
                let cov = self.sxy[i] - self.sx[i] * self.sy[i] / n;
                let a = (cov / var).clamp(-1.5, 1.5);
                let c = (self.sy[i] - a * self.sx[i]) / n;
                a * x[i] + c
            })
            .collect()
    }
}

/// Online latent world model over one JEPA encoder's embeddings.
pub struct WorldScenePredictor {
    dim: usize,
    grid_w: usize,
    grid_h: usize,
    proj: Vec<f32>, // dim x PROJ_DIM, deterministic
    ar1: Ar1,
    /// Projection of the previous frame's pooled embedding, and the prediction made
    /// last tick for the current one.
    prev_proj: Option<Vec<f32>>,
    pred_proj: Option<Vec<f32>>,
    /// Recent per-patch grids (t-1 and t) for the velocity predictor, and the per-patch
    /// prediction made last tick.
    prev_grid: Option<Grid>,
    last_grid: Option<Grid>,
    pred_grid: Option<Grid>,
    /// Running scale of the prediction error, for normalising surprise to 0..1.
    err_scale: f32,
    /// Remembered unit pooled embeddings (recognition memory).
    memory: Vec<Vec<f32>>,
    /// Last observed unit pooled embedding (for saving a named snapshot).
    last_pooled_unit: Option<Vec<f32>>,
    /// User-named states: (label, unit pooled embedding).
    snapshots: Vec<(String, Vec<f32>)>,
    steps: u64,
}

impl Default for WorldScenePredictor {
    fn default() -> Self {
        Self::new()
    }
}

fn unit(v: &[f32]) -> Vec<f32> {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
    v.iter().map(|x| x / n).collect()
}

fn cos(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>().sqrt()
}

impl WorldScenePredictor {
    pub fn new() -> Self {
        Self {
            dim: 0,
            grid_w: 0,
            grid_h: 0,
            proj: Vec::new(),
            ar1: Ar1::new(PROJ_DIM),
            prev_proj: None,
            pred_proj: None,
            prev_grid: None,
            last_grid: None,
            pred_grid: None,
            err_scale: 1e-3,
            memory: Vec::new(),
            last_pooled_unit: None,
            snapshots: Vec::new(),
            steps: 0,
        }
    }

    /// Deterministic dim x PROJ_DIM matrix (stable across restarts).
    fn build_proj(dim: usize) -> Vec<f32> {
        let mut state: u64 = 0xD1B54A32D192ED03 ^ (dim as u64);
        let mut next = || {
            state = state.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^= z >> 31;
            (z as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        };
        (0..dim * PROJ_DIM).map(|_| next()).collect()
    }

    fn project(&self, pooled_unit: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; PROJ_DIM];
        for (i, &u) in pooled_unit.iter().enumerate() {
            let base = i * PROJ_DIM;
            for (k, o) in out.iter_mut().enumerate() {
                *o += u * self.proj[base + k];
            }
        }
        out
    }

    /// The first three coordinates of `project`, computed without building the whole
    /// 48-dimensional vector. Used for the display projection of the recognition
    /// memory, which would otherwise cost a full projection per remembered state and
    /// per frame.
    fn project3(&self, pooled_unit: &[f32]) -> [f32; 3] {
        let mut out = [0.0f32; 3];
        for (i, &u) in pooled_unit.iter().enumerate() {
            let base = i * PROJ_DIM;
            for (k, o) in out.iter_mut().enumerate() {
                *o += u * self.proj[base + k];
            }
        }
        out
    }

    /// Reset when the encoder (dimension) or the grid changes.
    fn ensure_shape(&mut self, dim: usize, grid_w: usize, grid_h: usize) {
        if self.dim != dim || self.grid_w != grid_w || self.grid_h != grid_h {
            *self = Self::new();
            self.dim = dim;
            self.grid_w = grid_w;
            self.grid_h = grid_h;
            self.proj = Self::build_proj(dim);
        }
    }

    /// Background prototype of a grid: the mean of the patches closest to the grid
    /// centroid (the bulk of a scene is background). Foreground relief is the distance
    /// from it. Same signal the World reconstruction uses.
    fn relief(grid: &Grid) -> Vec<f32> {
        let n = grid.len();
        if n == 0 {
            return Vec::new();
        }
        let dim = grid[0].len();
        let mut centroid = vec![0.0f32; dim];
        for p in grid {
            for (c, v) in centroid.iter_mut().zip(p) {
                *c += v;
            }
        }
        for c in centroid.iter_mut() {
            *c /= n as f32;
        }
        let mut d: Vec<f32> = grid.iter().map(|p| l2(p, &centroid)).collect();
        let mut sorted = d.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = sorted[sorted.len() / 2].max(1e-6);
        let mut bg = vec![0.0f32; dim];
        let mut cnt = 0.0f32;
        for (p, &dd) in grid.iter().zip(&d) {
            if dd <= median {
                for (b, v) in bg.iter_mut().zip(p) {
                    *b += v;
                }
                cnt += 1.0;
            }
        }
        for b in bg.iter_mut() {
            *b /= cnt.max(1.0);
        }
        let mut max = 1e-6f32;
        for (r, p) in d.iter_mut().zip(grid) {
            *r = l2(p, &bg);
            max = max.max(*r);
        }
        for r in d.iter_mut() {
            *r /= max;
        }
        d
    }

    /// Observe one frame (pooled embedding + per-patch grid) and advance the model.
    pub fn step(&mut self, pooled: &[f32], grid: &Grid, grid_w: usize, grid_h: usize) -> SceneReport {
        self.ensure_shape(pooled.len(), grid_w, grid_h);
        self.steps += 1;
        let pu = unit(pooled);
        let proj = self.project(&pu);

        // Surprise: how much larger the prediction error is than the recent typical
        // error. Baseline stays near zero for a scene the model tracks well and spikes
        // when reality diverges from what it expected (LeWM's surprise signal).
        let mut surprise = 0.0f32;
        if let Some(pred) = &self.pred_proj {
            let err = l2(pred, &proj);
            if self.err_scale <= 1e-3 {
                self.err_scale = err.max(1e-4);
            }
            if self.steps > 2 {
                surprise = ((err / self.err_scale - 1.0) / 1.5).clamp(0.0, 1.0);
            }
            // Adapt the typical-error estimate after reading the surprise.
            self.err_scale = self.err_scale * 0.9 + err * 0.1;
        }

        // Recognition: closeness to the nearest remembered state; learn novel ones.
        let mut recognition = 0.0f32;
        for m in &self.memory {
            recognition = recognition.max(cos(m, &pu));
        }
        if recognition < NOVELTY_COS {
            if self.memory.len() >= MEMORY_CAP {
                self.memory.remove(0);
            }
            self.memory.push(pu.clone());
        }

        // Per-patch surprise map: compare last tick's per-patch prediction to now.
        let mut surprise_map = Vec::new();
        if let (Some(pred), true) = (&self.pred_grid, grid.len() == self.pred_grid.as_ref().map_or(0, |g| g.len())) {
            let mut max = 1e-6f32;
            surprise_map = grid.iter().zip(pred).map(|(a, b)| l2(a, b)).collect();
            for &v in &surprise_map {
                max = max.max(v);
            }
            for v in surprise_map.iter_mut() {
                *v = (*v / max).clamp(0.0, 1.0);
            }
        }

        // Predict the next frame, per patch, by constant velocity in latent space
        // (z_next = z + (z - z_prev)); this is what the model expects to see next.
        let mut predicted_next: Option<Grid> = None;
        if let Some(last) = &self.last_grid
            && last.len() == grid.len()
        {
            let next: Grid = grid
                .iter()
                .zip(last)
                .map(|(cur, prev)| cur.iter().zip(prev).map(|(c, p)| 2.0 * c - p).collect())
                .collect();
            predicted_next = Some(next);
        }
        let predicted_heights = predicted_next.as_ref().map(Self::relief).unwrap_or_default();

        // Fit the global predictor on the observed transition and predict next.
        if let Some(prev) = &self.prev_proj {
            self.ar1.observe(prev, &proj);
        }
        self.pred_proj = Some(self.ar1.predict(&proj));
        self.pred_grid = predicted_next;
        let latent = [proj[0], proj[1], proj[2]];
        self.prev_proj = Some(proj);
        self.prev_grid = self.last_grid.take();
        self.last_grid = Some(grid.clone());
        self.last_pooled_unit = Some(pu.clone());

        // Named-state recognition: nearest saved snapshot.
        let mut recognized_label = None;
        let mut recognized_conf = 0.0f32;
        for (label, emb) in &self.snapshots {
            let c = cos(emb, &pu);
            if c > recognized_conf {
                recognized_conf = c;
                recognized_label = Some(label.clone());
            }
        }
        // Only claim a named match when it is clearly the closest and close enough.
        if recognized_conf < 0.82 {
            recognized_label = None;
        }

        let state = if self.steps > 3 && surprise > 0.6 {
            "surprised"
        } else if recognition > 0.7 {
            "recognized"
        } else {
            "learning"
        };

        SceneReport {
            surprise,
            recognition,
            known_states: self.memory.len(),
            state,
            surprise_map,
            predicted_heights,
            steps: self.steps,
            recognized_label,
            recognized_conf,
            snapshots: self.snapshots.iter().map(|(n, _)| n.clone()).collect(),
            // `proj` and `pred_proj` already live in the projected space, so their
            // first three coordinates are exactly `project3` of the same vectors.
            latent,
            predicted_latent: self.pred_proj.as_ref().map(|p| [p[0], p[1], p[2]]).unwrap_or(latent),
            memory_latent: self.memory.iter().map(|m| self.project3(m)).collect(),
            snapshot_latent: self.snapshots.iter().map(|(_, v)| self.project3(v)).collect(),
        }
    }

    /// Save the current view under `label` (replaces one with the same name).
    /// Returns false when no frame has been observed yet.
    pub fn save_snapshot(&mut self, label: &str) -> bool {
        let Some(pu) = self.last_pooled_unit.clone() else { return false };
        self.snapshots.retain(|(n, _)| n != label);
        self.snapshots.push((label.to_string(), pu));
        true
    }

    pub fn delete_snapshot(&mut self, label: &str) -> bool {
        let before = self.snapshots.len();
        self.snapshots.retain(|(n, _)| n != label);
        self.snapshots.len() != before
    }

    pub fn snapshot_names(&self) -> Vec<String> {
        self.snapshots.iter().map(|(n, _)| n.clone()).collect()
    }

    /// Forget the learned dynamics and auto memory, but keep the user's named states.
    pub fn reset(&mut self) {
        let snaps = std::mem::take(&mut self.snapshots);
        let (dim, gw, gh, proj) = (self.dim, self.grid_w, self.grid_h, std::mem::take(&mut self.proj));
        *self = Self::new();
        self.dim = dim;
        self.grid_w = gw;
        self.grid_h = gh;
        self.proj = proj;
        self.snapshots = snaps;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(vals: &[f32], dim: usize) -> Grid {
        vals.chunks(dim).map(|c| c.to_vec()).collect()
    }

    #[test]
    fn latent_display_projection_is_consistent_with_the_memory() {
        let mut p = WorldScenePredictor::new();
        let dim = 4;
        let mut base = vec![0.2f32, 0.1, 0.0, 0.3, 0.4, 0.2, 0.1, 0.0];
        let mut last = SceneReport::default();
        for step in 0..24 {
            for (i, v) in base.iter_mut().enumerate() {
                *v += 0.02 * ((i + step) % 3) as f32;
            }
            let g = grid(&base, dim);
            let pooled: Vec<f32> = (0..dim).map(|k| g.iter().map(|q| q[k]).sum::<f32>()).collect();
            last = p.step(&pooled, &g, 2, 1);
        }
        // One display point per remembered state, and one per named snapshot.
        assert_eq!(last.memory_latent.len(), last.known_states);
        assert_eq!(last.snapshot_latent.len(), last.snapshots.len());
        // The reported point is the first three coordinates of the projection the
        // predictor itself works in, not a separate projection that could drift.
        let pu = unit(&p.last_pooled_unit.clone().unwrap());
        let full = p.project(&pu);
        assert!((full[0] - last.latent[0]).abs() < 1e-5);
        assert!((full[1] - last.latent[1]).abs() < 1e-5);
        assert!((full[2] - last.latent[2]).abs() < 1e-5);
        // Every coordinate is finite, so the view never has to guard against NaN.
        for v in last.latent.iter().chain(last.predicted_latent.iter()) {
            assert!(v.is_finite(), "latent coordinate must be finite");
        }
    }

    #[test]
    fn learns_a_repeating_scene_and_flags_a_jump() {
        let mut p = WorldScenePredictor::new();
        let dim = 4;
        // A slowly drifting scene: two patches translating a little each step.
        let mut base = vec![0.2f32, 0.1, 0.0, 0.3, 0.4, 0.2, 0.1, 0.0];
        let mut last = SceneReport::default();
        for _ in 0..30 {
            for (i, v) in base.iter_mut().enumerate() {
                *v += 0.01 * ((i % 3) as f32 - 1.0);
            }
            let g = grid(&base, dim);
            let pooled: Vec<f32> = (0..dim).map(|k| g.iter().map(|p| p[k]).sum::<f32>()).collect();
            last = p.step(&pooled, &g, 2, 1);
        }
        // A steady drift is well predicted: surprise stays low, states are recognised.
        assert!(last.surprise < 0.6, "steady scene should not surprise: {}", last.surprise);
        assert!(last.known_states >= 1);
        assert_eq!(last.surprise_map.len(), 2);
        assert_eq!(last.predicted_heights.len(), 2);

        // A sudden, large, unpredicted change spikes surprise.
        let jump = vec![5.0f32, -4.0, 3.0, -2.0, -5.0, 4.0, -3.0, 2.0];
        let g = grid(&jump, dim);
        let pooled: Vec<f32> = (0..dim).map(|k| g.iter().map(|p| p[k]).sum::<f32>()).collect();
        let r = p.step(&pooled, &g, 2, 1);
        assert!(r.surprise > last.surprise, "a jump must raise surprise: {} vs {}", r.surprise, last.surprise);
    }

    #[test]
    fn resets_when_the_encoder_changes() {
        let mut p = WorldScenePredictor::new();
        let g4 = grid(&[0.1, 0.2, 0.3, 0.4], 4);
        p.step(&[0.4, 0.6, 0.0, 0.0], &g4, 1, 1);
        assert_eq!(p.dim, 4);
        let g8 = grid(&[0.0; 8], 8);
        let r = p.step(&[0.0; 8], &g8, 1, 1);
        assert_eq!(p.dim, 8);
        assert_eq!(r.steps, 1, "dimension change resets the model");
    }
}
