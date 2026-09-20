//! Latent world model learned from camera observations (JEPA style).
//!
//! The robot never predicts pixels. It records transitions `(z_t, a, z_{t+1})` where
//! `z` is the JEPA embedding of the camera frame and `a` the joint delta it applied,
//! then fits a local linear dynamics model in embedding space:
//!
//! `z_{t+1} = z_t + W [a; 1]`
//!
//! `W` is a ridge regression refit after every transition (an 8 x 8 normal equation
//! per fit, cheap enough for 30 Hz hardware). Planning evaluates many candidate
//! actions inside the model and executes the one whose predicted embedding is closest
//! to the goal; the real observation that follows becomes a new training transition,
//! so the model keeps correcting itself while it acts.

use rand::RngExt;
use serde::{Deserialize, Serialize};

use crate::robot::controller::energy;
use crate::robot::DOF;
use crate::types::normalize_l2;

/// Action = joint deltas plus gripper delta.
pub const ACTION_DIM: usize = DOF + 1;
const FEATURES: usize = ACTION_DIM + 1; // bias term

/// Minimum transitions before the model is trusted for planning.
pub const MIN_TRANSITIONS_TO_PLAN: usize = 12;
/// Upper bound on stored transitions (oldest are dropped).
pub const MAX_TRANSITIONS: usize = 2000;
/// Ridge regularisation.
const LAMBDA: f32 = 1e-2;
/// Locality kernel width in joint space (radians). Transitions taken far from the
/// current pose barely influence the local model, which makes the linear dynamics
/// state dependent.
const LOCALITY_RAD: f32 = 0.6;
/// Minimum weight so that a sparse neighbourhood still yields a usable fit.
const LOCALITY_FLOOR: f32 = 0.05;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Transition {
    pub z: Vec<f32>,
    pub action: [f32; ACTION_DIM],
    pub z_next: Vec<f32>,
    pub joints: [f32; DOF],
    pub timestamp: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LatentWorldModel {
    pub model_name: String,
    pub dim: usize,
    pub transitions: Vec<Transition>,
    /// Row major `dim x FEATURES`.
    #[serde(skip)]
    weights: Vec<f32>,
    #[serde(skip)]
    fitted_on: usize,
    /// Relative error of the last fit on the most recent transitions (0 = perfect).
    #[serde(skip)]
    pub fit_error: Option<f32>,
}

/// Summary for telemetry.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WorldModelStats {
    pub model_name: String,
    pub dim: usize,
    pub transitions: usize,
    pub ready: bool,
    pub fit_error: Option<f32>,
}

/// Solve a small symmetric positive definite system `A x = b` (Gauss with pivoting).
fn solve(mut a: Vec<Vec<f32>>, mut b: Vec<f32>) -> Option<Vec<f32>> {
    let n = b.len();
    for col in 0..n {
        let pivot = (col..n).max_by(|i, j| a[*i][col].abs().partial_cmp(&a[*j][col].abs()).unwrap())?;
        if a[pivot][col].abs() < 1e-9 {
            return None;
        }
        a.swap(col, pivot);
        b.swap(col, pivot);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            let pivot_row = a[col].clone();
            for (k, v) in a[row].iter_mut().enumerate().skip(col) {
                *v -= f * pivot_row[k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut s = b[row];
        for k in row + 1..n {
            s -= a[row][k] * x[k];
        }
        x[row] = s / a[row][row];
    }
    Some(x)
}

fn features(action: &[f32; ACTION_DIM]) -> [f32; FEATURES] {
    let mut f = [1.0f32; FEATURES];
    f[..ACTION_DIM].copy_from_slice(action);
    f
}

impl LatentWorldModel {
    pub fn new(model_name: &str) -> Self {
        Self { model_name: model_name.to_string(), ..Self::default() }
    }

    pub fn is_ready(&self) -> bool {
        self.transitions.len() >= MIN_TRANSITIONS_TO_PLAN && !self.weights.is_empty()
    }

    pub fn stats(&self) -> WorldModelStats {
        WorldModelStats {
            model_name: self.model_name.clone(),
            dim: self.dim,
            transitions: self.transitions.len(),
            ready: self.is_ready(),
            fit_error: self.fit_error,
        }
    }

    /// Record one observed transition (embeddings are L2 normalised on entry).
    pub fn record(&mut self, z: &[f32], action: [f32; ACTION_DIM], z_next: &[f32], joints: [f32; DOF], now: u64) {
        if z.len() != z_next.len() || z.is_empty() {
            return;
        }
        if self.dim != z.len() {
            // Embedding space changed (other model): the old data is meaningless.
            self.transitions.clear();
            self.weights.clear();
            self.dim = z.len();
        }
        self.transitions.push(Transition {
            z: normalize_l2(z),
            action,
            z_next: normalize_l2(z_next),
            joints,
            timestamp: now,
        });
        if self.transitions.len() > MAX_TRANSITIONS {
            let drop = self.transitions.len() - MAX_TRANSITIONS;
            self.transitions.drain(..drop);
        }
        self.fit();
    }

    /// Refit around the most recent pose (called after every recorded transition).
    pub fn fit(&mut self) {
        let around = self.transitions.last().map(|t| t.joints);
        self.fit_around(around);
    }

    /// Weighted ridge regression of `dz = W [a; 1]`: recency times joint-space locality
    /// around `around` (when given), so the model is linear only locally.
    pub fn fit_around(&mut self, around: Option<[f32; DOF]>) {
        let n = self.transitions.len();
        if n < 2 || self.dim == 0 {
            return;
        }
        let weight = |idx: usize, t: &Transition| -> f32 {
            let recency = 0.5 + 0.5 * (idx as f32 + 1.0) / n as f32;
            let locality = match around {
                Some(j) => {
                    let d2: f32 = j.iter().zip(t.joints.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (-d2 / (LOCALITY_RAD * LOCALITY_RAD)).exp().max(LOCALITY_FLOOR)
                }
                None => 1.0,
            };
            recency * locality
        };
        // Normal matrix (FEATURES x FEATURES) shared by every output dimension.
        let mut ata = vec![vec![0.0f32; FEATURES]; FEATURES];
        for (idx, t) in self.transitions.iter().enumerate() {
            let w = weight(idx, t);
            let f = features(&t.action);
            for i in 0..FEATURES {
                for j in 0..FEATURES {
                    ata[i][j] += w * f[i] * f[j];
                }
            }
        }
        for (i, row) in ata.iter_mut().enumerate() {
            row[i] += LAMBDA;
        }
        // Right hand sides for every embedding dimension.
        let mut weights = vec![0.0f32; self.dim * FEATURES];
        let mut atb = vec![vec![0.0f32; FEATURES]; self.dim];
        for (idx, t) in self.transitions.iter().enumerate() {
            let w = weight(idx, t);
            let f = features(&t.action);
            for (d, row) in atb.iter_mut().enumerate() {
                let dz = t.z_next[d] - t.z[d];
                for i in 0..FEATURES {
                    row[i] += w * f[i] * dz;
                }
            }
        }
        for d in 0..self.dim {
            match solve(ata.clone(), atb[d].clone()) {
                Some(x) => weights[d * FEATURES..(d + 1) * FEATURES].copy_from_slice(&x),
                None => return,
            }
        }
        self.weights = weights;
        self.fitted_on = n;
        self.fit_error = self.holdout_error();
    }

    /// Relative prediction error on the most recent quarter of the transitions.
    fn holdout_error(&self) -> Option<f32> {
        let n = self.transitions.len();
        if n < 4 || self.weights.is_empty() {
            return None;
        }
        let recent = &self.transitions[n - (n / 4).max(1)..];
        let (mut err, mut base) = (0.0f32, 0.0f32);
        for t in recent {
            let pred = self.predict(&t.z, &t.action);
            err += pred.iter().zip(t.z_next.iter()).map(|(p, a)| (p - a) * (p - a)).sum::<f32>();
            base += t.z.iter().zip(t.z_next.iter()).map(|(p, a)| (p - a) * (p - a)).sum::<f32>();
        }
        Some(if base > 1e-9 { (err / base).min(9.99) } else { 0.0 })
    }

    /// Predicted next embedding for an action taken at `z`.
    pub fn predict(&self, z: &[f32], action: &[f32; ACTION_DIM]) -> Vec<f32> {
        if self.weights.is_empty() || z.len() != self.dim {
            return z.to_vec();
        }
        let f = features(action);
        let mut out = z.to_vec();
        for (d, o) in out.iter_mut().enumerate() {
            let row = &self.weights[d * FEATURES..(d + 1) * FEATURES];
            *o += row.iter().zip(f.iter()).map(|(w, x)| w * x).sum::<f32>();
        }
        normalize_l2(&out)
    }

    /// Remembered pose whose observed embedding is closest to `goal`, with that energy.
    /// Memory acts as a nonparametric world model: robust to noise, coarse in space.
    pub fn best_remembered_pose(&self, goal: &[f32]) -> Option<([f32; DOF], f32)> {
        self.transitions
            .iter()
            .map(|t| {
                // Pose after the action is where z_next was observed.
                let mut pose = t.joints;
                for (p, a) in pose.iter_mut().zip(t.action.iter()) {
                    *p += a;
                }
                (pose, energy(&t.z_next, goal))
            })
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Model predictive step: evaluate `samples` random actions of scale `step`, plus
    /// the previous action, its reverse and scaled versions (momentum), and keep the one
    /// whose predicted embedding is closest to `goal`. Returns `(action, predicted energy)`.
    pub fn plan(
        &self,
        z: &[f32],
        goal: &[f32],
        step: f32,
        samples: usize,
        previous: Option<[f32; ACTION_DIM]>,
        rng: &mut impl RngExt,
    ) -> ([f32; ACTION_DIM], f32) {
        let mut best: ([f32; ACTION_DIM], f32) = ([0.0; ACTION_DIM], energy(z, goal));
        let consider = |a: [f32; ACTION_DIM], best: &mut ([f32; ACTION_DIM], f32)| {
            let e = energy(&self.predict(z, &a), goal);
            if e < best.1 {
                *best = (a, e);
            }
        };
        if let Some(p) = previous {
            for k in [1.0f32, 0.5, 1.5, -1.0, -0.5] {
                let mut a = p;
                a.iter_mut().for_each(|v| *v *= k);
                consider(a, &mut best);
            }
        }
        for _ in 0..samples {
            let mut a = [0.0f32; ACTION_DIM];
            for v in a.iter_mut() {
                *v = rng.random_range(-step..=step);
            }
            consider(a, &mut best);
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Synthetic environment: the embedding is a fixed linear image of the joints.
    fn observe(joints: &[f32; DOF]) -> Vec<f32> {
        let mut z = vec![0.0f32; 16];
        for (i, v) in z.iter_mut().enumerate() {
            let j = i % DOF;
            *v = 0.3 + (joints[j] * (1.0 + i as f32 * 0.1)).sin() * 0.5;
        }
        normalize_l2(&z)
    }

    #[test]
    fn learns_dynamics_from_random_babbling_and_plans_toward_goal() {
        let mut wm = LatentWorldModel::new("test");
        let mut rng = rand::rng();
        let mut joints = [0.0f32; DOF];
        for t in 0..60 {
            let mut a = [0.0f32; ACTION_DIM];
            for v in a.iter_mut().take(DOF) {
                *v = rng.random_range(-0.1..=0.1);
            }
            let z = observe(&joints);
            let mut next = joints;
            for i in 0..DOF {
                next[i] += a[i];
            }
            wm.record(&z, a, &observe(&next), joints, t);
            joints = next;
        }
        assert!(wm.is_ready());
        assert!(wm.fit_error.unwrap() < 0.6, "fit error {:?}", wm.fit_error);

        // Plan toward a goal pose and check the chosen action reduces the true energy.
        let goal_pose = [0.3, -0.2, 0.25, 0.1, -0.15, 0.2];
        let goal = observe(&goal_pose);
        let start = [0.0f32; DOF];
        let z = observe(&start);
        let before = energy(&z, &goal);
        let (a, predicted) = wm.plan(&z, &goal, 0.1, 128, None, &mut rng);
        let mut after_pose = start;
        for i in 0..DOF {
            after_pose[i] += a[i];
        }
        let after = energy(&observe(&after_pose), &goal);
        assert!(predicted < before, "predicted {predicted} before {before}");
        assert!(after < before, "true energy after {after} before {before}");
    }

    #[test]
    fn dimension_change_resets_memory() {
        let mut wm = LatentWorldModel::new("m");
        wm.record(&[1.0, 0.0], [0.0; ACTION_DIM], &[0.0, 1.0], [0.0; DOF], 1);
        wm.record(&[1.0, 0.0], [0.0; ACTION_DIM], &[0.0, 1.0], [0.0; DOF], 2);
        assert_eq!(wm.transitions.len(), 2);
        wm.record(&[1.0, 0.0, 0.0], [0.0; ACTION_DIM], &[0.0, 1.0, 0.0], [0.0; DOF], 3);
        assert_eq!(wm.transitions.len(), 1);
        assert_eq!(wm.dim, 3);
    }
}
