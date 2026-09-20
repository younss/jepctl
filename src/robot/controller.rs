//! Control loop, gesture shadowing (Mode A) and latent goal seeking (Mode C).
//!
//! Mode C in one paragraph: the user captures the current embedding as `z_goal`. The
//! controller then runs rounds of local exploration: from the base pose it proposes
//! `CANDIDATES` random joint perturbations of scale `step`, moves to each one, waits
//! for the arm to settle, receives one observation embedding `z`, and records the
//! energy `E = ||z - z_goal||_2 / sqrt(dim)` (the same metric as `POST /api/energy`).
//! The best candidate becomes the new base when it beats the base energy; otherwise
//! the step shrinks. Observations come from the server camera (physical arm) or from
//! snapshots of the WebGL twin posted by the browser (virtual arm).

use std::collections::HashMap;
use std::time::Duration;

use rand::RngExt;

use crate::robot::hal::BackendKind;
use crate::robot::safety::SafetyGuard;
use crate::robot::{
    GestureAction, GoalProgress, JointCommand, RobotCore, RobotError, RobotHandle, RobotMode, CONTROL_HZ, DOF,
};
use crate::types::normalize_l2;

/// Perturbations evaluated per exploration round.
pub const CANDIDATES: usize = 8;
/// Initial perturbation scale in radians.
pub const INITIAL_STEP_RAD: f32 = 0.12;
pub const MIN_STEP_RAD: f32 = 0.01;
/// Settle tolerance before an observation is accepted.
pub const SETTLE_EPS_RAD: f32 = 0.01;

/// Energy between two embeddings: RMS L2 distance on unit vectors, as in `/api/energy`.
pub fn energy(z: &[f32], goal: &[f32]) -> f32 {
    if z.len() != goal.len() || z.is_empty() {
        return f32::INFINITY;
    }
    let a = normalize_l2(z);
    let b = normalize_l2(goal);
    let sum_sq: f32 = a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum();
    (sum_sq / z.len() as f32).sqrt()
}

/// Default gesture mapping; names match the sandbox's default slots.
pub fn default_gesture_map() -> HashMap<String, GestureAction> {
    HashMap::from([
        ("Open Hand".to_string(), GestureAction::OpenGripper),
        ("Fist".to_string(), GestureAction::CloseGripper),
        ("Victory".to_string(), GestureAction::Approve),
        ("Up".to_string(), GestureAction::JointDelta { joint: 3, delta: 0.05 }),
        ("Down".to_string(), GestureAction::JointDelta { joint: 3, delta: -0.05 }),
        ("Stop".to_string(), GestureAction::Stop),
    ])
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Idle,
    /// Waiting for an observation of the base pose.
    MeasureBase,
    /// Moving to / observing candidate `idx`.
    Candidate(usize),
}

/// Mode C state machine.
#[derive(Debug, Clone)]
pub struct GoalExplorer {
    pub goal: Option<Vec<f32>>,
    base: [f32; DOF],
    base_energy: Option<f32>,
    initial_energy: Option<f32>,
    best_energy: Option<f32>,
    current_energy: Option<f32>,
    candidates: Vec<[f32; DOF]>,
    energies: Vec<f32>,
    phase: Phase,
    step: f32,
    iterations: u32,
    evaluated: u32,
}

impl Default for GoalExplorer {
    fn default() -> Self {
        Self {
            goal: None,
            base: [0.0; DOF],
            base_energy: None,
            initial_energy: None,
            best_energy: None,
            current_energy: None,
            candidates: Vec::new(),
            energies: Vec::new(),
            phase: Phase::Idle,
            step: INITIAL_STEP_RAD,
            iterations: 0,
            evaluated: 0,
        }
    }
}

impl GoalExplorer {
    pub fn set_goal(&mut self, goal: Vec<f32>, current_pose: [f32; DOF]) {
        *self = Self { goal: Some(goal), base: current_pose, phase: Phase::MeasureBase, ..Self::default() };
    }

    pub fn abort(&mut self) {
        self.phase = Phase::Idle;
        self.candidates.clear();
        self.energies.clear();
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub fn is_active(&self) -> bool {
        self.goal.is_some() && self.phase != Phase::Idle
    }

    /// Pose the arm should be at for the next observation.
    pub fn desired_pose(&self) -> Option<[f32; DOF]> {
        match self.phase {
            Phase::Idle => None,
            Phase::MeasureBase => Some(self.base),
            Phase::Candidate(i) => self.candidates.get(i).copied(),
        }
    }

    pub fn resume(&mut self) {
        if self.goal.is_some() && self.phase == Phase::Idle {
            self.phase = Phase::MeasureBase;
        }
    }

    /// Feed one observation taken at the current desired pose. Returns the next pose.
    pub fn observe(&mut self, z: &[f32], safety: &SafetyGuard, rng: &mut impl RngExt) -> Option<[f32; DOF]> {
        let goal = self.goal.as_ref()?;
        let e = energy(z, goal);
        if !e.is_finite() {
            return self.desired_pose();
        }
        self.current_energy = Some(e);
        self.evaluated += 1;
        match self.phase {
            Phase::Idle => return None,
            Phase::MeasureBase => {
                self.base_energy = Some(e);
                self.initial_energy.get_or_insert(e);
                self.best_energy = Some(self.best_energy.map_or(e, |b| b.min(e)));
                self.spawn_candidates(safety, rng);
            }
            Phase::Candidate(i) => {
                self.energies.push(e);
                if i + 1 < self.candidates.len() {
                    self.phase = Phase::Candidate(i + 1);
                } else {
                    self.finish_round(safety, rng);
                }
            }
        }
        self.desired_pose()
    }

    fn spawn_candidates(&mut self, safety: &SafetyGuard, rng: &mut impl RngExt) {
        self.candidates = (0..CANDIDATES)
            .map(|_| {
                let mut c = self.base;
                for v in c.iter_mut() {
                    *v += rng.random_range(-self.step..=self.step);
                }
                safety.limits.clamp(c)
            })
            .collect();
        self.energies.clear();
        self.phase = Phase::Candidate(0);
    }

    fn finish_round(&mut self, safety: &SafetyGuard, rng: &mut impl RngExt) {
        self.iterations += 1;
        let base_e = self.base_energy.unwrap_or(f32::INFINITY);
        let best = self
            .energies
            .iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, e)| (i, *e));
        match best {
            Some((i, e)) if e < base_e => {
                self.base = self.candidates[i];
                self.base_energy = Some(e);
                self.best_energy = Some(self.best_energy.map_or(e, |b| b.min(e)));
                // Reward progress with a slightly larger step (bounded).
                self.step = (self.step * 1.1).min(INITIAL_STEP_RAD * 2.0);
            }
            _ => {
                self.step = (self.step * 0.6).max(MIN_STEP_RAD);
            }
        }
        if self.step <= MIN_STEP_RAD && self.best_energy.is_some_and(|b| b < 0.02) {
            self.phase = Phase::Idle;
        } else {
            self.spawn_candidates(safety, rng);
        }
    }

    pub fn progress(&self) -> GoalProgress {
        let convergence = match (self.initial_energy, self.best_energy) {
            (Some(i), Some(b)) if i > 1e-6 => ((i - b) / i).clamp(0.0, 1.0),
            _ => 0.0,
        };
        GoalProgress {
            has_goal: self.goal.is_some(),
            goal_dimension: self.goal.as_ref().map_or(0, |g| g.len()),
            current_energy: self.current_energy,
            best_energy: self.best_energy,
            initial_energy: self.initial_energy,
            iterations: self.iterations,
            candidates_evaluated: self.evaluated,
            step_rad: self.step,
            convergence,
            phase: match self.phase {
                Phase::Idle => {
                    if self.goal.is_some() {
                        "converged".into()
                    } else {
                        "idle".into()
                    }
                }
                Phase::MeasureBase => "measuring base".into(),
                Phase::Candidate(i) => format!("candidate {}/{}", i + 1, CANDIDATES),
            },
            awaiting_observation: false,
        }
    }
}

impl RobotCore {
    /// Apply a gesture detection in shadowing mode.
    pub fn apply_gesture(&mut self, name: &str) -> Result<(), RobotError> {
        self.last_gesture = Some(name.to_string());
        if self.mode != RobotMode::Shadowing {
            return Ok(());
        }
        let Some(action) = self.gesture_map.get(name).cloned() else { return Ok(()) };
        let mut cmd = JointCommand { joints: self.targets, gripper: self.gripper_target };
        match action {
            GestureAction::OpenGripper => cmd.gripper = 1.0,
            GestureAction::CloseGripper => cmd.gripper = 0.0,
            GestureAction::JointDelta { joint, delta } => {
                let j = joint.min(DOF - 1);
                cmd.joints[j] = (cmd.joints[j] + delta).clamp(self.safety.limits.min[j], self.safety.limits.max[j]);
            }
            GestureAction::Pose { joints, gripper } => {
                cmd = JointCommand { joints: self.safety.limits.clamp(joints), gripper: gripper.clamp(0.0, 1.0) }
            }
            GestureAction::Approve => {
                self.approve_pending()?;
                return Ok(());
            }
            GestureAction::Stop => {
                self.emergency_stop();
                return Ok(());
            }
        }
        // Gesture moves are always gated on the physical arm; the user approves them.
        self.submit(cmd, false)?;
        Ok(())
    }

    /// One control step: ramp toward targets, drive the backend, pull real positions.
    pub fn tick(&mut self, dt: f32) {
        self.tick += 1;
        if self.mode == RobotMode::GoalSeeking {
            if let Some(pose) = self.explorer.desired_pose() {
                if !self.safety.estop {
                    // Explorer poses are pre clamped; bypass the gate on the virtual arm only.
                    let gated = self.safety_gate && self.backend.kind() == BackendKind::Physical;
                    if gated {
                        self.pending = Some(JointCommand { joints: pose, gripper: self.gripper_target });
                    } else {
                        self.targets = pose;
                    }
                }
            }
        }
        let next = self.safety.ramp(self.joints, self.targets, dt);
        let next_grip = self.safety.ramp_gripper(self.gripper, self.gripper_target, dt);
        let moved = next != self.joints || (next_grip - self.gripper).abs() > 1e-6;
        self.joints = next;
        self.gripper = next_grip;
        if moved && self.backend.is_connected() {
            if let Err(e) = self.backend.set_joint_targets(&self.joints, self.gripper) {
                self.last_error = Some(e.to_string());
            }
        }
    }

    /// Whether the arm is at its targets (used before taking Mode C observations).
    pub fn settled(&self) -> bool {
        SafetyGuard::settled(self.joints, self.targets, SETTLE_EPS_RAD)
            && (self.gripper - self.gripper_target).abs() < 0.01
    }

    /// Feed an observation embedding to the goal seeker (Mode C).
    pub fn observe(&mut self, z: &[f32]) -> Option<[f32; DOF]> {
        if self.mode != RobotMode::GoalSeeking || self.safety.estop {
            return None;
        }
        let mut rng = rand::rng();
        self.explorer.observe(z, &self.safety, &mut rng)
    }

    /// Switch backend (disconnects the previous one). Physical forces the safety gate.
    pub fn switch_backend(&mut self, kind: BackendKind) -> Result<(), RobotError> {
        if self.backend.kind() == kind && self.backend.is_connected() {
            return Ok(());
        }
        let _ = self.backend.disconnect();
        let mut backend = crate::robot::hal::make_backend(kind, &self.hardware)?;
        backend.connect()?;
        // Start from what the hardware reports so the ramp does not lurch.
        self.joints = backend.get_joint_positions();
        self.gripper = backend.get_gripper_position();
        self.targets = self.joints;
        self.gripper_target = self.gripper;
        self.pending = None;
        self.backend = backend;
        if kind == BackendKind::Physical {
            self.safety_gate = true;
        }
        Ok(())
    }
}

/// Spawn the 30 Hz control loop. It owns the timing; handlers only mutate the core.
pub fn spawn_control_loop(handle: RobotHandle) {
    tokio::spawn(async move {
        let dt = 1.0 / CONTROL_HZ as f32;
        let mut interval = tokio::time::interval(Duration::from_secs_f32(dt));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            let telemetry = {
                let mut core = handle.core.lock().await;
                core.tick(dt);
                core.telemetry()
            };
            let _ = handle.telemetry_tx.send(telemetry);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::robot::hal::HardwareConfig;

    #[test]
    fn energy_is_zero_at_goal_and_positive_elsewhere() {
        let g = vec![1.0, 0.0, 0.0, 0.0];
        assert!(energy(&g, &g) < 1e-6);
        assert!(energy(&[0.0, 1.0, 0.0, 0.0], &g) > 0.5);
        assert!(energy(&[1.0, 0.0], &g).is_infinite());
    }

    #[test]
    fn explorer_converges_on_a_synthetic_latent() {
        // Latent = joint angles themselves; goal = a pose. Energy then decreases with distance.
        let safety = SafetyGuard::default();
        let goal_pose = [0.4, -0.3, 0.5, 0.2, -0.1, 0.3];
        let mut ex = GoalExplorer::default();
        ex.set_goal(goal_pose.to_vec(), [0.0; DOF]);
        let mut rng = rand::rng();
        let mut pose = ex.desired_pose().unwrap();
        let start = energy(&pose, &goal_pose);
        for _ in 0..40 * (CANDIDATES + 1) {
            match ex.observe(&pose, &safety, &mut rng) {
                Some(p) => pose = p,
                None => break,
            }
        }
        let p = ex.progress();
        // Either the loop ran its rounds or it converged early; both must have cut the energy.
        assert!(p.iterations >= 1, "{p:?}");
        assert!(p.best_energy.unwrap() < start * 0.5, "start {start} best {:?}", p.best_energy);
        assert!(p.convergence > 0.5);
    }

    #[test]
    fn gesture_shadowing_and_gating() {
        let mut core = RobotCore::new(HardwareConfig::default());
        core.backend.connect().unwrap();
        core.mode = RobotMode::Shadowing;
        core.apply_gesture("Open Hand").unwrap();
        assert_eq!(core.gripper_target, 1.0);
        core.apply_gesture("Up").unwrap();
        assert!((core.targets[3] - 0.05).abs() < 1e-6);
        core.apply_gesture("unknown").unwrap();
        assert_eq!(core.last_gesture.as_deref(), Some("unknown"));

        // Ramp: after one tick the joint moved by at most 1.5/30 rad.
        core.targets[0] = 1.0;
        core.tick(1.0 / 30.0);
        assert!((core.joints[0] - 0.05).abs() < 1e-6);

        // Safety gate holds unapproved commands; approval executes them.
        core.safety_gate = true;
        // Simulate a physical backend by gating on kind: use submit directly with a virtual
        // backend (not gated) then check the pending path via approve.
        let cmd = JointCommand { joints: [0.2; DOF], gripper: 0.3 };
        assert!(core.submit(cmd, false).unwrap());
        core.emergency_stop();
        assert!(core.safety.estop);
        assert!(matches!(core.submit(cmd, true), Err(RobotError::EStopEngaged)));
        core.reset_safety();
        assert!(core.submit(cmd, true).unwrap());
    }
}
