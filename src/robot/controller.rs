//! Control loop, gesture shadowing (Mode A) and the latent agent (Exploring / Mode C).
//!
//! The agent learns from what the camera sees, in JEPA fashion:
//! 1. observe: the camera frame is embedded to `z_t` (only once the arm has settled);
//! 2. act: an action `a` (joint deltas) is executed under the safety ramp;
//! 3. observe again: `z_{t+1}` and the transition `(z_t, a, z_{t+1})` train the latent
//!    world model (`world_model.rs`), a predictor in embedding space;
//! 4. plan: with a goal `z_goal`, candidate actions are evaluated inside the model and
//!    the best one is executed; without a goal (Exploring) actions are random babbling.
//!
//! Energy `E = ||z - z_goal||_2 / sqrt(dim)` is the same metric as `POST /api/energy`.

use std::collections::HashMap;
use std::time::Duration;

use rand::RngExt;

use crate::robot::hal::BackendKind;
use crate::robot::safety::SafetyGuard;
use crate::robot::world_model::{ACTION_DIM, LatentWorldModel};
use crate::robot::{
    CONTROL_HZ, DOF, GestureAction, GoalProgress, JointCommand, RobotCore, RobotError, RobotHandle, RobotMode,
};
use crate::types::normalize_l2;

/// Candidate actions evaluated inside the world model per planning step.
pub const PLAN_SAMPLES: usize = 96;
/// Initial action scale in radians.
pub const INITIAL_STEP_RAD: f32 = 0.10;
pub const MIN_STEP_RAD: f32 = 0.02;
pub const MAX_STEP_RAD: f32 = 0.25;
/// Absolute energy under which the goal always counts as reached.
pub const GOAL_REACHED_ENERGY: f32 = 0.004;
/// The goal also counts as reached below this multiple of the camera noise floor
/// (energy between two frames of the goal scene at rest): poses that give the same
/// view as the goal are, by definition, the goal.
pub const GOAL_NOISE_FACTOR: f32 = 2.5;
/// Lower bound on the reach threshold so that a near silent camera does not make the
/// goal unreachable.
pub const GOAL_MIN_THRESHOLD: f32 = 0.008;
/// Steps without improving the best energy after which the search is declared a
/// plateau (the view is as close as this model and camera can get).
pub const PLATEAU_STEPS: u32 = 60;
/// Settle tolerance before an observation is accepted.
pub const SETTLE_EPS_RAD: f32 = 0.01;
/// Largest per joint move toward a remembered pose in one step.
pub const MEMORY_JUMP_RAD: f32 = 0.35;

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
    /// Settled; the next camera embedding is the observation of the current pose.
    AwaitObservation,
    /// An action is being executed under the ramp.
    Moving,
    Converged,
    /// No improvement for `PLATEAU_STEPS`: as close as the model can get.
    Plateau,
}

/// Learning and planning agent (Exploring and Mode C).
#[derive(Debug, Clone)]
pub struct LatentAgent {
    pub world: LatentWorldModel,
    pub goal: Option<Vec<f32>>,
    phase: Phase,
    last_z: Option<Vec<f32>>,
    last_action: Option<[f32; ACTION_DIM]>,
    last_joints: [f32; DOF],
    step: f32,
    current_energy: Option<f32>,
    predicted_energy: Option<f32>,
    best_energy: Option<f32>,
    initial_energy: Option<f32>,
    steps: u32,
    planned: bool,
    /// Camera noise floor measured at capture (energy between two frames at rest).
    noise_floor: Option<f32>,
    /// Last executed action, offered to the planner as momentum.
    previous_action: Option<[f32; ACTION_DIM]>,
    /// Steps since `best_energy` last improved.
    stalled_steps: u32,
    /// Control ticks spent in `Converged`; the view is re-checked periodically so the
    /// arm reacts when the scene or its pose changes afterwards.
    converged_ticks: u32,
}

impl Default for LatentAgent {
    fn default() -> Self {
        Self {
            world: LatentWorldModel::default(),
            goal: None,
            phase: Phase::Idle,
            last_z: None,
            last_action: None,
            last_joints: [0.0; DOF],
            step: INITIAL_STEP_RAD,
            current_energy: None,
            predicted_energy: None,
            best_energy: None,
            initial_energy: None,
            steps: 0,
            planned: false,
            noise_floor: None,
            previous_action: None,
            stalled_steps: 0,
            converged_ticks: 0,
        }
    }
}

impl LatentAgent {
    /// `noise_floor` is the energy between two consecutive observations of the goal
    /// scene at rest (camera noise); the goal counts as reached below 1.3 times it.
    pub fn set_goal(&mut self, goal: Vec<f32>, noise_floor: Option<f32>) {
        self.goal = Some(goal);
        self.noise_floor = noise_floor.filter(|f| f.is_finite());
        self.current_energy = None;
        self.predicted_energy = None;
        self.best_energy = None;
        self.initial_energy = None;
        self.steps = 0;
        self.stalled_steps = 0;
        self.step = INITIAL_STEP_RAD;
        self.phase = Phase::AwaitObservation;
    }

    pub fn clear_goal(&mut self) {
        self.goal = None;
        self.current_energy = None;
        self.predicted_energy = None;
        self.best_energy = None;
        self.initial_energy = None;
        if matches!(self.phase, Phase::Converged | Phase::Plateau) {
            self.phase = Phase::AwaitObservation;
        }
    }

    /// Begin acting (called when a learning mode is selected).
    pub fn start(&mut self) {
        if matches!(self.phase, Phase::Idle | Phase::Converged | Phase::Plateau) {
            self.phase = Phase::AwaitObservation;
            self.last_z = None;
            self.last_action = None;
        }
    }

    /// Pause acting; the world model is kept.
    pub fn stop(&mut self) {
        self.phase = Phase::Idle;
        self.last_action = None;
    }

    pub fn awaiting(&self) -> bool {
        self.phase == Phase::AwaitObservation
    }

    /// The controller reports that the arm is settled. After an action this asks for
    /// the next observation; once converged it re-checks the view about once a second.
    pub fn arrived(&mut self) {
        match self.phase {
            Phase::Moving => self.phase = Phase::AwaitObservation,
            Phase::Converged | Phase::Plateau => {
                self.converged_ticks += 1;
                if self.converged_ticks >= CONTROL_HZ {
                    self.converged_ticks = 0;
                    self.last_action = None;
                    self.last_z = None;
                    self.phase = Phase::AwaitObservation;
                }
            }
            _ => {}
        }
    }

    /// Feed the camera observation of the current (settled) pose and choose the next
    /// action. Returns joint targets to execute, or `None` when idle or converged.
    pub fn observe(
        &mut self,
        z: &[f32],
        joints: [f32; DOF],
        gripper: f32,
        safety: &SafetyGuard,
        now: u64,
        rng: &mut impl RngExt,
    ) -> Option<JointCommand> {
        if self.phase != Phase::AwaitObservation {
            return None;
        }
        // Learn from the transition that just completed.
        if let (Some(prev_z), Some(action)) = (self.last_z.take(), self.last_action.take()) {
            self.world.record(&prev_z, action, z, self.last_joints, now);
        }
        // Measure progress toward the goal.
        if let Some(goal) = &self.goal {
            let e = energy(z, goal);
            self.current_energy = Some(e);
            self.initial_energy.get_or_insert(e);
            let improved = self.best_energy.is_none_or(|b| e < b - 0.0005);
            self.best_energy = Some(self.best_energy.map_or(e, |b| b.min(e)));
            self.stalled_steps = if improved { 0 } else { self.stalled_steps + 1 };
            if e < self.reach_threshold() {
                self.phase = Phase::Converged;
                return None;
            }
            if self.stalled_steps >= PLATEAU_STEPS {
                self.stalled_steps = 0;
                self.phase = Phase::Plateau;
                return None;
            }
            // Adapt the step: shrink when the model predicted better than reality delivered.
            if let Some(p) = self.predicted_energy {
                let floor = self.noise_floor.unwrap_or(0.005);
                if e > p + 2.0 * floor {
                    self.step = (self.step * 0.7).max(MIN_STEP_RAD);
                } else {
                    self.step = (self.step * 1.15).min(MAX_STEP_RAD);
                }
            }
        }
        // Choose the next action.
        let mut action = [0.0f32; ACTION_DIM];
        self.planned = false;
        match (&self.goal, self.world.is_ready()) {
            (Some(goal), true) => {
                // Refit the local model around the current pose before planning from it.
                self.world.fit_around(Some(joints));
                let (mut a, mut predicted) =
                    self.world.plan(z, goal, self.step, PLAN_SAMPLES, self.previous_action, rng);
                // Memory candidate: head toward the remembered pose closest to the goal
                // when it is clearly better than the current view.
                let current_e = self.current_energy.unwrap_or(f32::INFINITY);
                if let Some((pose, remembered_e)) = self.world.best_remembered_pose(goal) {
                    // Trust memory over the extrapolating linear model: a pose whose observed
                    // energy was clearly lower than now is a safe place to head to.
                    if remembered_e < current_e * 0.85 {
                        let mut toward = [0.0f32; ACTION_DIM];
                        let mut far = false;
                        for i in 0..DOF {
                            let d = pose[i] - joints[i];
                            toward[i] = d.clamp(-MEMORY_JUMP_RAD, MEMORY_JUMP_RAD);
                            far |= d.abs() > SETTLE_EPS_RAD * 2.0;
                        }
                        if far {
                            a = toward;
                            predicted = remembered_e;
                        }
                    }
                }
                action = a;
                // Exploration noise keeps the dataset informative.
                for v in action.iter_mut() {
                    *v += rng.random_range(-self.step * 0.2..=self.step * 0.2);
                }
                self.predicted_energy = Some(predicted);
                self.planned = true;
            }
            _ => {
                for v in action.iter_mut() {
                    *v = rng.random_range(-self.step..=self.step);
                }
                self.predicted_energy = None;
            }
        }
        // Clamp into the envelope and store the delta that will really be applied.
        let mut targets = joints;
        for i in 0..DOF {
            targets[i] = (joints[i] + action[i]).clamp(safety.limits.min[i], safety.limits.max[i]);
            action[i] = targets[i] - joints[i];
        }
        let grip = (gripper + action[DOF]).clamp(0.0, 1.0);
        action[DOF] = grip - gripper;

        self.last_z = Some(z.to_vec());
        self.last_action = Some(action);
        self.previous_action = Some(action);
        self.last_joints = joints;
        self.steps += 1;
        self.phase = Phase::Moving;
        Some(JointCommand { joints: targets, gripper: grip })
    }

    /// Energy under which the goal view counts as reached.
    fn reach_threshold(&self) -> f32 {
        match self.noise_floor {
            Some(f) => (f * GOAL_NOISE_FACTOR).max(GOAL_MIN_THRESHOLD),
            None => GOAL_REACHED_ENERGY.max(GOAL_MIN_THRESHOLD),
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
            predicted_energy: self.predicted_energy,
            best_energy: self.best_energy,
            initial_energy: self.initial_energy,
            steps: self.steps,
            step_rad: self.step,
            convergence,
            phase: match self.phase {
                Phase::Idle => "idle",
                Phase::AwaitObservation => "awaiting observation",
                Phase::Moving => "moving",
                Phase::Converged => "converged",
                Phase::Plateau => "plateau",
            }
            .to_string(),
            reached_below: self.reach_threshold(),
            policy: if self.planned { "planned".into() } else { "random".into() },
            last_action: self.last_action,
            world: self.world.stats(),
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

    /// Mirror mode: blend the poses of every mapped gesture by how much the person
    /// looks like each one right now. Called with the full score list of a frame.
    pub fn apply_mirror(&mut self, scores: &[crate::gestures::GestureScore]) -> Result<(), RobotError> {
        if self.mode != RobotMode::Mirror {
            return Ok(());
        }
        let candidates: Vec<(String, f32, [f32; DOF], f32)> = scores
            .iter()
            .filter_map(|s| match self.gesture_map.get(&s.name) {
                Some(GestureAction::Pose { joints, gripper }) => Some((s.name.clone(), s.combined, *joints, *gripper)),
                _ => None,
            })
            .collect();
        let (weights, blend) =
            crate::companion::blend_weights(&candidates.iter().map(|c| (c.0.clone(), c.1)).collect::<Vec<_>>());
        self.mirror = weights;
        let Some(weights) = blend else { return Ok(()) };
        let mut joints = [0.0f32; DOF];
        let mut gripper = 0.0f32;
        for (i, w) in weights.iter().enumerate() {
            for (j, v) in joints.iter_mut().enumerate() {
                *v += candidates[i].2[j] * w;
            }
            gripper += candidates[i].3 * w;
        }
        let cmd = JointCommand { joints: self.safety.limits.clamp(joints), gripper: gripper.clamp(0.0, 1.0) };
        self.submit(cmd, false)?;
        Ok(())
    }

    /// One control step: ramp toward targets, drive the backend, pull real positions.
    pub fn tick(&mut self, dt: f32) {
        self.tick += 1;
        let next = self.safety.ramp(self.joints, self.targets, dt);
        let next_grip = self.safety.ramp_gripper(self.gripper, self.gripper_target, dt);
        let moved = next != self.joints || (next_grip - self.gripper).abs() > 1e-6;
        self.joints = next;
        self.gripper = next_grip;
        if moved
            && self.backend.is_connected()
            && let Err(e) = self.backend.set_joint_targets(&self.joints, self.gripper)
        {
            self.last_error = Some(e.to_string());
        }
        if self.learning_mode() && !moved && self.settled() && self.pending.is_none() {
            self.agent.arrived();
        }
    }

    /// Whether the arm is at its targets (used before taking Mode C observations).
    pub fn settled(&self) -> bool {
        SafetyGuard::settled(self.joints, self.targets, SETTLE_EPS_RAD)
            && (self.gripper - self.gripper_target).abs() < 0.01
    }

    /// Feed a camera observation of the current pose to the agent (Exploring / Mode C).
    /// The chosen action is executed (or gated on the physical arm).
    pub fn observe(&mut self, z: &[f32], now: u64) -> Option<JointCommand> {
        if !self.learning_mode() || self.safety.estop || !self.settled() {
            return None;
        }
        let mut rng = rand::rng();
        let (joints, gripper) = (self.joints, self.gripper);
        let cmd = self.agent.observe(z, joints, gripper, &self.safety, now, &mut rng)?;
        let gated = self.safety_gate && self.backend.kind() == BackendKind::Physical;
        if gated {
            self.pending = Some(cmd);
        } else {
            self.targets = cmd.joints;
            self.gripper_target = cmd.gripper;
        }
        Some(cmd)
    }

    /// Switch backend (disconnects the previous one). Physical forces the safety gate.
    pub fn switch_backend(&mut self, kind: BackendKind) -> Result<(), RobotError> {
        if self.backend.kind() == kind && self.backend.is_connected() {
            return Ok(());
        }
        // Build and connect the new backend first: a failure leaves the current one untouched.
        let mut backend = crate::robot::hal::make_backend(kind, &self.hardware)?;
        backend.connect()?;
        let _ = self.backend.disconnect();
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
    fn agent_learns_then_plans_toward_a_goal() {
        // Synthetic camera: the embedding is a smooth function of the joints.
        fn observe(j: &[f32; DOF]) -> Vec<f32> {
            let mut z = vec![0.0f32; 16];
            for (i, v) in z.iter_mut().enumerate() {
                *v = 0.3 + (j[i % DOF] * (1.0 + i as f32 * 0.1)).sin() * 0.5;
            }
            normalize_l2(&z)
        }
        let safety = SafetyGuard::default();
        let mut rng = rand::rng();
        let mut agent = LatentAgent::default();
        let mut joints = [0.0f32; DOF];
        let mut gripper = 0.5;

        // Exploring: random babbling fills the world model.
        agent.start();
        for t in 0..30 {
            let cmd = agent.observe(&observe(&joints), joints, gripper, &safety, t, &mut rng).unwrap();
            joints = cmd.joints;
            gripper = cmd.gripper;
            agent.arrived();
        }
        assert!(agent.world.is_ready());
        assert_eq!(agent.progress().policy, "random");

        // Mode C: plan toward the goal; energy must drop substantially.
        let goal_pose = [0.4, -0.3, 0.3, 0.2, -0.2, 0.25];
        agent.set_goal(observe(&goal_pose), None);
        let start = energy(&observe(&joints), &observe(&goal_pose));
        for t in 100..260 {
            match agent.observe(&observe(&joints), joints, gripper, &safety, t, &mut rng) {
                Some(cmd) => {
                    joints = cmd.joints;
                    gripper = cmd.gripper;
                    agent.arrived();
                }
                None => break,
            }
        }
        let p = agent.progress();
        assert!(p.best_energy.unwrap() < start * 0.5, "start {start} progress {p:?}");
        assert!(p.world.transitions > 30);
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
