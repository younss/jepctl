//! Robot control subsystem: a hardware abstraction layer for a 6 DOF arm with a
//! gripper, a safety supervisor, and a controller that drives the arm from manual
//! targets, few-shot gesture detections (shadowing) or latent-goal convergence.
//!
//! Everything shared between the HTTP handlers, the WebSocket telemetry and the
//! control loop lives in [`RobotCore`] behind one `Arc<Mutex<_>>`; the loop ticks at
//! [`CONTROL_HZ`] and publishes a [`RobotTelemetry`] snapshot on a `watch` channel.

pub mod controller;
pub mod hal;
pub mod safety;
pub mod sim_view;
pub mod world_model;

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{watch, Mutex};

use crate::robot::hal::{BackendKind, RobotBackend, VirtualWebGlBackend};
use crate::robot::safety::{JointLimits, SafetyGuard};

/// Number of arm joints (the gripper is separate).
pub const DOF: usize = 6;

/// Control loop rate; telemetry is published at the same rate.
pub const CONTROL_HZ: u32 = 30;

#[derive(Error, Debug)]
pub enum RobotError {
    #[error("Emergency stop is engaged; reset safety before moving")]
    EStopEngaged,

    #[error("Joint {joint} target {value:.3} rad is outside [{min:.3}, {max:.3}]")]
    JointOutOfRange { joint: usize, value: f32, min: f32, max: f32 },

    #[error("Gripper target {0:.3} is outside [0, 1]")]
    GripperOutOfRange(f32),

    #[error("Backend '{0}' is not connected")]
    NotConnected(String),

    #[error("Serial error: {0}")]
    Serial(String),

    #[error("Feature '{0}' is not compiled in (rebuild with --features {0})")]
    FeatureMissing(&'static str),

    #[error("Invalid request: {0}")]
    Invalid(String),
}

/// Operating mode of the controller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RobotMode {
    /// Targets come from the UI sliders or the API.
    #[default]
    Manual,
    /// Mode A: gesture detections are mapped to joint deltas or poses.
    Shadowing,
    /// Mirror: every gesture mapped to a pose is blended by similarity, so the arm
    /// follows the person continuously between the poses it was taught.
    Mirror,
    /// Learn the latent world model by moving and watching (no goal yet).
    Exploring,
    /// Mode C: reach a goal embedding by planning inside the learned world model.
    GoalSeeking,
}

/// A full joint command.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JointCommand {
    pub joints: [f32; DOF],
    pub gripper: f32,
}

/// What a detected gesture does in shadowing mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum GestureAction {
    OpenGripper,
    CloseGripper,
    /// Add `delta` radians to `joint` (0 based) per detection tick.
    JointDelta {
        joint: usize,
        delta: f32,
    },
    /// Move to a fixed pose.
    Pose {
        joints: [f32; DOF],
        gripper: f32,
    },
    /// Approve the pending trajectory (Mode B validation gesture).
    Approve,
    /// Engage the emergency stop.
    Stop,
}

/// Progress of learning and goal seeking (modes Exploring and GoalSeeking).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GoalProgress {
    pub has_goal: bool,
    pub goal_dimension: usize,
    /// Energy of the last camera observation against the goal.
    pub current_energy: Option<f32>,
    /// Energy the world model predicted for the action it chose.
    pub predicted_energy: Option<f32>,
    pub best_energy: Option<f32>,
    /// Energy measured at the first observation after the goal was captured.
    pub initial_energy: Option<f32>,
    /// Steps taken (one action, one observation each).
    pub steps: u32,
    /// Current action scale in radians.
    pub step_rad: f32,
    /// 0 to 1: how far energy has dropped from initial toward zero.
    pub convergence: f32,
    /// `idle`, `awaiting observation`, `moving`, `converged`.
    pub phase: String,
    /// Energy under which the goal counts as reached (noise floor times 1.3).
    pub reached_below: f32,
    /// `random` while the model is not ready, `planned` afterwards.
    pub policy: String,
    pub last_action: Option<[f32; world_model::ACTION_DIM]>,
    pub world: world_model::WorldModelStats,
    /// The arm is settled and the controller is waiting for a camera observation.
    #[serde(default)]
    pub awaiting_observation: bool,
}

/// Snapshot published to WebSocket clients and returned by `/api/robot/status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotTelemetry {
    pub backend: BackendKind,
    pub connected: bool,
    pub mode: RobotMode,
    pub estop: bool,
    /// Mode B: targets for the physical arm wait for approval.
    pub safety_gate: bool,
    pub joints: [f32; DOF],
    pub gripper: f32,
    pub targets: [f32; DOF],
    pub gripper_target: f32,
    /// Command held by the safety gate, previewed on the twin only.
    pub pending: Option<JointCommand>,
    pub limits: JointLimits,
    pub max_rad_per_s: f32,
    pub goal: GoalProgress,
    pub last_gesture: Option<String>,
    /// Mirror mode: the taught poses and the weight each one has in the blend.
    #[serde(default)]
    pub mirror: Vec<crate::companion::MirrorWeight>,
    /// Virtual backend: observations use a frozen camera frame as background.
    pub freeze_background: bool,
    pub has_background: bool,
    /// Monotonic tick counter (30 Hz).
    pub tick: u64,
    pub last_error: Option<String>,
}

/// Everything the control loop and the handlers share.
pub struct RobotCore {
    pub backend: Box<dyn RobotBackend>,
    pub safety: SafetyGuard,
    pub mode: RobotMode,
    /// Mode B enabled. Forced on while the physical backend is selected.
    pub safety_gate: bool,
    pub joints: [f32; DOF],
    pub gripper: f32,
    pub targets: [f32; DOF],
    pub gripper_target: f32,
    pub pending: Option<JointCommand>,
    pub gesture_map: std::collections::HashMap<String, GestureAction>,
    pub last_gesture: Option<String>,
    pub mirror: Vec<crate::companion::MirrorWeight>,
    pub agent: controller::LatentAgent,
    pub tick: u64,
    pub last_error: Option<String>,
    /// Serial settings used when switching to the physical backend.
    pub hardware: hal::HardwareConfig,
    /// Virtual backend only: draw the twin over a frozen camera frame instead of the
    /// live one, so the only thing changing between observations is the arm itself.
    pub freeze_background: bool,
    /// The frozen frame (taken when a learning mode starts or a goal is captured).
    pub background: Option<image::RgbImage>,
}

impl RobotCore {
    pub fn new(hardware: hal::HardwareConfig) -> Self {
        Self {
            backend: Box::new(VirtualWebGlBackend::default()),
            safety: SafetyGuard::default(),
            mode: RobotMode::Manual,
            safety_gate: false,
            joints: [0.0; DOF],
            gripper: 0.5,
            targets: [0.0; DOF],
            gripper_target: 0.5,
            pending: None,
            gesture_map: controller::default_gesture_map(),
            last_gesture: None,
            mirror: Vec::new(),
            agent: controller::LatentAgent::default(),
            tick: 0,
            last_error: None,
            hardware,
            freeze_background: true,
            background: None,
        }
    }

    pub fn telemetry(&self) -> RobotTelemetry {
        RobotTelemetry {
            backend: self.backend.kind(),
            connected: self.backend.is_connected(),
            mode: self.mode,
            estop: self.safety.estop,
            safety_gate: self.safety_gate,
            joints: self.joints,
            gripper: self.gripper,
            targets: self.targets,
            gripper_target: self.gripper_target,
            pending: self.pending,
            limits: self.safety.limits,
            max_rad_per_s: self.safety.max_rad_per_s,
            goal: {
                let mut g = self.agent.progress();
                g.awaiting_observation =
                    self.learning_mode() && self.agent.awaiting() && !self.safety.estop && self.pending.is_none();
                g
            },
            last_gesture: self.last_gesture.clone(),
            mirror: self.mirror.clone(),
            freeze_background: self.freeze_background,
            has_background: self.background.is_some(),
            tick: self.tick,
            last_error: self.last_error.clone(),
        }
    }

    /// Submit a command. Validated against the limits; held by the safety gate when
    /// the physical backend is active and the command is not approved.
    pub fn submit(&mut self, cmd: JointCommand, approved: bool) -> Result<bool, RobotError> {
        self.safety.validate(&cmd)?;
        let gated = self.safety_gate && self.backend.kind() == BackendKind::Physical && !approved;
        if gated {
            self.pending = Some(cmd);
            Ok(false)
        } else {
            self.pending = None;
            self.targets = cmd.joints;
            self.gripper_target = cmd.gripper;
            Ok(true)
        }
    }

    /// Approve and execute the command held by the safety gate.
    pub fn approve_pending(&mut self) -> Result<bool, RobotError> {
        match self.pending.take() {
            Some(cmd) => self.submit(cmd, true),
            None => Ok(false),
        }
    }

    pub fn emergency_stop(&mut self) {
        self.safety.estop = true;
        self.pending = None;
        self.targets = self.joints;
        self.gripper_target = self.gripper;
        self.agent.stop();
        if let Err(e) = self.backend.emergency_stop() {
            self.last_error = Some(e.to_string());
        }
    }

    pub fn reset_safety(&mut self) {
        self.safety.estop = false;
        self.last_error = None;
    }

    /// Modes in which the controller acts and learns from camera observations.
    pub fn learning_mode(&self) -> bool {
        matches!(self.mode, RobotMode::Exploring | RobotMode::GoalSeeking)
    }
}

/// Shared handle: core state plus the telemetry broadcast.
#[derive(Clone)]
pub struct RobotHandle {
    pub core: Arc<Mutex<RobotCore>>,
    pub telemetry_tx: watch::Sender<RobotTelemetry>,
}

impl RobotHandle {
    pub fn new(hardware: hal::HardwareConfig) -> Self {
        let core = RobotCore::new(hardware);
        let (telemetry_tx, _) = watch::channel(core.telemetry());
        Self { core: Arc::new(Mutex::new(core)), telemetry_tx }
    }

    pub fn subscribe(&self) -> watch::Receiver<RobotTelemetry> {
        self.telemetry_tx.subscribe()
    }
}
