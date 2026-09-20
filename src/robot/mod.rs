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
    /// Mode C: random local exploration minimising latent energy to a goal embedding.
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

/// Progress of the latent goal search (Mode C).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GoalProgress {
    pub has_goal: bool,
    pub goal_dimension: usize,
    /// Energy of the last observation.
    pub current_energy: Option<f32>,
    /// Best energy reached so far.
    pub best_energy: Option<f32>,
    /// Energy measured when the goal was captured (upper reference).
    pub initial_energy: Option<f32>,
    pub iterations: u32,
    pub candidates_evaluated: u32,
    /// Current perturbation scale in radians.
    pub step_rad: f32,
    /// 0 to 1: how far energy has dropped from initial toward zero.
    pub convergence: f32,
    pub phase: String,
    /// The arm is settled at the explorer's pose and an observation is expected now.
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
    pub explorer: controller::GoalExplorer,
    pub tick: u64,
    pub last_error: Option<String>,
    /// Serial settings used when switching to the physical backend.
    pub hardware: hal::HardwareConfig,
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
            explorer: controller::GoalExplorer::default(),
            tick: 0,
            last_error: None,
            hardware,
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
                let mut g = self.explorer.progress();
                g.awaiting_observation = self.mode == RobotMode::GoalSeeking
                    && self.explorer.is_active()
                    && !self.safety.estop
                    && self.settled()
                    && self.pending.is_none();
                g
            },
            last_gesture: self.last_gesture.clone(),
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
        self.explorer.abort();
        if let Err(e) = self.backend.emergency_stop() {
            self.last_error = Some(e.to_string());
        }
    }

    pub fn reset_safety(&mut self) {
        self.safety.estop = false;
        self.last_error = None;
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
