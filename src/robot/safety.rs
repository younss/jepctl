//! Safety supervisor: joint limits, velocity ramp and emergency stop.

use serde::{Deserialize, Serialize};

use crate::robot::{DOF, JointCommand, RobotError};

/// Per joint angle limits in radians, `[min, max]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct JointLimits {
    pub min: [f32; DOF],
    pub max: [f32; DOF],
}

impl Default for JointLimits {
    /// Base, shoulder, elbow, wrist pitch, wrist roll, wrist rotate.
    fn default() -> Self {
        Self { min: [-2.6, -1.8, -2.0, -1.8, -2.6, -3.1], max: [2.6, 1.8, 2.0, 1.8, 2.6, 3.1] }
    }
}

impl JointLimits {
    pub fn clamp(&self, joints: [f32; DOF]) -> [f32; DOF] {
        let mut out = joints;
        for i in 0..DOF {
            out[i] = joints[i].clamp(self.min[i], self.max[i]);
        }
        out
    }
}

/// Default maximum angular speed per joint.
pub const DEFAULT_MAX_RAD_PER_S: f32 = 1.5;

/// Gripper opening speed (fraction of full travel per second).
pub const GRIPPER_UNITS_PER_S: f32 = 1.0;

#[derive(Debug, Clone)]
pub struct SafetyGuard {
    pub limits: JointLimits,
    pub max_rad_per_s: f32,
    pub estop: bool,
}

impl Default for SafetyGuard {
    fn default() -> Self {
        Self { limits: JointLimits::default(), max_rad_per_s: DEFAULT_MAX_RAD_PER_S, estop: false }
    }
}

impl SafetyGuard {
    /// Reject commands outside the mechanical envelope or while stopped.
    pub fn validate(&self, cmd: &JointCommand) -> Result<(), RobotError> {
        if self.estop {
            return Err(RobotError::EStopEngaged);
        }
        for (i, v) in cmd.joints.iter().enumerate() {
            if !v.is_finite() || *v < self.limits.min[i] || *v > self.limits.max[i] {
                return Err(RobotError::JointOutOfRange {
                    joint: i + 1,
                    value: *v,
                    min: self.limits.min[i],
                    max: self.limits.max[i],
                });
            }
        }
        if !cmd.gripper.is_finite() || !(0.0..=1.0).contains(&cmd.gripper) {
            return Err(RobotError::GripperOutOfRange(cmd.gripper));
        }
        Ok(())
    }

    /// Advance `current` toward `target` by at most `max_rad_per_s * dt` per joint.
    /// Returns the new position; never jumps, never leaves the limits.
    pub fn ramp(&self, current: [f32; DOF], target: [f32; DOF], dt: f32) -> [f32; DOF] {
        if self.estop {
            return current;
        }
        let max_step = self.max_rad_per_s * dt;
        let mut out = current;
        for i in 0..DOF {
            let goal = target[i].clamp(self.limits.min[i], self.limits.max[i]);
            let delta = goal - current[i];
            out[i] = current[i] + delta.clamp(-max_step, max_step);
        }
        out
    }

    pub fn ramp_gripper(&self, current: f32, target: f32, dt: f32) -> f32 {
        if self.estop {
            return current;
        }
        let max_step = GRIPPER_UNITS_PER_S * dt;
        let delta = target.clamp(0.0, 1.0) - current;
        (current + delta.clamp(-max_step, max_step)).clamp(0.0, 1.0)
    }

    /// Whether the arm has reached its targets (within `eps` rad).
    pub fn settled(current: [f32; DOF], target: [f32; DOF], eps: f32) -> bool {
        current.iter().zip(target.iter()).all(|(c, t)| (c - t).abs() <= eps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_rejects_out_of_range_and_estop() {
        let mut g = SafetyGuard::default();
        let ok = JointCommand { joints: [0.0; DOF], gripper: 0.5 };
        assert!(g.validate(&ok).is_ok());
        let bad = JointCommand { joints: [3.0, 0.0, 0.0, 0.0, 0.0, 0.0], gripper: 0.5 };
        assert!(matches!(g.validate(&bad), Err(RobotError::JointOutOfRange { joint: 1, .. })));
        let bad_grip = JointCommand { joints: [0.0; DOF], gripper: 1.5 };
        assert!(matches!(g.validate(&bad_grip), Err(RobotError::GripperOutOfRange(_))));
        g.estop = true;
        assert!(matches!(g.validate(&ok), Err(RobotError::EStopEngaged)));
    }

    #[test]
    fn ramp_limits_speed_and_never_jumps() {
        let g = SafetyGuard::default();
        let current = [0.0; DOF];
        let target = [1.0, -1.0, 0.01, 0.0, 0.0, 0.0];
        let next = g.ramp(current, target, 1.0 / 30.0);
        let step = 1.5 / 30.0;
        assert!((next[0] - step).abs() < 1e-6);
        assert!((next[1] + step).abs() < 1e-6);
        assert!((next[2] - 0.01).abs() < 1e-6);
        // Targets beyond the limit are clamped, not chased.
        let far = g.ramp([2.5, 0.0, 0.0, 0.0, 0.0, 0.0], [9.0, 0.0, 0.0, 0.0, 0.0, 0.0], 10.0);
        assert!((far[0] - 2.6).abs() < 1e-6);
        assert!(SafetyGuard::settled(far, [2.6, 0.0, 0.0, 0.0, 0.0, 0.0], 1e-4));
        let mut stopped = g.clone();
        stopped.estop = true;
        assert_eq!(stopped.ramp(current, target, 1.0), current);
    }
}
