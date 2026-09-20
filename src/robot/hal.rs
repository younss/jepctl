//! Hardware abstraction layer: the [`RobotBackend`] trait, the in-memory virtual arm
//! that feeds the WebGL twin, and the serial backend for a physical arm.

use serde::{Deserialize, Serialize};

use crate::robot::{RobotError, DOF};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Virtual,
    Physical,
}

/// A 6 DOF arm with a gripper. Positions are radians; the gripper is 0 (closed) to 1 (open).
pub trait RobotBackend: Send {
    fn kind(&self) -> BackendKind;
    fn is_connected(&self) -> bool;
    fn connect(&mut self) -> Result<(), RobotError>;
    fn disconnect(&mut self) -> Result<(), RobotError>;
    /// Push already ramped, already validated targets to the actuators.
    fn set_joint_targets(&mut self, targets: &[f32; DOF], gripper: f32) -> Result<(), RobotError>;
    fn get_joint_positions(&self) -> [f32; DOF];
    fn get_gripper_position(&self) -> f32;
    /// Cut motion immediately (torque off or hold, depending on the hardware).
    fn emergency_stop(&mut self) -> Result<(), RobotError>;
}

/// Virtual arm: state in memory, rendered by the WebGL twin from telemetry.
#[derive(Debug, Default)]
pub struct VirtualWebGlBackend {
    joints: [f32; DOF],
    gripper: f32,
    connected: bool,
}

impl RobotBackend for VirtualWebGlBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Virtual
    }

    fn is_connected(&self) -> bool {
        self.connected
    }

    fn connect(&mut self) -> Result<(), RobotError> {
        self.connected = true;
        Ok(())
    }

    fn disconnect(&mut self) -> Result<(), RobotError> {
        self.connected = false;
        Ok(())
    }

    fn set_joint_targets(&mut self, targets: &[f32; DOF], gripper: f32) -> Result<(), RobotError> {
        self.joints = *targets;
        self.gripper = gripper;
        Ok(())
    }

    fn get_joint_positions(&self) -> [f32; DOF] {
        self.joints
    }

    fn get_gripper_position(&self) -> f32 {
        self.gripper
    }

    fn emergency_stop(&mut self) -> Result<(), RobotError> {
        Ok(())
    }
}

/// Serial settings for a physical arm (Feetech STS3215 / SO-100 defaults).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HardwareConfig {
    pub port: String,
    pub baud: u32,
    /// Servo bus IDs for joints 1 to 6.
    pub joint_ids: [u8; DOF],
    pub gripper_id: u8,
    /// Encoder ticks per full revolution (4096 for STS3215).
    pub ticks_per_rev: u32,
    /// Tick value at joint angle 0 rad.
    pub zero_ticks: [i32; DOF],
    /// +1 or -1 per joint to match the mechanical direction.
    pub direction: [i8; DOF],
    /// Gripper ticks at closed (0.0) and open (1.0).
    pub gripper_closed_ticks: i32,
    pub gripper_open_ticks: i32,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            port: if cfg!(windows) { "COM3".into() } else { "/dev/ttyUSB0".into() },
            baud: 1_000_000,
            joint_ids: [1, 2, 3, 4, 5, 6],
            gripper_id: 7,
            ticks_per_rev: 4096,
            zero_ticks: [2048; DOF],
            direction: [1; DOF],
            gripper_closed_ticks: 2048,
            gripper_open_ticks: 3000,
        }
    }
}

impl HardwareConfig {
    pub fn rad_to_ticks(&self, joint: usize, rad: f32) -> i32 {
        let ticks = rad / std::f32::consts::TAU * self.ticks_per_rev as f32 * self.direction[joint] as f32;
        (self.zero_ticks[joint] + ticks.round() as i32).clamp(0, self.ticks_per_rev as i32 - 1)
    }

    pub fn ticks_to_rad(&self, joint: usize, ticks: i32) -> f32 {
        (ticks - self.zero_ticks[joint]) as f32 / self.ticks_per_rev as f32
            * std::f32::consts::TAU
            * self.direction[joint] as f32
    }

    pub fn gripper_to_ticks(&self, open: f32) -> i32 {
        let span = (self.gripper_open_ticks - self.gripper_closed_ticks) as f32;
        (self.gripper_closed_ticks as f32 + span * open.clamp(0.0, 1.0)).round() as i32
    }
}

/// Feetech STS / Dynamixel 1.0 style frames.
pub mod protocol {
    pub const HEADER: [u8; 2] = [0xFF, 0xFF];
    pub const BROADCAST_ID: u8 = 0xFE;
    pub const INSTR_READ: u8 = 0x02;
    pub const INSTR_WRITE: u8 = 0x03;
    pub const INSTR_SYNC_WRITE: u8 = 0x83;
    /// STS3215 register map (subset).
    pub const REG_TORQUE_ENABLE: u8 = 0x28;
    pub const REG_GOAL_POSITION: u8 = 0x2A;
    pub const REG_PRESENT_POSITION: u8 = 0x38;

    fn checksum(body: &[u8]) -> u8 {
        !(body.iter().fold(0u32, |a, b| a + *b as u32) as u8)
    }

    /// `FF FF ID LEN INSTR PARAMS... CHK`
    pub fn frame(id: u8, instr: u8, params: &[u8]) -> Vec<u8> {
        let len = (params.len() + 2) as u8;
        let mut body = vec![id, len, instr];
        body.extend_from_slice(params);
        let chk = checksum(&body);
        let mut out = HEADER.to_vec();
        out.extend_from_slice(&body);
        out.push(chk);
        out
    }

    pub fn write_u16(id: u8, reg: u8, value: u16) -> Vec<u8> {
        frame(id, INSTR_WRITE, &[reg, (value & 0xFF) as u8, (value >> 8) as u8])
    }

    pub fn write_u8(id: u8, reg: u8, value: u8) -> Vec<u8> {
        frame(id, INSTR_WRITE, &[reg, value])
    }

    /// One frame setting a 16 bit register on many servos.
    pub fn sync_write_u16(reg: u8, values: &[(u8, u16)]) -> Vec<u8> {
        let mut params = vec![reg, 2];
        for (id, v) in values {
            params.extend_from_slice(&[*id, (v & 0xFF) as u8, (v >> 8) as u8]);
        }
        frame(BROADCAST_ID, INSTR_SYNC_WRITE, &params)
    }

    pub fn read_u16_request(id: u8, reg: u8) -> Vec<u8> {
        frame(id, INSTR_READ, &[reg, 2])
    }

    /// Parse a status packet `FF FF ID LEN ERR D0 D1 CHK`; returns the value.
    pub fn parse_u16_response(buf: &[u8], expect_id: u8) -> Option<u16> {
        let start = buf.windows(2).position(|w| w == HEADER)?;
        let p = &buf[start..];
        if p.len() < 8 || p[2] != expect_id || p[3] < 4 {
            return None;
        }
        let body_len = p[3] as usize + 2;
        if p.len() < 2 + body_len || checksum(&p[2..2 + body_len - 1]) != p[2 + body_len - 1] {
            return None;
        }
        Some(u16::from_le_bytes([p[5], p[6]]))
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn frames_have_valid_checksums() {
            let f = write_u16(1, REG_GOAL_POSITION, 2048);
            assert_eq!(&f[..2], &HEADER);
            assert_eq!(f[2], 1);
            assert_eq!(f[3], 5);
            assert_eq!(f[4], INSTR_WRITE);
            assert_eq!(&f[5..8], &[0x2A, 0x00, 0x08]);
            let body_sum: u32 = f[2..f.len() - 1].iter().map(|b| *b as u32).sum();
            assert_eq!(f[f.len() - 1], !(body_sum as u8));

            let s = sync_write_u16(REG_GOAL_POSITION, &[(1, 100), (2, 200)]);
            assert_eq!(s[2], BROADCAST_ID);
            assert_eq!(s[4], INSTR_SYNC_WRITE);
            assert_eq!(s.len(), 2 + 3 + 2 + 6 + 1);

            let mut resp = frame(3, 0, &[0x00, 0x10]);
            // Turn the frame into a status packet: ERR byte then data.
            resp[4] = 0;
            let resp = {
                let body = [3u8, 4, 0, 0x00, 0x10];
                let mut v = HEADER.to_vec();
                v.extend_from_slice(&body);
                v.push(!(body.iter().map(|b| *b as u32).sum::<u32>() as u8));
                v
            };
            assert_eq!(parse_u16_response(&resp, 3), Some(0x1000));
            assert_eq!(parse_u16_response(&resp, 4), None);
        }
    }
}

/// Physical arm over USB serial. Compiled only with `--features serial`.
#[cfg(feature = "serial")]
pub struct PhysicalSerialBackend {
    cfg: HardwareConfig,
    port: Option<Box<dyn serialport::SerialPort>>,
    joints: [f32; DOF],
    gripper: f32,
}

#[cfg(feature = "serial")]
impl PhysicalSerialBackend {
    pub fn new(cfg: HardwareConfig) -> Self {
        Self { cfg, port: None, joints: [0.0; DOF], gripper: 0.5 }
    }

    fn port_mut(&mut self) -> Result<&mut Box<dyn serialport::SerialPort>, RobotError> {
        self.port.as_mut().ok_or_else(|| RobotError::NotConnected("physical".into()))
    }

    fn write_all(&mut self, bytes: &[u8]) -> Result<(), RobotError> {
        let port = self.port_mut()?;
        std::io::Write::write_all(port, bytes).map_err(|e| RobotError::Serial(e.to_string()))?;
        std::io::Write::flush(port).map_err(|e| RobotError::Serial(e.to_string()))
    }

    /// Read back every servo's present position (best effort; keeps last values on timeout).
    pub fn poll_positions(&mut self) {
        for i in 0..DOF {
            let id = self.cfg.joint_ids[i];
            if let Some(ticks) = self.read_u16(id, protocol::REG_PRESENT_POSITION) {
                self.joints[i] = self.cfg.ticks_to_rad(i, ticks as i32);
            }
        }
    }

    fn read_u16(&mut self, id: u8, reg: u8) -> Option<u16> {
        let req = protocol::read_u16_request(id, reg);
        self.write_all(&req).ok()?;
        let port = self.port.as_mut()?;
        let mut buf = [0u8; 32];
        let n = std::io::Read::read(port, &mut buf).ok()?;
        protocol::parse_u16_response(&buf[..n], id)
    }
}

#[cfg(feature = "serial")]
impl RobotBackend for PhysicalSerialBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Physical
    }

    fn is_connected(&self) -> bool {
        self.port.is_some()
    }

    fn connect(&mut self) -> Result<(), RobotError> {
        let port = serialport::new(&self.cfg.port, self.cfg.baud)
            .timeout(std::time::Duration::from_millis(20))
            .open()
            .map_err(|e| RobotError::Serial(format!("{}: {e}", self.cfg.port)))?;
        self.port = Some(port);
        // Torque on for every servo.
        for id in self.cfg.joint_ids.into_iter().chain(std::iter::once(self.cfg.gripper_id)) {
            self.write_all(&protocol::write_u8(id, protocol::REG_TORQUE_ENABLE, 1))?;
        }
        self.poll_positions();
        tracing::info!("Serial arm connected on {} @ {} baud", self.cfg.port, self.cfg.baud);
        Ok(())
    }

    fn disconnect(&mut self) -> Result<(), RobotError> {
        if self.port.is_some() {
            for id in self.cfg.joint_ids.into_iter().chain(std::iter::once(self.cfg.gripper_id)) {
                let _ = self.write_all(&protocol::write_u8(id, protocol::REG_TORQUE_ENABLE, 0));
            }
        }
        self.port = None;
        Ok(())
    }

    fn set_joint_targets(&mut self, targets: &[f32; DOF], gripper: f32) -> Result<(), RobotError> {
        let mut values: Vec<(u8, u16)> =
            (0..DOF).map(|i| (self.cfg.joint_ids[i], self.cfg.rad_to_ticks(i, targets[i]) as u16)).collect();
        values.push((self.cfg.gripper_id, self.cfg.gripper_to_ticks(gripper) as u16));
        self.write_all(&protocol::sync_write_u16(protocol::REG_GOAL_POSITION, &values))?;
        // Trust the command until the next poll.
        self.joints = *targets;
        self.gripper = gripper;
        Ok(())
    }

    fn get_joint_positions(&self) -> [f32; DOF] {
        self.joints
    }

    fn get_gripper_position(&self) -> f32 {
        self.gripper
    }

    fn emergency_stop(&mut self) -> Result<(), RobotError> {
        // Hold position: re-send the current position as goal, then torque off.
        let current = self.joints;
        let grip = self.gripper;
        let _ = self.set_joint_targets(&current, grip);
        for id in self.cfg.joint_ids.into_iter().chain(std::iter::once(self.cfg.gripper_id)) {
            self.write_all(&protocol::write_u8(id, protocol::REG_TORQUE_ENABLE, 0))?;
        }
        Ok(())
    }
}

/// Build the backend for a kind; the physical one needs the `serial` feature.
pub fn make_backend(kind: BackendKind, hardware: &HardwareConfig) -> Result<Box<dyn RobotBackend>, RobotError> {
    match kind {
        BackendKind::Virtual => Ok(Box::new(VirtualWebGlBackend::default())),
        BackendKind::Physical => {
            #[cfg(feature = "serial")]
            {
                Ok(Box::new(PhysicalSerialBackend::new(hardware.clone())))
            }
            #[cfg(not(feature = "serial"))]
            {
                let _ = hardware;
                Err(RobotError::FeatureMissing("serial"))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tick_conversion_round_trips() {
        let cfg = HardwareConfig::default();
        assert_eq!(cfg.rad_to_ticks(0, 0.0), 2048);
        let t = cfg.rad_to_ticks(1, 1.0);
        assert!((cfg.ticks_to_rad(1, t) - 1.0).abs() < 0.002);
        assert_eq!(cfg.gripper_to_ticks(0.0), 2048);
        assert_eq!(cfg.gripper_to_ticks(1.0), 3000);
        assert_eq!(cfg.rad_to_ticks(0, 100.0), 4095);
    }

    #[test]
    fn virtual_backend_tracks_targets() {
        let mut b = VirtualWebGlBackend::default();
        assert_eq!(b.kind(), BackendKind::Virtual);
        b.connect().unwrap();
        b.set_joint_targets(&[0.1; DOF], 0.9).unwrap();
        assert_eq!(b.get_joint_positions(), [0.1; DOF]);
        assert_eq!(b.get_gripper_position(), 0.9);
        assert!(b.is_connected());
    }

    #[cfg(not(feature = "serial"))]
    #[test]
    fn physical_backend_needs_the_feature() {
        assert!(matches!(
            make_backend(BackendKind::Physical, &HardwareConfig::default()),
            Err(RobotError::FeatureMissing("serial"))
        ));
    }
}
