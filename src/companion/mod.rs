//! Companion robot: a virtual WebGL character that learns from the person in front
//! of it by watching (camera, vision model) and hearing (microphone, audio model).
//!
//! Teaching is few-shot: the user shows a pose or makes a sound, names it, and picks
//! what the companion should do when it recognises it (a behaviour). Recognition
//! uses the same prototype matching as the gesture sandbox, in both modalities.
//! Taught poses mapped to a body pose are blended by similarity ("mirror"), so the
//! companion follows the user continuously between the poses it was taught.
//!
//! Attention is not learned: the head follows where the picture moves, so the
//! companion visibly watches the person even before anything was taught.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::{watch, Mutex};

use crate::gestures::GestureScore;

/// Control loop and telemetry rate.
pub const CONTROL_HZ: u32 = 30;

/// Number of pose parameters (see [`CompanionPose`]).
pub const POSE_DIM: usize = 6;

/// Softmax temperature for mirror blending: how sharply the best matching pose
/// dominates the blend.
pub const MIRROR_TEMPERATURE: f32 = 0.06;

/// Poses scoring less than this below the best one get no weight.
pub const MIRROR_WINDOW: f32 = 0.20;

/// Minimum blended score before mirroring moves the body at all.
pub const MIRROR_FLOOR: f32 = 0.35;

/// Body parameters of the companion, all in `[-1, 1]` except `mood` in `[0, 1]`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CompanionPose {
    /// Head yaw: negative looks to the companion's right (the viewer's left).
    pub head_pan: f32,
    /// Head pitch: positive looks up.
    pub head_tilt: f32,
    /// Arm elevation: -1 hanging, 1 straight up.
    pub left_arm: f32,
    pub right_arm: f32,
    /// Body lean forward (positive) or back.
    pub lean: f32,
    /// 0 calm (blue) to 1 excited (orange); drives the chest light.
    pub mood: f32,
}

impl Default for CompanionPose {
    fn default() -> Self {
        Self { head_pan: 0.0, head_tilt: 0.0, left_arm: -0.8, right_arm: -0.8, lean: 0.0, mood: 0.3 }
    }
}

impl CompanionPose {
    pub fn as_array(&self) -> [f32; POSE_DIM] {
        [self.head_pan, self.head_tilt, self.left_arm, self.right_arm, self.lean, self.mood]
    }

    pub fn from_array(a: [f32; POSE_DIM]) -> Self {
        Self { head_pan: a[0], head_tilt: a[1], left_arm: a[2], right_arm: a[3], lean: a[4], mood: a[5] }
    }

    pub fn clamped(&self) -> Self {
        let mut a = self.as_array();
        for (i, v) in a.iter_mut().enumerate() {
            *v = if i == POSE_DIM - 1 { v.clamp(0.0, 1.0) } else { v.clamp(-1.0, 1.0) };
        }
        Self::from_array(a)
    }

    /// Move toward `target` by at most `max_step` per parameter.
    fn approach(&self, target: &Self, max_step: f32) -> Self {
        let mut a = self.as_array();
        let t = target.as_array();
        for (v, goal) in a.iter_mut().zip(t.iter()) {
            let d = goal - *v;
            *v += d.clamp(-max_step, max_step);
        }
        Self::from_array(a)
    }
}

/// What the companion does when a cue is recognised.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "behaviour", rename_all = "snake_case")]
pub enum Behaviour {
    /// Hold this body pose; poses taught this way are blended by similarity.
    Pose {
        pose: CompanionPose,
    },
    Nod,
    Shake,
    WaveLeft,
    WaveRight,
    Cheer,
    Dance,
    /// Flinch back with the arms up, then relax (loud or unknown sounds).
    Startle,
    Sleep,
    /// Set the mood light without moving.
    Mood {
        value: f32,
    },
    /// Short "got it" reaction played when something was taught.
    Acknowledge,
}

impl Behaviour {
    /// Length of the animation in ticks (`None` for poses and mood, which hold).
    fn duration_ticks(&self) -> Option<u32> {
        match self {
            Behaviour::Pose { .. } | Behaviour::Mood { .. } => None,
            Behaviour::Nod | Behaviour::Shake => Some(CONTROL_HZ * 3 / 2),
            Behaviour::WaveLeft | Behaviour::WaveRight => Some(CONTROL_HZ * 2),
            Behaviour::Cheer => Some(CONTROL_HZ * 2),
            Behaviour::Dance => Some(CONTROL_HZ * 4),
            Behaviour::Startle => Some(CONTROL_HZ * 3 / 2),
            Behaviour::Sleep => Some(CONTROL_HZ * 6),
            Behaviour::Acknowledge => Some(CONTROL_HZ * 6 / 5),
        }
    }

    pub fn label(&self) -> String {
        match self {
            Behaviour::Pose { .. } => "pose".into(),
            Behaviour::Nod => "nod".into(),
            Behaviour::Shake => "shake".into(),
            Behaviour::WaveLeft => "wave_left".into(),
            Behaviour::WaveRight => "wave_right".into(),
            Behaviour::Cheer => "cheer".into(),
            Behaviour::Dance => "dance".into(),
            Behaviour::Startle => "startle".into(),
            Behaviour::Sleep => "sleep".into(),
            Behaviour::Mood { .. } => "mood".into(),
            Behaviour::Acknowledge => "acknowledge".into(),
        }
    }
}

/// Where a cue comes from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CueKind {
    /// A registered gesture (vision model, camera).
    Gesture,
    /// A registered sound (audio model, microphone).
    Sound,
}

/// A taught association: when `name` of `kind` is recognised, do `behaviour`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Cue {
    pub kind: CueKind,
    pub name: String,
    /// Flattened: `{"kind": "sound", "name": "clap", "behaviour": "nod"}` or
    /// `{"behaviour": "pose", "pose": {...}}`.
    #[serde(flatten)]
    pub behaviour: Behaviour,
}

impl Cue {
    pub fn key(kind: CueKind, name: &str) -> String {
        match kind {
            CueKind::Gesture => format!("gesture:{name}"),
            CueKind::Sound => format!("sound:{name}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum CompanionMode {
    /// The sliders and the API set the pose; nothing is observed.
    #[default]
    Manual,
    /// Watch and listen: attention, mirroring and cues drive the body.
    Interactive,
}

/// Where the picture is moving (from consecutive camera frames).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize, Default)]
pub struct Attention {
    /// Motion centroid, -1 (left of the frame) to 1 (right).
    pub x: f32,
    /// Motion centroid, -1 (bottom) to 1 (top).
    pub y: f32,
    /// Fraction of the frame that changed, 0 to 1.
    pub motion: f32,
    /// Something moved recently enough to be worth looking at.
    pub tracking: bool,
}

/// One blended contribution in mirror mode.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MirrorWeight {
    pub name: String,
    pub weight: f32,
    pub score: f32,
}

/// What the companion currently hears.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Hearing {
    pub level: f32,
    pub last_sound: Option<String>,
    pub last_sound_confidence: f32,
    pub last_sound_at: Option<f64>,
}

/// What the companion currently sees.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub struct Seeing {
    pub last_gesture: Option<String>,
    pub last_gesture_confidence: f32,
    pub last_gesture_at: Option<f64>,
    pub mirror: Vec<MirrorWeight>,
}

/// Snapshot published to WebSocket clients and `/api/companion/status`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompanionTelemetry {
    pub mode: CompanionMode,
    pub pose: CompanionPose,
    pub target: CompanionPose,
    pub animation: Option<String>,
    pub animation_progress: f32,
    pub attention: Attention,
    pub hearing: Hearing,
    pub seeing: Seeing,
    pub cues: Vec<Cue>,
    /// Eyes closed this tick (blink or sleep).
    pub eyes_closed: bool,
    pub last_behaviour: Option<String>,
    pub tick: u64,
    pub last_error: Option<String>,
}

/// Persisted part of the state.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompanionMemory {
    pub cues: Vec<Cue>,
}

struct Animation {
    behaviour: Behaviour,
    started: u64,
    duration: u32,
}

/// Everything the loop, the observers and the handlers share.
pub struct CompanionCore {
    pub mode: CompanionMode,
    pub pose: CompanionPose,
    /// Pose the body settles to when no animation runs (manual, mirror or rest).
    pub target: CompanionPose,
    /// Blended mirror target, applied on top of `target` in interactive mode.
    mirror_target: Option<CompanionPose>,
    animation: Option<Animation>,
    pub cues: HashMap<String, Cue>,
    pub attention: Attention,
    attention_updated: u64,
    pub hearing: Hearing,
    pub seeing: Seeing,
    pub last_behaviour: Option<String>,
    /// Tick of the last triggered animation, for cooldown.
    last_trigger: HashMap<String, u64>,
    pub tick: u64,
    pub last_error: Option<String>,
    pub dirty: bool,
}

impl Default for CompanionCore {
    fn default() -> Self {
        Self::new()
    }
}

impl CompanionCore {
    pub fn new() -> Self {
        Self {
            mode: CompanionMode::Manual,
            pose: CompanionPose::default(),
            target: CompanionPose::default(),
            mirror_target: None,
            animation: None,
            cues: HashMap::new(),
            attention: Attention::default(),
            attention_updated: 0,
            hearing: Hearing::default(),
            seeing: Seeing::default(),
            last_behaviour: None,
            last_trigger: HashMap::new(),
            tick: 0,
            last_error: None,
            dirty: false,
        }
    }

    pub fn memory(&self) -> CompanionMemory {
        let mut cues: Vec<Cue> = self.cues.values().cloned().collect();
        cues.sort_by(|a, b| Cue::key(a.kind, &a.name).cmp(&Cue::key(b.kind, &b.name)));
        CompanionMemory { cues }
    }

    pub fn restore(&mut self, memory: CompanionMemory) {
        self.cues = memory.cues.into_iter().map(|c| (Cue::key(c.kind, &c.name), c)).collect();
    }

    pub fn telemetry(&self) -> CompanionTelemetry {
        let (animation, progress) = match &self.animation {
            Some(a) => {
                (Some(a.behaviour.label()), ((self.tick - a.started) as f32 / a.duration.max(1) as f32).min(1.0))
            }
            None => (None, 0.0),
        };
        let sleeping = matches!(self.animation.as_ref().map(|a| &a.behaviour), Some(Behaviour::Sleep));
        // A blink every ~4 s lasting 4 ticks.
        let blink = self.tick % (CONTROL_HZ as u64 * 4) < 4;
        CompanionTelemetry {
            mode: self.mode,
            pose: self.pose,
            target: self.effective_target(),
            animation,
            animation_progress: progress,
            attention: self.attention,
            hearing: self.hearing.clone(),
            seeing: self.seeing.clone(),
            cues: self.memory().cues,
            eyes_closed: blink || sleeping,
            last_behaviour: self.last_behaviour.clone(),
            tick: self.tick,
            last_error: self.last_error.clone(),
        }
    }

    pub fn set_cue(&mut self, cue: Cue) {
        self.cues.insert(Cue::key(cue.kind, &cue.name), cue);
        self.dirty = true;
    }

    pub fn remove_cue(&mut self, kind: CueKind, name: &str) -> bool {
        let removed = self.cues.remove(&Cue::key(kind, name)).is_some();
        self.dirty |= removed;
        removed
    }

    /// Set the manual target (clamped). Also stops mirroring until the next observation.
    pub fn set_target(&mut self, pose: CompanionPose) {
        self.target = pose.clamped();
        self.mirror_target = None;
    }

    pub fn set_mode(&mut self, mode: CompanionMode) {
        self.mode = mode;
        self.mirror_target = None;
        if mode == CompanionMode::Manual {
            self.seeing.mirror.clear();
            self.attention.tracking = false;
        }
    }

    /// Start a behaviour now (from a cue or the API).
    pub fn perform(&mut self, behaviour: Behaviour) {
        self.last_behaviour = Some(behaviour.label());
        match behaviour.duration_ticks() {
            Some(duration) => self.animation = Some(Animation { behaviour, started: self.tick, duration }),
            None => match behaviour {
                Behaviour::Pose { pose } => self.set_target(pose),
                Behaviour::Mood { value } => {
                    self.target.mood = value.clamp(0.0, 1.0);
                    self.pose.mood = self.target.mood;
                }
                _ => {}
            },
        }
    }

    /// Camera motion update from the observer (centroid in frame coordinates).
    pub fn observe_motion(&mut self, x: f32, y: f32, motion: f32) {
        let tracking = motion > 0.004;
        if tracking {
            // Smooth so the head does not jitter with every pixel of noise.
            let k = 0.35;
            self.attention.x += (x.clamp(-1.0, 1.0) - self.attention.x) * k;
            self.attention.y += (y.clamp(-1.0, 1.0) - self.attention.y) * k;
            self.attention_updated = self.tick;
        }
        self.attention.motion = motion;
        self.attention.tracking = tracking || self.tick.saturating_sub(self.attention_updated) < CONTROL_HZ as u64 * 2;
    }

    /// Gesture scores from the vision observer: mirror blending plus cue triggers.
    pub fn observe_gestures(&mut self, scores: &[GestureScore], matched: Option<&str>, confidence: f32, now: f64) {
        if self.mode != CompanionMode::Interactive {
            return;
        }
        if let Some(name) = matched {
            self.seeing.last_gesture = Some(name.to_string());
            self.seeing.last_gesture_confidence = confidence;
            self.seeing.last_gesture_at = Some(now);
            if let Some(cue) = self.cues.get(&Cue::key(CueKind::Gesture, name)).cloned() {
                if !matches!(cue.behaviour, Behaviour::Pose { .. }) {
                    self.trigger(&Cue::key(CueKind::Gesture, name), cue.behaviour);
                }
            }
        }
        // Mirror: blend every taught pose by similarity.
        let mut candidates: Vec<(String, f32, CompanionPose)> = Vec::new();
        for s in scores {
            if let Some(Cue { behaviour: Behaviour::Pose { pose }, .. }) =
                self.cues.get(&Cue::key(CueKind::Gesture, &s.name))
            {
                candidates.push((s.name.clone(), s.combined, *pose));
            }
        }
        let (weights, blended) = blend_poses(&candidates);
        self.seeing.mirror = weights;
        self.mirror_target = blended;
    }

    /// Sound match from the audio observer.
    pub fn observe_sound(&mut self, level: f32, matched: Option<&str>, confidence: f32, now: f64) {
        self.hearing.level = level;
        if self.mode != CompanionMode::Interactive {
            return;
        }
        if let Some(name) = matched {
            self.hearing.last_sound = Some(name.to_string());
            self.hearing.last_sound_confidence = confidence;
            self.hearing.last_sound_at = Some(now);
            if let Some(cue) = self.cues.get(&Cue::key(CueKind::Sound, name)).cloned() {
                self.trigger(&Cue::key(CueKind::Sound, name), cue.behaviour);
            }
        } else if level > 0.35 && self.animation.is_none() {
            // Loud and unknown: startle, once in a while.
            self.trigger("sound:*loud", Behaviour::Startle);
        }
    }

    fn trigger(&mut self, key: &str, behaviour: Behaviour) {
        let cooldown = CONTROL_HZ as u64 * 2;
        if let Some(last) = self.last_trigger.get(key) {
            if self.tick.saturating_sub(*last) < cooldown {
                return;
            }
        }
        self.last_trigger.insert(key.to_string(), self.tick);
        self.perform(behaviour);
    }

    fn effective_target(&self) -> CompanionPose {
        let mut t = match (self.mode, self.mirror_target) {
            (CompanionMode::Interactive, Some(m)) => m,
            _ => self.target,
        };
        if self.mode == CompanionMode::Interactive && self.attention.tracking {
            // Look where the picture moves. Pan follows x directly (the companion
            // faces the camera, so motion on the viewer's right is on its left).
            t.head_pan = (-self.attention.x * 0.8).clamp(-1.0, 1.0);
            t.head_tilt = (self.attention.y * 0.6).clamp(-1.0, 1.0);
        }
        t
    }

    /// One control step: animations, idle motion and the approach to the target.
    pub fn tick(&mut self) {
        self.tick += 1;
        let t = self.tick as f32 / CONTROL_HZ as f32;
        let mut target = self.effective_target();
        // Idle breathing.
        target.lean += (t * 1.5).sin() * 0.03;

        let mut max_step = 0.05;
        if let Some(anim) = &self.animation {
            let elapsed = (self.tick - anim.started) as f32;
            if elapsed >= anim.duration as f32 {
                self.animation = None;
            } else {
                let p = elapsed / anim.duration as f32; // 0 to 1
                let env = (p * std::f32::consts::PI).sin(); // ease in and out
                let w = elapsed / CONTROL_HZ as f32; // seconds
                max_step = 0.12;
                match anim.behaviour {
                    Behaviour::Nod => target.head_tilt = (w * 12.0).sin() * 0.7 * env,
                    Behaviour::Shake => target.head_pan = (w * 12.0).sin() * 0.8 * env,
                    Behaviour::WaveLeft => {
                        target.left_arm = 0.9;
                        target.head_pan = 0.3 * env;
                        target.lean = (w * 14.0).sin() * 0.15 * env;
                    }
                    Behaviour::WaveRight => {
                        target.right_arm = 0.9;
                        target.head_pan = -0.3 * env;
                        target.lean = (w * 14.0).sin() * 0.15 * env;
                    }
                    Behaviour::Cheer => {
                        target.left_arm = 1.0;
                        target.right_arm = 1.0;
                        target.head_tilt = 0.5 * env;
                        target.mood = 0.9;
                    }
                    Behaviour::Dance => {
                        target.left_arm = (w * 8.0).sin() * 0.8;
                        target.right_arm = -(w * 8.0).sin() * 0.8;
                        target.head_pan = (w * 4.0).sin() * 0.6 * env;
                        target.lean = (w * 8.0).cos() * 0.3 * env;
                        target.mood = 1.0;
                    }
                    Behaviour::Startle => {
                        target.lean = -0.8 * env;
                        target.left_arm = 0.6 * env;
                        target.right_arm = 0.6 * env;
                        target.head_tilt = 0.4 * env;
                        target.mood = 0.8;
                    }
                    Behaviour::Sleep => {
                        target.head_tilt = -0.7;
                        target.lean = 0.3;
                        target.left_arm = -1.0;
                        target.right_arm = -1.0;
                        target.mood = 0.05;
                        max_step = 0.02;
                    }
                    Behaviour::Acknowledge => {
                        // Two quick nods, arms up a little, light flash.
                        target.head_tilt = (w * 16.0).sin() * 0.5 * env;
                        target.left_arm += 0.5 * env;
                        target.right_arm += 0.5 * env;
                        target.mood = 0.95;
                    }
                    Behaviour::Pose { .. } | Behaviour::Mood { .. } => {}
                }
            }
        }
        self.pose = self.pose.approach(&target.clamped(), max_step);
    }
}

/// Softmax weights over `(name, score)` candidates. Returns the weights for display
/// (sorted, descending) and, when the best score clears [`MIRROR_FLOOR`], one weight
/// per candidate in input order (summing to 1).
pub fn blend_weights(candidates: &[(String, f32)]) -> (Vec<MirrorWeight>, Option<Vec<f32>>) {
    let Some(best) = candidates.iter().map(|c| c.1).fold(None, |m: Option<f32>, s| Some(m.map_or(s, |m| m.max(s))))
    else {
        return (Vec::new(), None);
    };
    if best < MIRROR_FLOOR {
        return (
            candidates.iter().map(|(n, s)| MirrorWeight { name: n.clone(), weight: 0.0, score: *s }).collect(),
            None,
        );
    }
    let mut weights: Vec<f32> = candidates
        .iter()
        .map(|(_, s)| if best - s > MIRROR_WINDOW { 0.0 } else { ((s - best) / MIRROR_TEMPERATURE).exp() })
        .collect();
    let total: f32 = weights.iter().sum();
    if total <= 0.0 {
        return (Vec::new(), None);
    }
    for w in weights.iter_mut() {
        *w /= total;
    }
    let mut out: Vec<MirrorWeight> = candidates
        .iter()
        .zip(weights.iter())
        .map(|((n, s), w)| MirrorWeight { name: n.clone(), weight: *w, score: *s })
        .collect();
    out.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));
    (out, Some(weights))
}

/// Softmax blend of taught poses by score. Returns the weights (sorted, descending)
/// and the blended pose, or `None` when nothing scores above the floor.
pub fn blend_poses(candidates: &[(String, f32, CompanionPose)]) -> (Vec<MirrorWeight>, Option<CompanionPose>) {
    let scored: Vec<(String, f32)> = candidates.iter().map(|c| (c.0.clone(), c.1)).collect();
    let (out, weights) = blend_weights(&scored);
    let Some(weights) = weights else { return (out, None) };
    let mut acc = [0.0f32; POSE_DIM];
    for (c, w) in candidates.iter().zip(weights.iter()) {
        let a = c.2.as_array();
        for (k, v) in acc.iter_mut().enumerate() {
            *v += a[k] * w;
        }
    }
    (out, Some(CompanionPose::from_array(acc)))
}

/// Motion centroid between two grayscale thumbnails of the same size
/// (`width * height` values). Returns `(x, y, motion)` with x, y in `[-1, 1]`
/// (y up) and `motion` the fraction of pixels that changed.
pub fn motion_centroid(prev: &[u8], cur: &[u8], width: usize, height: usize) -> (f32, f32, f32) {
    if prev.len() != cur.len() || prev.len() != width * height || width == 0 || height == 0 {
        return (0.0, 0.0, 0.0);
    }
    let mut sx = 0.0f32;
    let mut sy = 0.0f32;
    let mut n = 0usize;
    for (i, (a, b)) in prev.iter().zip(cur.iter()).enumerate() {
        if a.abs_diff(*b) > 28 {
            sx += (i % width) as f32;
            sy += (i / width) as f32;
            n += 1;
        }
    }
    if n == 0 {
        return (0.0, 0.0, 0.0);
    }
    let cx = sx / n as f32 / (width - 1).max(1) as f32 * 2.0 - 1.0;
    let cy = 1.0 - sy / n as f32 / (height - 1).max(1) as f32 * 2.0;
    (cx, cy, n as f32 / (width * height) as f32)
}

/// Shared handle: core state plus the telemetry broadcast.
#[derive(Clone)]
pub struct CompanionHandle {
    pub core: Arc<Mutex<CompanionCore>>,
    pub telemetry_tx: watch::Sender<CompanionTelemetry>,
}

impl Default for CompanionHandle {
    fn default() -> Self {
        Self::new()
    }
}

impl CompanionHandle {
    pub fn new() -> Self {
        let core = CompanionCore::new();
        let (telemetry_tx, _) = watch::channel(core.telemetry());
        Self { core: Arc::new(Mutex::new(core)), telemetry_tx }
    }

    pub fn subscribe(&self) -> watch::Receiver<CompanionTelemetry> {
        self.telemetry_tx.subscribe()
    }
}

/// Tick the companion at [`CONTROL_HZ`] and publish telemetry.
pub fn spawn_control_loop(handle: CompanionHandle) {
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs_f32(1.0 / CONTROL_HZ as f32));
        interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            interval.tick().await;
            let telemetry = {
                let mut core = handle.core.lock().await;
                core.tick();
                core.telemetry()
            };
            let _ = handle.telemetry_tx.send(telemetry);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn score(name: &str, s: f32) -> GestureScore {
        GestureScore {
            name: name.into(),
            is_neutral: false,
            raw_cosine: s,
            contrastive: None,
            combined: s,
            sample_count: 1,
        }
    }

    #[test]
    fn mirror_blends_between_taught_poses() {
        let up = CompanionPose { left_arm: 1.0, ..Default::default() };
        let down = CompanionPose { left_arm: -1.0, ..Default::default() };
        let c = vec![("up".to_string(), 0.8, up), ("down".to_string(), 0.8, down)];
        let (w, pose) = blend_poses(&c);
        assert_eq!(w.len(), 2);
        assert!((pose.unwrap().left_arm).abs() < 1e-5, "equal scores blend to the middle");
        let c = vec![("up".to_string(), 0.9, up), ("down".to_string(), 0.5, down)];
        let (w, pose) = blend_poses(&c);
        assert_eq!(w[0].name, "up");
        assert!(pose.unwrap().left_arm > 0.99, "a clear winner dominates");
        let c = vec![("up".to_string(), 0.2, up)];
        assert!(blend_poses(&c).1.is_none(), "below the floor nothing moves");
    }

    #[test]
    fn cues_trigger_behaviours_and_poses_mirror() {
        let mut core = CompanionCore::new();
        core.set_mode(CompanionMode::Interactive);
        core.set_cue(Cue { kind: CueKind::Gesture, name: "hi".into(), behaviour: Behaviour::WaveRight });
        core.set_cue(Cue {
            kind: CueKind::Gesture,
            name: "arms".into(),
            behaviour: Behaviour::Pose { pose: CompanionPose { left_arm: 1.0, right_arm: 1.0, ..Default::default() } },
        });
        core.observe_gestures(&[score("hi", 0.9)], Some("hi"), 0.9, 1.0);
        assert_eq!(core.telemetry().animation.as_deref(), Some("wave_right"));
        for _ in 0..(CONTROL_HZ * 3) {
            core.tick();
        }
        assert!(core.telemetry().animation.is_none());
        core.observe_gestures(&[score("arms", 0.9), score("hi", 0.3)], Some("arms"), 0.9, 2.0);
        for _ in 0..CONTROL_HZ * 2 {
            core.tick();
        }
        assert!(core.pose.left_arm > 0.9 && core.pose.right_arm > 0.9, "{:?}", core.pose);
        assert_eq!(core.seeing.mirror[0].name, "arms");
        // Sound cue with cooldown.
        core.set_cue(Cue { kind: CueKind::Sound, name: "clap".into(), behaviour: Behaviour::Nod });
        core.observe_sound(0.2, Some("clap"), 0.8, 3.0);
        assert_eq!(core.telemetry().animation.as_deref(), Some("nod"));
        // Manual mode ignores observations.
        core.set_mode(CompanionMode::Manual);
        core.animation = None;
        core.observe_sound(0.2, Some("clap"), 0.8, 9.0);
        assert!(core.telemetry().animation.is_none());
    }

    #[test]
    fn attention_follows_motion() {
        let w = 8;
        let h = 4;
        let prev = vec![0u8; w * h];
        let mut cur = prev.clone();
        cur[3 * w + 7] = 255; // bottom right
        let (x, y, m) = motion_centroid(&prev, &cur, w, h);
        assert!(x > 0.99 && y < -0.99 && m > 0.0);
        let mut core = CompanionCore::new();
        core.set_mode(CompanionMode::Interactive);
        for _ in 0..8 {
            core.observe_motion(x, y, 0.05);
        }
        for _ in 0..CONTROL_HZ {
            core.tick();
        }
        assert!(core.pose.head_pan < -0.3, "{:?}", core.pose);
        assert!(core.pose.head_tilt < -0.2);
        let mem = serde_json::to_string(&core.memory()).unwrap();
        let back: CompanionMemory = serde_json::from_str(&mem).unwrap();
        assert!(back.cues.is_empty());
    }
}
