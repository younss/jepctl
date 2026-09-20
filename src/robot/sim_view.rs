//! Simulated camera for the virtual backend: the twin is drawn into the real camera
//! frame so that what the agent observes depends on its own joints. Without this a
//! virtual arm has no effect on the camera and there is nothing to learn from.
//!
//! The forward kinematics and link lengths mirror the WebGL twin in `app.js`; the
//! virtual camera is fixed in front and slightly above the base.

use image::{Rgb, RgbImage};

use crate::robot::DOF;

/// Link lengths in metres (same as the WebGL twin).
const BASE: f32 = 0.06;
const SHOULDER: f32 = 0.24;
const ELBOW: f32 = 0.22;
const WRIST_PITCH: f32 = 0.06;
const WRIST_ROLL: f32 = 0.05;
const JAW: f32 = 0.05;

type Vec3 = [f32; 3];
type Mat4 = [[f32; 4]; 4];

fn identity() -> Mat4 {
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

fn mul(a: &Mat4, b: &Mat4) -> Mat4 {
    let mut o = [[0.0f32; 4]; 4];
    for (i, row) in o.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            *cell = (0..4).map(|k| a[i][k] * b[k][j]).sum();
        }
    }
    o
}

fn translate(x: f32, y: f32, z: f32) -> Mat4 {
    let mut m = identity();
    m[0][3] = x;
    m[1][3] = y;
    m[2][3] = z;
    m
}

fn rot_x(a: f32) -> Mat4 {
    let (c, s) = (a.cos(), a.sin());
    let mut m = identity();
    m[1][1] = c;
    m[1][2] = -s;
    m[2][1] = s;
    m[2][2] = c;
    m
}

fn rot_y(a: f32) -> Mat4 {
    let (c, s) = (a.cos(), a.sin());
    let mut m = identity();
    m[0][0] = c;
    m[0][2] = s;
    m[2][0] = -s;
    m[2][2] = c;
    m
}

fn rot_z(a: f32) -> Mat4 {
    let (c, s) = (a.cos(), a.sin());
    let mut m = identity();
    m[0][0] = c;
    m[0][1] = -s;
    m[1][0] = s;
    m[1][1] = c;
    m
}

fn apply(m: &Mat4, p: Vec3) -> Vec3 {
    let mut o = [0.0f32; 3];
    for (i, v) in o.iter_mut().enumerate() {
        *v = m[i][0] * p[0] + m[i][1] * p[1] + m[i][2] * p[2] + m[i][3];
    }
    o
}

/// World-space key points of the arm: base, shoulder, elbow, wrist, roll end, two jaw tips.
pub fn arm_points(joints: &[f32; DOF], gripper: f32) -> Vec<Vec3> {
    let mut t = mul(&translate(0.0, 0.02, 0.0), &rot_y(joints[0]));
    let base_top = apply(&t, [0.0, BASE, 0.0]);
    t = mul(&t, &mul(&translate(0.0, BASE, 0.0), &rot_z(joints[1])));
    let shoulder_end = apply(&t, [0.0, SHOULDER, 0.0]);
    t = mul(&t, &mul(&translate(0.0, SHOULDER, 0.0), &rot_z(joints[2])));
    let elbow_end = apply(&t, [0.0, ELBOW, 0.0]);
    t = mul(&t, &mul(&translate(0.0, ELBOW, 0.0), &rot_z(joints[3])));
    let wrist_end = apply(&t, [0.0, WRIST_PITCH, 0.0]);
    t = mul(&t, &mul(&translate(0.0, WRIST_PITCH, 0.0), &rot_x(joints[4])));
    t = mul(&t, &rot_y(joints[5]));
    let roll_end = apply(&t, [0.0, WRIST_ROLL, 0.0]);
    let gap = 0.006 + 0.03 * gripper.clamp(0.0, 1.0);
    let jaw_base = mul(&t, &translate(0.0, WRIST_ROLL, 0.0));
    let jaw_l = apply(&jaw_base, [gap / 2.0 + 0.006, JAW, 0.0]);
    let jaw_r = apply(&jaw_base, [-gap / 2.0 - 0.006, JAW, 0.0]);
    vec![[0.0, 0.02, 0.0], base_top, shoulder_end, elbow_end, wrist_end, roll_end, jaw_l, jaw_r]
}

/// Fixed pinhole camera in front of the arm, slightly above, looking at the workspace.
fn project(p: Vec3, width: u32, height: u32) -> Option<(i32, i32)> {
    // Three quarter top view at (0.45, 0.62, 0.45) looking at (0, 0.2, 0): base rotation
    // sweeps across the image instead of into its depth, so every joint is observable.
    let eye = [0.45f32, 0.62, 0.45];
    let target = [0.0f32, 0.2, 0.0];
    let fwd = norm([target[0] - eye[0], target[1] - eye[1], target[2] - eye[2]]);
    let right = norm(cross(fwd, [0.0, 1.0, 0.0]));
    let up = cross(right, fwd);
    let d = [p[0] - eye[0], p[1] - eye[1], p[2] - eye[2]];
    let z = dot(d, fwd);
    if z <= 0.05 {
        return None;
    }
    let f = 1.6; // focal length relative to half height
    let x = dot(d, right) / z * f;
    let y = dot(d, up) / z * f;
    let half_h = height as f32 / 2.0;
    let cx = width as f32 / 2.0 + x * half_h;
    let cy = half_h - y * half_h;
    Some((cx.round() as i32, cy.round() as i32))
}

fn dot(a: Vec3, b: Vec3) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn norm(v: Vec3) -> Vec3 {
    let l = dot(v, v).sqrt().max(1e-6);
    [v[0] / l, v[1] / l, v[2] / l]
}

fn draw_disc(img: &mut RgbImage, cx: i32, cy: i32, r: i32, color: Rgb<u8>) {
    let (w, h) = (img.width() as i32, img.height() as i32);
    for y in (cy - r).max(0)..=(cy + r).min(h - 1) {
        for x in (cx - r).max(0)..=(cx + r).min(w - 1) {
            if (x - cx) * (x - cx) + (y - cy) * (y - cy) <= r * r {
                img.put_pixel(x as u32, y as u32, color);
            }
        }
    }
}

fn draw_thick_line(img: &mut RgbImage, a: (i32, i32), b: (i32, i32), thickness: i32, color: Rgb<u8>) {
    let steps = (a.0 - b.0).abs().max((a.1 - b.1).abs()).max(1);
    for i in 0..=steps {
        let t = i as f32 / steps as f32;
        let x = a.0 as f32 + (b.0 - a.0) as f32 * t;
        let y = a.1 as f32 + (b.1 - a.1) as f32 * t;
        draw_disc(img, x.round() as i32, y.round() as i32, thickness, color);
    }
}

/// Draw the arm over `frame` (modified in place).
pub fn overlay_arm(frame: &mut RgbImage, joints: &[f32; DOF], gripper: f32) {
    let (w, h) = (frame.width(), frame.height());
    let pts: Vec<Option<(i32, i32)>> = arm_points(joints, gripper).into_iter().map(|p| project(p, w, h)).collect();
    let thick = ((h as f32) * 0.012).round().max(2.0) as i32;
    let segments: [(usize, usize, Rgb<u8>, i32); 7] = [
        (0, 1, Rgb([70, 78, 92]), thick + 2),
        (1, 2, Rgb([150, 160, 178]), thick + 1),
        (2, 3, Rgb([140, 150, 168]), thick),
        (3, 4, Rgb([130, 140, 158]), thick - 1),
        (4, 5, Rgb([120, 130, 148]), thick - 1),
        (5, 6, Rgb([220, 224, 232]), (thick / 2).max(1)),
        (5, 7, Rgb([220, 224, 232]), (thick / 2).max(1)),
    ];
    // Base plate
    if let Some((bx, by)) = pts[0] {
        draw_disc(frame, bx, by, thick * 3, Rgb([60, 66, 80]));
    }
    for (a, b, color, t) in segments {
        if let (Some(pa), Some(pb)) = (pts[a], pts[b]) {
            draw_thick_line(frame, pa, pb, t, color);
        }
    }
    for (i, p) in pts.iter().enumerate().take(6) {
        if let Some((x, y)) = p {
            draw_disc(frame, *x, *y, thick, if i == 5 { Rgb([74, 222, 128]) } else { Rgb([200, 205, 215]) });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn overlay_changes_with_joints_and_stays_in_frame() {
        let blank = RgbImage::from_pixel(320, 240, Rgb([10, 10, 10]));
        let mut a = blank.clone();
        overlay_arm(&mut a, &[0.0; DOF], 0.5);
        let mut b = blank.clone();
        overlay_arm(&mut b, &[1.2, 0.8, -0.9, 0.3, 0.0, 0.0], 1.0);
        let painted = |img: &RgbImage| img.pixels().filter(|p| p[0] > 30).count();
        assert!(painted(&a) > 500, "arm should be visible: {}", painted(&a));
        assert!(painted(&b) > 500);
        let differing = a.pixels().zip(b.pixels()).filter(|(p, q)| p != q).count();
        assert!(differing > 500, "different joints must change the picture: {differing}");
        // Extreme joints never panic (projection clamps and clips).
        let mut c = blank;
        overlay_arm(&mut c, &[2.6, 1.8, -2.0, 1.8, 2.6, 3.1], 0.0);
    }
}
