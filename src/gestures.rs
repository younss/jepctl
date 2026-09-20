//! Few-shot gesture registry and explainable latent matching.
//!
//! A gesture is a *prototype*: the L2-normalised mean of one or more reference
//! embeddings captured with a given model. Matching an input embedding against the
//! registry produces a fully inspectable [`GestureMatchResult`]: raw cosine,
//! contrastive score, blended score and runner-up margin for every gesture, plus an
//! optional per-patch difference map that shows *where* the input diverges from the
//! best prototype.
//!
//! Why contrastive scoring: a mean-pooled ViT embedding of a webcam frame is
//! dominated by what never changes between gestures (background, face, torso,
//! lighting). Raw cosine therefore saturates near 1.0 for every registered pose.
//! Removing the centroid of all prototypes projects both the input and the
//! prototypes into the subspace that actually differs between gestures.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::types::{dot_product, normalize_l2, JepaError};

/// Minimum score lead over the runner-up required to declare a detection.
pub const DEFAULT_MARGIN: f32 = 0.04;

/// Default blended-score threshold.
pub const DEFAULT_THRESHOLD: f32 = 0.70;

/// Raw cosine below which a candidate is considered unrelated and the contrastive
/// term is not trusted (the residual direction of an unrelated frame is noise).
const RAW_COSINE_FLOOR: f32 = 0.40;

/// Blend weights between raw cosine and contrastive score.
const RAW_WEIGHT: f32 = 0.35;
const CONTRAST_WEIGHT: f32 = 0.65;

/// Registered reference gesture (few-shot prototype).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RegisteredGesture {
    pub name: String,
    /// Model the samples were embedded with. Embeddings from different models are
    /// not comparable, so matching only considers gestures of the active model.
    pub model_name: String,
    pub dimension: usize,
    /// Individual L2-normalised reference embeddings.
    pub samples: Vec<Vec<f32>>,
    /// L2-normalised mean of `samples`.
    pub prototype: Vec<f32>,
    /// Mean of per-patch token vectors across samples (for spatial diff maps).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub patch_prototype: Option<Vec<Vec<f32>>>,
    /// A neutral gesture ("rest pose") competes in scoring but is never reported as
    /// a detection. It absorbs frames where no intentional gesture is shown.
    #[serde(default)]
    pub is_neutral: bool,
    pub created_at: u64,
    #[serde(default)]
    pub updated_at: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thumbnail: Option<String>,
}

impl RegisteredGesture {
    pub fn new(name: String, model_name: String, is_neutral: bool, now: u64) -> Self {
        Self {
            name,
            model_name,
            dimension: 0,
            samples: Vec::new(),
            prototype: Vec::new(),
            patch_prototype: None,
            is_neutral,
            created_at: now,
            updated_at: now,
            thumbnail: None,
        }
    }

    /// Append a reference sample and recompute the prototype.
    pub fn add_sample(&mut self, embedding: &[f32], patches: Option<&[Vec<f32>]>, now: u64) -> Result<(), JepaError> {
        if embedding.is_empty() {
            return Err(JepaError::InvalidPayload("Empty embedding".into()));
        }
        if self.dimension != 0 && self.dimension != embedding.len() {
            return Err(JepaError::InvalidPayload(format!(
                "Embedding dimension {} does not match gesture dimension {}",
                embedding.len(),
                self.dimension
            )));
        }
        self.dimension = embedding.len();
        self.samples.push(normalize_l2(embedding));
        self.prototype = mean_normalized(&self.samples);
        self.updated_at = now;

        if let Some(p) = patches {
            let n = self.samples.len() as f32;
            match self.patch_prototype.as_mut() {
                Some(existing) if existing.len() == p.len() => {
                    // Running mean: new = old + (x - old) / n
                    for (row, new_row) in existing.iter_mut().zip(p.iter()) {
                        for (v, x) in row.iter_mut().zip(new_row.iter()) {
                            *v += (x - *v) / n;
                        }
                    }
                }
                _ => self.patch_prototype = Some(p.to_vec()),
            }
        }
        Ok(())
    }
}

/// L2-normalised mean of a set of vectors (all assumed the same length).
fn mean_normalized(vectors: &[Vec<f32>]) -> Vec<f32> {
    let dim = vectors.first().map(|v| v.len()).unwrap_or(0);
    let mut mean = vec![0.0f32; dim];
    for v in vectors {
        for (m, x) in mean.iter_mut().zip(v.iter()) {
            *m += x;
        }
    }
    let n = vectors.len().max(1) as f32;
    for m in mean.iter_mut() {
        *m /= n;
    }
    normalize_l2(&mean)
}

/// Score breakdown for one registered gesture.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct GestureScore {
    pub name: String,
    pub is_neutral: bool,
    /// Cosine similarity between the input and the prototype (what a naive matcher uses).
    pub raw_cosine: f32,
    /// Cosine similarity in the centroid-removed subspace. `None` with a single gesture.
    pub contrastive: Option<f32>,
    /// Final blended score the decision is based on.
    pub combined: f32,
    pub sample_count: usize,
}

/// Full, inspectable result of matching one input against the registry.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GestureMatchResult {
    /// Name of the detected gesture, if any.
    pub matched: Option<String>,
    /// Best non-neutral blended score.
    pub confidence: f32,
    pub detected: bool,
    /// Lead of the best candidate over the runner-up (including neutral).
    pub margin: f32,
    pub threshold: f32,
    pub margin_required: f32,
    /// `"cosine"` (single gesture) or `"contrastive"` (two or more).
    pub method: String,
    /// Human-readable explanation of the decision.
    pub reason: String,
    /// Every gesture, sorted by `combined` descending.
    pub scores: Vec<GestureScore>,
    /// Per-patch dissimilarity (0 = identical, higher = more different) between the
    /// input and the best candidate's patch prototype, row-major over the ViT grid.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub patch_diff: Option<Vec<f32>>,
    /// Side length of the patch grid (`patch_diff.len() == grid_size²`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub grid_size: Option<usize>,
}

impl GestureMatchResult {
    pub fn empty(threshold: f32, reason: &str) -> Self {
        Self {
            matched: None,
            confidence: 0.0,
            detected: false,
            margin: 0.0,
            threshold,
            margin_required: DEFAULT_MARGIN,
            method: "none".to_string(),
            reason: reason.to_string(),
            scores: Vec::new(),
            patch_diff: None,
            grid_size: None,
        }
    }
}

/// Match an input embedding against a set of gestures.
///
/// `input_patches` (optional) are the input's per-patch tokens used to compute the
/// spatial difference map against the best candidate.
pub fn match_gestures(
    input_embedding: &[f32],
    input_patches: Option<&[Vec<f32>]>,
    gestures: &[&RegisteredGesture],
    threshold: f32,
    margin_required: f32,
) -> GestureMatchResult {
    let input = normalize_l2(input_embedding);
    let candidates: Vec<&RegisteredGesture> =
        gestures.iter().copied().filter(|g| g.dimension == input.len() && !g.prototype.is_empty()).collect();

    if candidates.is_empty() {
        return GestureMatchResult::empty(threshold, "No registered gesture matches the input dimension");
    }

    let dim = input.len();
    let method = if candidates.len() >= 2 { "contrastive" } else { "cosine" };

    // Centroid of prototypes = the shared component (background, person, lighting).
    let centroid: Option<Vec<f32>> = (candidates.len() >= 2).then(|| {
        let mut c = vec![0.0f32; dim];
        for g in &candidates {
            for (ci, p) in c.iter_mut().zip(g.prototype.iter()) {
                *ci += p;
            }
        }
        let k = candidates.len() as f32;
        c.iter_mut().for_each(|ci| *ci /= k);
        c
    });

    let input_residual =
        centroid.as_ref().map(|c| normalize_l2(&input.iter().zip(c.iter()).map(|(a, b)| a - b).collect::<Vec<_>>()));

    let mut scores: Vec<GestureScore> = candidates
        .iter()
        .map(|g| {
            let raw = dot_product(&input, &g.prototype).clamp(0.0, 1.0);
            let contrastive = match (&centroid, &input_residual) {
                (Some(c), Some(res)) => {
                    let g_res = normalize_l2(&g.prototype.iter().zip(c.iter()).map(|(a, b)| a - b).collect::<Vec<_>>());
                    Some(dot_product(res, &g_res).clamp(-1.0, 1.0))
                }
                _ => None,
            };
            let combined = match contrastive {
                Some(ct) if raw >= RAW_COSINE_FLOOR => {
                    (RAW_WEIGHT * raw + CONTRAST_WEIGHT * ct.max(0.0)).clamp(0.0, 1.0)
                }
                Some(_) => raw * 0.5,
                None => raw,
            };
            GestureScore {
                name: g.name.clone(),
                is_neutral: g.is_neutral,
                raw_cosine: raw,
                contrastive,
                combined,
                sample_count: g.samples.len(),
            }
        })
        .collect();

    scores.sort_by(|a, b| b.combined.partial_cmp(&a.combined).unwrap_or(std::cmp::Ordering::Equal));

    let best = &scores[0];
    let second = scores.get(1).map(|s| s.combined).unwrap_or(0.0);
    let margin = best.combined - second;

    let (detected, matched, reason) = if best.is_neutral {
        (false, None, format!("Closest to neutral pose '{}' ({:.2})", best.name, best.combined))
    } else if best.combined < threshold {
        (false, None, format!("Best '{}' scored {:.2} < threshold {:.2}", best.name, best.combined, threshold))
    } else if margin < margin_required {
        (
            false,
            None,
            format!(
                "'{}' ({:.2}) is too close to runner-up ({:.2}); margin {:.3} < {:.3}",
                best.name, best.combined, second, margin, margin_required
            ),
        )
    } else {
        (
            true,
            Some(best.name.clone()),
            format!("'{}' scored {:.2} with margin {:.3}", best.name, best.combined, margin),
        )
    };

    let confidence = scores.iter().filter(|s| !s.is_neutral).map(|s| s.combined).fold(0.0, f32::max);

    // Spatial explanation against the best candidate (neutral or not: it is still
    // the most useful reference to show "what differs").
    let best_gesture = candidates.iter().find(|g| g.name == best.name);
    let (patch_diff, grid_size) = match (input_patches, best_gesture.and_then(|g| g.patch_prototype.as_ref())) {
        (Some(inp), Some(proto)) if inp.len() == proto.len() && !inp.is_empty() => {
            let diff: Vec<f32> = inp
                .iter()
                .zip(proto.iter())
                .map(|(a, b)| (1.0 - dot_product(&normalize_l2(a), &normalize_l2(b))).clamp(0.0, 2.0))
                .collect();
            let grid = (diff.len() as f64).sqrt().round() as usize;
            (Some(diff), Some(grid))
        }
        _ => (None, None),
    };

    GestureMatchResult {
        matched,
        confidence,
        detected,
        margin,
        threshold,
        margin_required,
        method: method.to_string(),
        reason,
        scores,
        patch_diff,
        grid_size,
    }
}

/// Portable set of gestures: what `GET /api/gestures/export` returns and
/// `POST /api/gestures/import` / `jepctl gestures import` accept. Train prototypes on one
/// machine, deploy them on many.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GestureBundle {
    /// Schema version of this bundle.
    pub version: u32,
    pub exported_at: u64,
    /// Decision parameters the exporter was tuned with (pass them to `/api/gestures/match`).
    pub threshold: f32,
    pub margin: f32,
    /// Camera region of interest the prototypes were captured with, if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub roi: Option<crate::types::Roi>,
    pub gestures: Vec<RegisteredGesture>,
}

pub const BUNDLE_VERSION: u32 = 1;

/// Result of importing a bundle.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ImportReport {
    pub imported: usize,
    /// Gestures of the same model removed beforehand (`replace = true`).
    pub removed: usize,
    pub models: Vec<String>,
}

/// Persistent gesture registry (`~/.jepctl/gestures.json`).
#[derive(Default, Serialize, Deserialize)]
pub struct GestureStore {
    #[serde(default)]
    pub gestures: HashMap<String, RegisteredGesture>,
    #[serde(skip)]
    path: Option<PathBuf>,
}

impl GestureStore {
    /// Load the registry from disk, or start empty if the file is missing/corrupt.
    pub fn load(path: &Path) -> Self {
        let mut store = match std::fs::read_to_string(path) {
            Ok(text) => serde_json::from_str::<GestureStore>(&text).unwrap_or_else(|e| {
                tracing::warn!("Ignoring unreadable gesture registry {}: {}", path.display(), e);
                GestureStore::default()
            }),
            Err(_) => GestureStore::default(),
        };
        store.path = Some(path.to_path_buf());
        store
    }

    /// In-memory registry that is never written to disk.
    pub fn ephemeral() -> Self {
        Self::default()
    }

    /// Gestures registered for a given model.
    pub fn for_model<'a>(&'a self, model_name: &str) -> Vec<&'a RegisteredGesture> {
        let mut v: Vec<&RegisteredGesture> = self.gestures.values().filter(|g| g.model_name == model_name).collect();
        v.sort_by_key(|g| g.created_at);
        v
    }

    pub fn get_mut_or_insert(
        &mut self,
        name: &str,
        model_name: &str,
        is_neutral: bool,
        now: u64,
    ) -> &mut RegisteredGesture {
        let entry = self
            .gestures
            .entry(name.to_string())
            .or_insert_with(|| RegisteredGesture::new(name.to_string(), model_name.to_string(), is_neutral, now));
        // Re-registering under a different model starts a fresh prototype.
        if entry.model_name != model_name {
            *entry = RegisteredGesture::new(name.to_string(), model_name.to_string(), is_neutral, now);
        }
        entry.is_neutral = is_neutral;
        entry
    }

    pub fn remove(&mut self, name: &str) -> Option<RegisteredGesture> {
        self.gestures.remove(name)
    }

    /// Export gestures (of one model, or all) as a portable bundle.
    pub fn export(
        &self,
        model: Option<&str>,
        threshold: f32,
        margin: f32,
        with_thumbnails: bool,
        now: u64,
    ) -> GestureBundle {
        let mut gestures: Vec<RegisteredGesture> = self
            .gestures
            .values()
            .filter(|g| model.is_none_or(|m| g.model_name == m))
            .cloned()
            .map(|mut g| {
                if !with_thumbnails {
                    g.thumbnail = None;
                }
                g
            })
            .collect();
        gestures.sort_by_key(|g| g.created_at);
        GestureBundle { version: BUNDLE_VERSION, exported_at: now, threshold, margin, roi: None, gestures }
    }

    /// Import a bundle. With `replace`, gestures of every model present in the bundle are
    /// removed first; otherwise same-named gestures are overwritten and others kept.
    pub fn import(&mut self, bundle: GestureBundle, replace: bool) -> Result<ImportReport, JepaError> {
        if bundle.version != BUNDLE_VERSION {
            return Err(JepaError::InvalidPayload(format!(
                "Unsupported bundle version {} (expected {})",
                bundle.version, BUNDLE_VERSION
            )));
        }
        for g in &bundle.gestures {
            if g.name.trim().is_empty() || g.model_name.trim().is_empty() {
                return Err(JepaError::InvalidPayload("Gesture name and model_name must not be empty".into()));
            }
            if g.prototype.len() != g.dimension || g.samples.iter().any(|s| s.len() != g.dimension) {
                return Err(JepaError::InvalidPayload(format!("Gesture '{}' has inconsistent dimensions", g.name)));
            }
        }

        let mut models: Vec<String> = bundle.gestures.iter().map(|g| g.model_name.clone()).collect();
        models.sort();
        models.dedup();

        let mut removed = 0;
        if replace {
            let before = self.gestures.len();
            self.gestures.retain(|_, g| !models.contains(&g.model_name));
            removed = before - self.gestures.len();
        }
        let imported = bundle.gestures.len();
        for g in bundle.gestures {
            self.gestures.insert(g.name.clone(), g);
        }
        Ok(ImportReport { imported, removed, models })
    }

    /// Persist to disk (no-op for ephemeral stores).
    pub fn save(&self) -> Result<(), JepaError> {
        if let Some(path) = &self.path {
            let json = serde_json::to_string(self)?;
            let tmp = path.with_extension("json.tmp");
            std::fs::write(&tmp, json)?;
            std::fs::rename(&tmp, path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gesture(name: &str, samples: &[Vec<f32>], neutral: bool) -> RegisteredGesture {
        let mut g = RegisteredGesture::new(name.into(), "m".into(), neutral, 0);
        for s in samples {
            g.add_sample(s, None, 1).unwrap();
        }
        g
    }

    #[test]
    fn single_gesture_uses_raw_cosine() {
        let g = gesture("open", &[vec![1.0, 0.0, 0.0]], false);
        let r = match_gestures(&[0.9, 0.1, 0.0], None, &[&g], 0.7, DEFAULT_MARGIN);
        assert_eq!(r.method, "cosine");
        assert!(r.detected);
        assert_eq!(r.matched.as_deref(), Some("open"));
        assert!(r.scores[0].contrastive.is_none());
    }

    #[test]
    fn contrastive_separates_gestures_sharing_a_background() {
        let bg = vec![5.0, 5.0, 5.0, 5.0];
        let mk = |i: usize| {
            let mut v = bg.clone();
            v[i] += 2.0;
            v
        };
        let a = gesture("a", &[mk(0)], false);
        let b = gesture("b", &[mk(1)], false);
        let c = gesture("c", &[mk(2)], false);

        let mut input = vec![5.2, 4.8, 5.0, 5.1];
        input[0] += 1.9;
        let r = match_gestures(&input, None, &[&a, &b, &c], 0.7, DEFAULT_MARGIN);

        assert_eq!(r.method, "contrastive");
        assert!(r.detected, "{}", r.reason);
        assert_eq!(r.matched.as_deref(), Some("a"));
        // Raw cosine is useless here (everything ≈ 0.99) ...
        assert!(r.scores.iter().all(|s| s.raw_cosine > 0.95));
        // ... but the blended score has a large margin.
        assert!(r.scores[0].combined - r.scores[1].combined > 0.4);
    }

    #[test]
    fn neutral_pose_wins_but_is_never_reported() {
        let rest = gesture("rest", &[vec![1.0, 0.0, 0.0]], true);
        let fist = gesture("fist", &[vec![0.0, 1.0, 0.0]], false);
        let r = match_gestures(&[1.0, 0.05, 0.0], None, &[&rest, &fist], 0.5, DEFAULT_MARGIN);
        assert!(!r.detected);
        assert!(r.matched.is_none());
        assert!(r.reason.contains("neutral"));
        assert_eq!(r.scores[0].name, "rest");
    }

    #[test]
    fn margin_blocks_ambiguous_detection() {
        let a = gesture("a", &[vec![1.0, 0.0]], false);
        let b = gesture("b", &[vec![0.0, 1.0]], false);
        let r = match_gestures(&[1.0, 1.0], None, &[&a, &b], 0.1, DEFAULT_MARGIN);
        assert!(!r.detected);
        assert!(r.margin.abs() < 1e-5);
        assert!(r.reason.contains("margin"));
    }

    #[test]
    fn multi_sample_prototype_is_normalized_mean() {
        let g = gesture("g", &[vec![1.0, 0.0], vec![0.0, 1.0]], false);
        assert_eq!(g.samples.len(), 2);
        let n: f32 = g.prototype.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((n - 1.0).abs() < 1e-5);
        assert!((g.prototype[0] - g.prototype[1]).abs() < 1e-5);
    }

    #[test]
    fn dimension_mismatch_is_rejected_and_ignored() {
        let mut g = gesture("g", &[vec![1.0, 0.0]], false);
        assert!(g.add_sample(&[1.0, 0.0, 0.0], None, 2).is_err());
        let r = match_gestures(&[1.0, 0.0, 0.0], None, &[&g], 0.5, DEFAULT_MARGIN);
        assert!(r.scores.is_empty());
        assert!(!r.detected);
    }

    #[test]
    fn patch_diff_highlights_changed_patches() {
        let mut g = RegisteredGesture::new("g".into(), "m".into(), false, 0);
        let proto_patches = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0]];
        g.add_sample(&[1.0, 0.0], Some(&proto_patches), 1).unwrap();

        let input_patches = vec![vec![1.0, 0.0], vec![1.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
        let r = match_gestures(&[1.0, 0.0], Some(&input_patches), &[&g], 0.5, DEFAULT_MARGIN);
        let diff = r.patch_diff.expect("diff map");
        assert_eq!(r.grid_size, Some(2));
        assert!(diff[0] < 1e-5 && diff[3] > 0.99);
    }

    #[test]
    fn bundle_export_import_roundtrip_and_replace() {
        let mut a = GestureStore::ephemeral();
        a.get_mut_or_insert("open", "m1", false, 1).add_sample(&[1.0, 0.0], None, 1).unwrap();
        a.get_mut_or_insert("fist", "m1", false, 2).add_sample(&[0.0, 1.0], None, 2).unwrap();
        a.get_mut_or_insert("other", "m2", false, 3).add_sample(&[1.0], None, 3).unwrap();

        let bundle = a.export(Some("m1"), 0.7, 0.04, false, 99);
        assert_eq!(bundle.version, BUNDLE_VERSION);
        assert_eq!(bundle.gestures.len(), 2);
        let json = serde_json::to_string(&bundle).unwrap();

        let mut b = GestureStore::ephemeral();
        b.get_mut_or_insert("stale", "m1", false, 0).add_sample(&[0.5, 0.5], None, 0).unwrap();
        let report = b.import(serde_json::from_str(&json).unwrap(), true).unwrap();
        assert_eq!(report, ImportReport { imported: 2, removed: 1, models: vec!["m1".into()] });
        assert_eq!(b.for_model("m1").len(), 2);

        // Bad version and inconsistent dimensions are rejected.
        let mut bad = bundle.clone();
        bad.version = 42;
        assert!(b.import(bad, false).is_err());
        let mut bad = bundle.clone();
        bad.gestures[0].dimension = 7;
        assert!(b.import(bad, false).is_err());
    }

    #[test]
    fn store_roundtrip_and_model_filter() {
        let dir = std::env::temp_dir().join(format!("jepa_gestures_{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("gestures.json");

        let mut store = GestureStore::load(&path);
        store.get_mut_or_insert("a", "model-x", false, 1).add_sample(&[1.0, 0.0], None, 1).unwrap();
        store.get_mut_or_insert("b", "model-y", true, 2).add_sample(&[0.0, 1.0], None, 2).unwrap();
        store.save().unwrap();

        let reloaded = GestureStore::load(&path);
        assert_eq!(reloaded.gestures.len(), 2);
        assert_eq!(reloaded.for_model("model-x").len(), 1);
        assert!(reloaded.gestures["b"].is_neutral);

        let _ = std::fs::remove_dir_all(&dir);
    }
}
