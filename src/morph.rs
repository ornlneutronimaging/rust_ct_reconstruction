//! MorphSpotClean — a native port of MuhRec's morphological spot cleaning
//! (imagingsuite `ImagingAlgorithms::MorphSpotClean`, re-implemented from
//! the published method; grayscale reconstruction after Vincent 1993).
//!
//! Dark spots ("holes") and bright spots ("peaks") are detected by
//! comparing the image with a morphological reconstruction that fills only
//! true local extrema not connected to the image border — so real sample
//! structure is left alone. The detection threshold is chosen automatically
//! as a cumulative-histogram fraction of the difference image, and flagged
//! pixels are blended toward the reconstruction with a sigmoid transition
//! (no hard replacement edges).

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MorphDetect {
    /// Dark spots (dead pixels): local minima.
    Holes,
    /// Bright spots (gamma hits): local maxima.
    Peaks,
    /// Both dark and bright spots.
    Both,
}

impl MorphDetect {
    pub const ALL: [MorphDetect; 3] = [MorphDetect::Both, MorphDetect::Peaks, MorphDetect::Holes];

    pub fn label(self) -> &'static str {
        match self {
            MorphDetect::Holes => "dark spots (holes)",
            MorphDetect::Peaks => "bright spots (peaks)",
            MorphDetect::Both => "dark + bright spots",
        }
    }
}

/// The 8 neighbors of `(x, y)` inside a `w`×`h` grid.
fn neighbors(x: usize, y: usize, w: usize, h: usize) -> impl Iterator<Item = (usize, usize)> {
    let (x, y, w, h) = (x as isize, y as isize, w as isize, h as isize);
    [
        (-1, -1),
        (0, -1),
        (1, -1),
        (-1, 0),
        (1, 0),
        (-1, 1),
        (0, 1),
        (1, 1),
    ]
    .into_iter()
    .filter_map(move |(dx, dy)| {
        let (nx, ny) = (x + dx, y + dy);
        (0 <= nx && nx < w && 0 <= ny && ny < h).then_some((nx as usize, ny as usize))
    })
}

/// Grayscale reconstruction by erosion (Vincent's hybrid raster + FIFO
/// algorithm, 8-connectivity): the `marker` (≥ mask) shrinks toward the
/// `mask`; regional minima of the mask whose marker seed stays high (i.e.
/// not connected to the border seed) end up filled.
fn reconstruct_by_erosion(mask: &[f32], mut marker: Vec<f32>, w: usize, h: usize) -> Vec<f32> {
    use std::collections::VecDeque;
    let idx = |x: usize, y: usize| y * w + x;
    // Forward raster scan: propagate from the already-visited neighbors.
    for y in 0..h {
        for x in 0..w {
            let mut m = marker[idx(x, y)];
            if x > 0 {
                m = m.min(marker[idx(x - 1, y)]);
            }
            if y > 0 {
                if x > 0 {
                    m = m.min(marker[idx(x - 1, y - 1)]);
                }
                m = m.min(marker[idx(x, y - 1)]);
                if x + 1 < w {
                    m = m.min(marker[idx(x + 1, y - 1)]);
                }
            }
            marker[idx(x, y)] = m.max(mask[idx(x, y)]);
        }
    }
    // Backward raster scan, queueing pixels whose forward neighbors could
    // still be lowered.
    let mut queue = VecDeque::new();
    for y in (0..h).rev() {
        for x in (0..w).rev() {
            let mut m = marker[idx(x, y)];
            if x + 1 < w {
                m = m.min(marker[idx(x + 1, y)]);
            }
            if y + 1 < h {
                if x > 0 {
                    m = m.min(marker[idx(x - 1, y + 1)]);
                }
                m = m.min(marker[idx(x, y + 1)]);
                if x + 1 < w {
                    m = m.min(marker[idx(x + 1, y + 1)]);
                }
            }
            let v = m.max(mask[idx(x, y)]);
            marker[idx(x, y)] = v;
            let must_queue = neighbors(x, y, w, h).any(|(nx, ny)| {
                let q = idx(nx, ny);
                marker[q] > v && marker[q] > mask[q]
            });
            if must_queue {
                queue.push_back((x, y));
            }
        }
    }
    // FIFO propagation until stable.
    while let Some((x, y)) = queue.pop_front() {
        let v = marker[idx(x, y)];
        for (nx, ny) in neighbors(x, y, w, h) {
            let q = idx(nx, ny);
            if marker[q] > v && marker[q] > mask[q] {
                marker[q] = v.max(mask[q]);
                queue.push_back((nx, ny));
            }
        }
    }
    marker
}

/// Fill the regional minima ("holes", dark spots) of the image: grayscale
/// reconstruction by erosion from a marker that is +inf everywhere except
/// the border, where it equals the image. Result ≥ image; equal away from
/// interior minima.
pub fn fill_holes(values: &[f32], w: usize, h: usize) -> Vec<f32> {
    let mut marker = vec![f32::INFINITY; values.len()];
    for x in 0..w {
        marker[x] = values[x];
        marker[(h - 1) * w + x] = values[(h - 1) * w + x];
    }
    for y in 0..h {
        marker[y * w] = values[y * w];
        marker[y * w + w - 1] = values[y * w + w - 1];
    }
    reconstruct_by_erosion(values, marker, w, h)
}

/// Fill the regional maxima ("peaks", bright spots): the dual of
/// `fill_holes` via negation. Result ≤ image.
pub fn fill_peaks(values: &[f32], w: usize, h: usize) -> Vec<f32> {
    let negated: Vec<f32> = values.iter().map(|v| -v).collect();
    fill_holes(&negated, w, h).into_iter().map(|v| -v).collect()
}

/// Absolute detection threshold from a cumulative-histogram `fraction` of
/// the difference image (MuhRec's threshold-by-fraction, 1024 bins): the
/// difference value below which `fraction` of ALL pixels lie — so 0.95
/// touches at most 5% of the pixels. `None` when there is no positive
/// difference at all (a spot-free image).
fn fraction_threshold(diffs: &[f32], fraction: f32) -> Option<f32> {
    const BINS: usize = 1024;
    let max = diffs.iter().copied().fold(0.0f32, f32::max);
    if max <= 0.0 {
        return None;
    }
    let mut hist = [0usize; BINS];
    for d in diffs {
        let bin = ((d.max(0.0) / max) * BINS as f32) as usize;
        hist[bin.min(BINS - 1)] += 1;
    }
    let target = (diffs.len() as f64 * f64::from(fraction)).ceil() as usize;
    let mut cumulative = 0usize;
    for (i, count) in hist.iter().enumerate() {
        cumulative += count;
        if cumulative >= target {
            return Some((i + 1) as f32 / BINS as f32 * max);
        }
    }
    Some(max)
}

fn sigmoid(x: f32, level: f32, width: f32) -> f32 {
    1.0 / (1.0 + (-(x - level) / width).exp())
}

/// Blend the original toward the reconstruction: hard swap when `sigma` is
/// 0, otherwise a sigmoid transition of width `sigma·threshold`.
fn blend(value: f32, reference: f32, diff: f32, threshold: f32, sigma: f32) -> f32 {
    if sigma <= 0.0 {
        if diff > threshold { reference } else { value }
    } else {
        value + (reference - value) * sigmoid(diff, threshold, sigma * threshold)
    }
}

/// Clean one image; returns the cleaned values and the number of pixels
/// whose difference exceeded the threshold (i.e. mostly replaced).
pub fn morph_spot_clean(
    values: &[f32],
    w: usize,
    h: usize,
    detect: MorphDetect,
    threshold_fraction: f32,
    sigma: f32,
) -> (Vec<f32>, usize) {
    let noholes = matches!(detect, MorphDetect::Holes | MorphDetect::Both)
        .then(|| fill_holes(values, w, h));
    let nopeaks = matches!(detect, MorphDetect::Peaks | MorphDetect::Both)
        .then(|| fill_peaks(values, w, h));
    let th_holes = noholes.as_ref().and_then(|nh| {
        let diffs: Vec<f32> = nh.iter().zip(values).map(|(r, v)| r - v).collect();
        fraction_threshold(&diffs, threshold_fraction)
    });
    let th_peaks = nopeaks.as_ref().and_then(|np| {
        let diffs: Vec<f32> = values.iter().zip(np).map(|(v, r)| v - r).collect();
        fraction_threshold(&diffs, threshold_fraction)
    });

    let mut out = values.to_vec();
    let mut replaced = 0usize;
    for i in 0..values.len() {
        let v = values[i];
        let dh = noholes.as_ref().map(|nh| nh[i] - v);
        let dp = nopeaks.as_ref().map(|np| v - np[i]);
        // For "both", the larger difference tells which kind of spot the
        // pixel is (a dark spot has dh >> 0 and dp ~ 0, and vice versa).
        let hole_wins = match (dh, dp) {
            (Some(dh), Some(dp)) => dh >= dp,
            (Some(_), None) => true,
            _ => false,
        };
        if hole_wins {
            if let (Some(dh), Some(nh), Some(th)) = (dh, noholes.as_ref(), th_holes) {
                out[i] = blend(v, nh[i], dh, th, sigma);
                if dh > th {
                    replaced += 1;
                }
            }
        } else if let (Some(dp), Some(np), Some(th)) = (dp, nopeaks.as_ref(), th_peaks) {
            out[i] = blend(v, np[i], dp, th, sigma);
            if dp > th {
                replaced += 1;
            }
        }
    }
    (out, replaced)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Brute-force reference: iterate `marker = max(mask, erode8(marker))`
    /// until stable.
    fn reference_reconstruction(mask: &[f32], mut marker: Vec<f32>, w: usize, h: usize) -> Vec<f32> {
        loop {
            let mut next = marker.clone();
            for y in 0..h {
                for x in 0..w {
                    let mut m = marker[y * w + x];
                    for (nx, ny) in neighbors(x, y, w, h) {
                        m = m.min(marker[ny * w + nx]);
                    }
                    next[y * w + x] = m.max(mask[y * w + x]);
                }
            }
            if next == marker {
                return marker;
            }
            marker = next;
        }
    }

    #[test]
    fn hybrid_reconstruction_matches_brute_force() {
        // Deterministic pseudo-random images (LCG).
        let (w, h) = (17usize, 13usize);
        let mut seed = 42u64;
        let mut rand = move || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 33) as f32) / (u32::MAX >> 1) as f32
        };
        for _ in 0..5 {
            let mask: Vec<f32> = (0..w * h).map(|_| rand()).collect();
            let mut marker = vec![f32::INFINITY; w * h];
            for x in 0..w {
                marker[x] = mask[x];
                marker[(h - 1) * w + x] = mask[(h - 1) * w + x];
            }
            for y in 0..h {
                marker[y * w] = mask[y * w];
                marker[y * w + w - 1] = mask[y * w + w - 1];
            }
            let fast = reconstruct_by_erosion(&mask, marker.clone(), w, h);
            let slow = reference_reconstruction(&mask, marker, w, h);
            for (a, b) in fast.iter().zip(&slow) {
                assert!((a - b).abs() < 1e-6, "hybrid deviates from brute force");
            }
        }
    }

    #[test]
    fn fills_interior_extrema_only() {
        let (w, h) = (11usize, 9usize);
        let mut img = vec![1.0f32; w * h];
        img[4 * w + 5] = 0.2; // interior dark spot
        img[3 * w + 8] = 2.5; // interior bright spot
        img[5 * w] = 0.2; // dark pixel ON the border: not a hole
        let noholes = fill_holes(&img, w, h);
        assert!(noholes[4 * w + 5] > 0.9, "interior hole must be filled");
        assert!((noholes[5 * w] - 0.2).abs() < 1e-6, "border pixel untouched");
        assert!((noholes[3 * w + 8] - 2.5).abs() < 1e-6, "peaks untouched by fill_holes");
        let nopeaks = fill_peaks(&img, w, h);
        assert!(nopeaks[3 * w + 8] < 1.1, "interior peak must be filled");
        assert!((nopeaks[4 * w + 5] - 0.2).abs() < 1e-6, "holes untouched by fill_peaks");
    }

    #[test]
    fn spot_clean_removes_specks_keeps_gradient() {
        let (w, h) = (32usize, 24usize);
        // A smooth gradient background with a few strong specks.
        let mut img: Vec<f32> = (0..w * h)
            .map(|i| 1.0 + 0.3 * ((i % w) as f32) / w as f32)
            .collect();
        img[10 * w + 10] = 6.0;
        img[15 * w + 20] = 5.5;
        img[7 * w + 4] = 0.01;
        let (cleaned, replaced) =
            morph_spot_clean(&img, w, h, MorphDetect::Both, 0.95, 0.025);
        assert!(replaced >= 3, "the three specks must be flagged, got {replaced}");
        assert!(cleaned[10 * w + 10] < 2.0, "bright speck must be removed");
        assert!(cleaned[15 * w + 20] < 2.0, "bright speck must be removed");
        assert!(cleaned[7 * w + 4] > 0.5, "dark speck must be filled");
        // The gradient background must be essentially untouched.
        let untouched = (0..w * h)
            .filter(|&i| ![10 * w + 10, 15 * w + 20, 7 * w + 4].contains(&i))
            .all(|i| (cleaned[i] - img[i]).abs() < 0.05);
        assert!(untouched, "background must stay unchanged");
    }
}
