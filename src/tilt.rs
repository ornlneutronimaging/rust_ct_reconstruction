//! Tilt (and rotation-axis offset) correction — a native port of neutompy's
//! `find_COR` / `correction_COR` (the `test_tilt_correction` path of the
//! Python notebook).
//!
//! The projection at 180° is flipped horizontally; on each sampled row the
//! integer shift best aligning it with the 0° projection is found by RMSE,
//! and a linear fit of shift versus row gives the rotation-axis tilt
//! `atan(slope / 2)` and its offset from the detector center. The correction
//! rotates every projection by the tilt (bilinear, edge-padded) and rolls it
//! horizontally by the offset.

use crate::combine::{LoadedStack, Projection};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

#[derive(Clone, Copy, Debug)]
pub struct TiltResult {
    /// Tilt of the rotation axis with respect to the vertical, in degrees.
    pub tilt_deg: f64,
    /// Horizontal shift of the rotation axis with respect to the detector
    /// center, in pixels (the roll applied by the correction).
    pub shift_px: i64,
    /// Fit diagnostics: shift-per-row slope, intercept, rows used.
    pub slope: f64,
    pub intercept: f64,
    pub rows_used: usize,
}

impl TiltResult {
    /// Estimated rotation-axis column at `row` (from `t = w - 1 - 2c`, the
    /// relation between the fitted per-row shift and the axis position).
    pub fn axis_column(&self, row: f64, width: usize) -> f64 {
        (width as f64 - 1.0 - (self.slope * row + self.intercept)) / 2.0
    }
}

/// neutompy `find_COR`: estimate the tilt and offset of the rotation axis
/// from the projections at 0° and 180°, using rows `y_top..=y_bottom`
/// sampled every `ystep`.
pub fn find_cor(
    proj_0: &[f32],
    proj_180: &[f32],
    width: usize,
    height: usize,
    y_top: usize,
    y_bottom: usize,
    ystep: usize,
) -> Result<TiltResult, String> {
    if proj_0.len() != width * height || proj_180.len() != width * height {
        return Err("projection buffers do not match the given size".to_owned());
    }
    let y_top = y_top.min(height - 1);
    let y_bottom = y_bottom.min(height - 1);
    if y_bottom <= y_top {
        return Err("the slice range for the tilt is empty".to_owned());
    }
    let rows: Vec<usize> = (y_top..=y_bottom).step_by(ystep.max(1)).collect();
    if rows.len() < 2 {
        return Err("the slice range holds fewer than 2 sampled rows".to_owned());
    }

    let w = width as isize;
    let t_min = -(w / 2);
    let t_max = w - w / 2;
    // Best-aligning shift per sampled row (np.roll semantics: roll right by t).
    let shifts: Vec<f64> = rows
        .par_iter()
        .map(|&row| {
            let p0 = &proj_0[row * width..(row + 1) * width];
            let p180 = &proj_180[row * width..(row + 1) * width];
            let mut best = (f64::MAX, t_min);
            for t in t_min..=t_max {
                let mut sum = 0.0f64;
                for i in 0..w {
                    let rolled = p0[(i - t).rem_euclid(w) as usize];
                    // proj_180 flipped horizontally.
                    let flipped = p180[(w - 1 - i) as usize];
                    let d = f64::from(rolled - flipped);
                    sum += d * d;
                }
                if sum <= best.0 {
                    best = (sum, t);
                }
            }
            best.1 as f64
        })
        .collect();

    // Linear least-squares fit: shift = m * row + q.
    let n = rows.len() as f64;
    let mean_x = rows.iter().map(|&r| r as f64).sum::<f64>() / n;
    let mean_y = shifts.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var = 0.0;
    for (&row, &shift) in rows.iter().zip(&shifts) {
        let dx = row as f64 - mean_x;
        cov += dx * (shift - mean_y);
        var += dx * dx;
    }
    if var == 0.0 {
        return Err("degenerate slice range for the tilt fit".to_owned());
    }
    let m = cov / var;
    let q = mean_y - m * mean_x;
    let tilt_deg = (0.5 * m).atan().to_degrees();
    let middle = (m * height as f64 * 0.5 + q).round() as i64;
    let shift_px = middle.div_euclid(2);
    Ok(TiltResult {
        tilt_deg,
        shift_px,
        slope: m,
        intercept: q,
        rows_used: rows.len(),
    })
}

/// Rotate an image by `theta_deg` around its center (bilinear interpolation,
/// edge-clamped like neutompy's edge padding) and then roll it horizontally
/// by `shift` pixels.
pub fn rotate_roll(
    data: &[f32],
    width: usize,
    height: usize,
    theta_deg: f64,
    shift: i64,
) -> Vec<f32> {
    let (w, h) = (width as f64, height as f64);
    let (cx, cy) = ((w - 1.0) * 0.5, (h - 1.0) * 0.5);
    let theta = theta_deg.to_radians();
    let (sin, cos) = theta.sin_cos();
    let mut out = vec![0.0f32; data.len()];
    out.par_chunks_mut(width).enumerate().for_each(|(y, row)| {
        let dy = y as f64 - cy;
        for (x, value) in row.iter_mut().enumerate() {
            // The output pixel, rolled back, then rotated back to the source.
            let xr = (x as i64 - shift).rem_euclid(width as i64) as f64;
            let dx = xr - cx;
            let sx = cos * dx - sin * dy + cx;
            let sy = sin * dx + cos * dy + cy;
            // Bilinear sample with edge clamping.
            let x0 = sx.floor();
            let y0 = sy.floor();
            let fx = (sx - x0) as f32;
            let fy = (sy - y0) as f32;
            let clamp_x = |v: f64| (v.max(0.0) as usize).min(width - 1);
            let clamp_y = |v: f64| (v.max(0.0) as usize).min(height - 1);
            let (x0i, x1i) = (clamp_x(x0), clamp_x(x0 + 1.0));
            let (y0i, y1i) = (clamp_y(y0), clamp_y(y0 + 1.0));
            let top = data[y0i * width + x0i] * (1.0 - fx) + data[y0i * width + x1i] * fx;
            let bottom = data[y1i * width + x0i] * (1.0 - fx) + data[y1i * width + x1i] * fx;
            *value = top * (1.0 - fy) + bottom * fy;
        }
    });
    out
}

/// The tilt estimation on a background thread.
pub struct TiltCalcJob {
    rx: Receiver<Result<TiltResult, String>>,
}

impl TiltCalcJob {
    pub fn start(
        stack: Arc<LoadedStack>,
        index_0: usize,
        index_180: usize,
        y_top: usize,
        y_bottom: usize,
    ) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let result = (|| {
                let p0 = stack
                    .sample
                    .get(index_0)
                    .ok_or("0-degree projection not found")?;
                let p180 = stack
                    .sample
                    .get(index_180)
                    .ok_or("180-degree projection not found")?;
                if (p0.width, p0.height) != (p180.width, p180.height) {
                    return Err("the 0 and 180 degree projections differ in size".to_owned());
                }
                find_cor(
                    &p0.mean,
                    &p180.mean,
                    p0.width,
                    p0.height,
                    y_top,
                    y_bottom,
                    5,
                )
            })();
            let _ = tx.send(result);
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<TiltResult, String>> {
        self.rx.try_recv().ok()
    }
}

/// Applying the correction (rotate + roll of every sample projection) on a
/// background thread.
pub struct TiltApplyJob {
    rx: Receiver<LoadedStack>,
}

impl TiltApplyJob {
    pub fn start(stack: Arc<LoadedStack>, result: TiltResult) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let sample: Vec<Projection> = stack
                .sample
                .par_iter()
                .map(|p| {
                    let mean =
                        rotate_roll(&p.mean, p.width, p.height, result.tilt_deg, result.shift_px);
                    let sum: f64 = mean.iter().map(|v| f64::from(*v)).sum();
                    Projection {
                        name: p.name.clone(),
                        run_number: p.run_number,
                        angle_deg: p.angle_deg,
                        n_images_used: p.n_images_used,
                        height: p.height,
                        width: p.width,
                        mean,
                        total_counts: sum * p.n_images_used.max(1) as f64,
                    }
                })
                .collect();
            let mut metadata = stack.metadata.clone();
            metadata.retain(|(name, _)| name != "tilt_correction");
            metadata.push((
                "tilt_correction".to_owned(),
                format!(
                    "tilt {:.4} deg, axis shift {} px (neutompy find_COR port)",
                    result.tilt_deg, result.shift_px
                ),
            ));
            metadata.sort();
            let _ = tx.send(LoadedStack {
                path: stack.path.clone(),
                sample,
                ob: stack.ob.clone(),
                metadata,
            });
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<LoadedStack> {
        self.rx.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 96;
    const H: usize = 96;

    /// A pair of 0°/180° projections of a single bright rod at distance `d`
    /// from a rotation axis whose column is `c0 + k * row`.
    fn synthetic_pair(c0: f64, k: f64, d: f64) -> (Vec<f32>, Vec<f32>) {
        let peak = |x: f64, center: f64| (-((x - center) * (x - center)) / 6.0).exp() as f32;
        let mut p0 = vec![0.0f32; W * H];
        let mut p180 = vec![0.0f32; W * H];
        for y in 0..H {
            let axis = c0 + k * y as f64;
            for x in 0..W {
                p0[y * W + x] = peak(x as f64, axis + d);
                p180[y * W + x] = peak(x as f64, axis - d);
            }
        }
        (p0, p180)
    }

    #[test]
    fn straight_axis_is_detected_as_straight() {
        let (p0, p180) = synthetic_pair(W as f64 / 2.0, 0.0, 12.0);
        let r = find_cor(&p0, &p180, W, H, 4, H - 4, 3).unwrap();
        assert!(r.tilt_deg.abs() < 0.2, "tilt {}", r.tilt_deg);
        assert!(r.shift_px.abs() <= 1, "shift {}", r.shift_px);
    }

    #[test]
    fn tilt_correction_straightens_a_tilted_axis() {
        // Axis leaning ~2.3° with an offset from the detector center.
        let k = 0.04;
        let (p0, p180) = synthetic_pair(W as f64 / 2.0 + 5.0, k, 12.0);
        let found = find_cor(&p0, &p180, W, H, 4, H - 4, 3).unwrap();
        // Expected tilt: atan(-k) (the fitted slope is -2k).
        let expected = (-k).atan().to_degrees();
        assert!(
            (found.tilt_deg - expected).abs() < 0.3,
            "found {} expected {expected}",
            found.tilt_deg
        );
        // The estimated axis line tracks the synthetic axis c0 + k * row.
        for row in [10.0, 50.0, 80.0] {
            let expected_c = W as f64 / 2.0 + 5.0 + k * row;
            let got = found.axis_column(row, W);
            assert!((got - expected_c).abs() < 1.5, "row {row}: {got} vs {expected_c}");
        }
        // Apply the correction to both projections and re-estimate: the
        // axis must now be vertical and centered.
        let c0 = rotate_roll(&p0, W, H, found.tilt_deg, found.shift_px);
        let c180 = rotate_roll(&p180, W, H, found.tilt_deg, found.shift_px);
        let after = find_cor(&c0, &c180, W, H, 8, H - 8, 3).unwrap();
        assert!(after.tilt_deg.abs() < 0.15, "residual tilt {}", after.tilt_deg);
        assert!(after.shift_px.abs() <= 1, "residual shift {}", after.shift_px);
    }
}
