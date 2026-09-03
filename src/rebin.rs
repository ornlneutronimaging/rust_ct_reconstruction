//! "Rebin" pre-processing step — the last one, run on the normalized
//! projections: every n×n block of pixels is averaged into one pixel. The
//! images get n times narrower and shorter (trailing rows / columns that do
//! not fill a whole block are dropped), with better statistics per pixel
//! and a coarser resolution. Open beam / dark current images still in the
//! stack are rebinned the same way.
//!
//! The mbirjax reconstruction is the reason this step exists: its GPU memory
//! use grows with the projection width, and full-frame (4096 px wide)
//! stacks do not fit, however few slices are reconstructed at a time.

use crate::combine::{LoadedStack, Projection};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, channel};

/// The rebin factors offered in the UI.
pub const FACTORS: [usize; 6] = [2, 3, 4, 5, 6, 8];

/// The image size after an n×n rebin of `width` × `height`.
pub fn rebinned_size(width: usize, height: usize, n: usize) -> (usize, usize) {
    (width / n.max(1), height / n.max(1))
}

/// One projection rebinned n×n (block mean).
pub fn rebin_projection(p: &Projection, n: usize) -> Projection {
    let n = n.max(1);
    let (width, height) = rebinned_size(p.width, p.height, n);
    let mut mean = vec![0.0f32; width * height];
    let inv = 1.0 / (n * n) as f32;
    for (row, out) in mean.chunks_mut(width).enumerate() {
        for (col, v) in out.iter_mut().enumerate() {
            let mut sum = 0.0f32;
            for dy in 0..n {
                let start = (row * n + dy) * p.width + col * n;
                sum += p.mean[start..start + n].iter().sum::<f32>();
            }
            *v = sum * inv;
        }
    }
    let sum: f64 = mean.iter().map(|v| f64::from(*v)).sum();
    Projection {
        name: p.name.clone(),
        run_number: p.run_number,
        angle_deg: p.angle_deg,
        n_images_used: p.n_images_used,
        height,
        width,
        mean,
        total_counts: sum * p.n_images_used.max(1) as f64,
    }
}

/// The `/metadata` line recording an n×n rebin of `width` × `height` images.
pub fn describe(n: usize, width: usize, height: usize) -> String {
    let (w, h) = rebinned_size(width, height, n);
    format!("{n}x{n} (block mean), {width}x{height} -> {w}x{h}")
}

/// The factor recorded by [`describe`], back from the metadata line.
pub fn factor_from_description(desc: &str) -> Option<usize> {
    desc.split('x').next()?.trim().parse().ok()
}

/// A center of rotation in the rebinned image: pixel centers move from
/// `x` to `(x + 0.5) / n - 0.5`.
pub fn rebin_center(cor: f64, n: usize) -> f64 {
    (cor + 0.5) / n.max(1) as f64 - 0.5
}

/// The rebin pass on background threads (rayon over the images); `done()`
/// counts images finished out of `total`.
pub struct RebinJob {
    rx: Receiver<LoadedStack>,
    progress: Arc<AtomicUsize>,
    pub total: usize,
    pub factor: usize,
}

impl RebinJob {
    pub fn start(stack: Arc<LoadedStack>, n: usize) -> Self {
        let (tx, rx) = channel();
        let progress = Arc::new(AtomicUsize::new(0));
        let thread_progress = Arc::clone(&progress);
        let total = stack.sample.len() + stack.ob.len() + stack.dc.len();
        std::thread::spawn(move || {
            let run = |projections: &[Projection]| -> Vec<Projection> {
                projections
                    .par_iter()
                    .map(|p| {
                        let out = rebin_projection(p, n);
                        thread_progress.fetch_add(1, Ordering::Relaxed);
                        out
                    })
                    .collect()
            };
            let (width, height) = stack
                .sample
                .first()
                .map(|p| (p.width, p.height))
                .unwrap_or((0, 0));
            let mut metadata = stack.metadata.clone();
            metadata.retain(|(name, _)| name != "rebin");
            metadata.push(("rebin".to_owned(), describe(n, width, height)));
            metadata.sort();
            let rebinned = LoadedStack {
                path: stack.path.clone(),
                sample: run(&stack.sample),
                ob: run(&stack.ob),
                dc: run(&stack.dc),
                metadata,
                center_of_rotation: stack.center_of_rotation.map(|c| rebin_center(c, n)),
            };
            let _ = tx.send(rebinned);
        });
        Self {
            rx,
            progress,
            total,
            factor: n,
        }
    }

    pub fn poll(&mut self) -> Option<LoadedStack> {
        self.rx.try_recv().ok()
    }

    pub fn done(&self) -> usize {
        self.progress.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projection(width: usize, height: usize, mean: Vec<f32>) -> Projection {
        Projection {
            name: "p".to_owned(),
            run_number: None,
            angle_deg: Some(1.0),
            n_images_used: 1,
            height,
            width,
            mean,
            total_counts: 0.0,
        }
    }

    #[test]
    fn block_mean_2x2() {
        // 4x2 image: two 2x2 blocks.
        let p = projection(4, 2, vec![1.0, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0]);
        let r = rebin_projection(&p, 2);
        assert_eq!((r.width, r.height), (2, 1));
        assert_eq!(r.mean, vec![2.5, 25.0]);
        assert_eq!(r.angle_deg, Some(1.0));
        assert_eq!(r.total_counts, 27.5);
    }

    #[test]
    fn trailing_pixels_are_dropped() {
        // 5x5 rebinned 2x2 -> 2x2, the last row and column ignored.
        let mean: Vec<f32> = (0..25).map(|v| v as f32).collect();
        let r = rebin_projection(&projection(5, 5, mean), 2);
        assert_eq!((r.width, r.height), (2, 2));
        // Block (0,0) = mean(0, 1, 5, 6) = 3.
        assert_eq!(r.mean[0], 3.0);
        // Block (1,1) = mean(12, 13, 17, 18) = 15.
        assert_eq!(r.mean[3], 15.0);
    }

    #[test]
    fn description_round_trip_and_center() {
        let desc = describe(4, 4096, 2048);
        assert_eq!(desc, "4x4 (block mean), 4096x2048 -> 1024x512");
        assert_eq!(factor_from_description(&desc), Some(4));
        assert_eq!(factor_from_description("nonsense"), None);
        // The center of the full image stays the center of the rebinned one.
        assert!((rebin_center(2047.5, 2) - 1023.5).abs() < 1e-9);
        assert!((rebin_center(0.5, 2) - 0.0).abs() < 1e-9);
    }
}
