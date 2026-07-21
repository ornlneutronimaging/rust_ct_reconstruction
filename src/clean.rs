//! "Remove outliers" cleaning of the loaded stack — the three methods of the
//! Python `ImagesCleaner`, ported natively:
//!
//! - **In-house (histogram)**: per image, everything at or below the
//!   `exclude_left`-th histogram bin edge or above the
//!   `(bins - exclude_right)`-th edge is replaced from a median-filtered
//!   version of the image (`(2r+1)²` window).
//! - **Tomopy `remove_outlier`**: pixels brighter than the 3×3 median of
//!   their slice by more than `diff` are replaced by that median.
//! - **Scipy `median_filter` size (1,3,3)**: every pixel replaced by the 3×3
//!   median of its slice.
//!
//! Applied in that order (the Python `cleaning()` order) to the sample AND
//! open-beam stacks.

use crate::combine::{LoadedStack, Projection};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, channel};

#[derive(Clone, Debug)]
pub struct CleanSettings {
    pub in_house: bool,
    /// Histogram bins of the in-house method (10–1000).
    pub nbr_bins: usize,
    pub exclude_left: usize,
    pub exclude_right: usize,
    /// Replacement window radius r of the in-house method: (2r+1)².
    pub correct_radius: usize,
    pub tomopy: bool,
    pub tomopy_diff: f32,
    pub scipy: bool,
}

impl Default for CleanSettings {
    fn default() -> Self {
        Self {
            in_house: false,
            nbr_bins: 100,
            exclude_left: 1,
            exclude_right: 1,
            correct_radius: 1,
            tomopy: false,
            tomopy_diff: 20.0,
            scipy: false,
        }
    }
}

impl CleanSettings {
    pub fn any_enabled(&self) -> bool {
        self.in_house || self.tomopy || self.scipy
    }

    /// One line for the log and the HDF5 provenance.
    pub fn describe(&self) -> String {
        let mut parts = Vec::new();
        if self.in_house {
            parts.push(format!(
                "in-house histogram (bins {}, exclude {} left / {} right, radius {})",
                self.nbr_bins, self.exclude_left, self.exclude_right, self.correct_radius
            ));
        }
        if self.tomopy {
            parts.push(format!("tomopy remove_outlier (diff {})", self.tomopy_diff));
        }
        if self.scipy {
            parts.push("scipy 3x3 median filter".to_owned());
        }
        parts.join(" + ")
    }
}

/// Mirror an index into `[0, n)` the way scipy's `reflect` border mode does.
fn reflect(index: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = index;
    if i < 0 {
        i = -i - 1;
    }
    if i >= n {
        i = 2 * n - i - 1;
    }
    i.clamp(0, n - 1) as usize
}

/// 2-D median filter with a `(2r+1)²` window and reflected borders.
pub fn median_filter_2d(data: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let r = radius as isize;
    let mut out = vec![0.0f32; data.len()];
    out.par_chunks_mut(width)
        .enumerate()
        .for_each(|(y, out_row)| {
            let mut window = Vec::with_capacity((2 * radius + 1).pow(2));
            for (x, out_value) in out_row.iter_mut().enumerate() {
                window.clear();
                for dy in -r..=r {
                    let yy = reflect(y as isize + dy, height);
                    for dx in -r..=r {
                        let xx = reflect(x as isize + dx, width);
                        window.push(data[yy * width + xx]);
                    }
                }
                let mid = window.len() / 2;
                window.select_nth_unstable_by(mid, f32::total_cmp);
                *out_value = window[mid];
            }
        });
    out
}

/// In-house histogram cleaning of one image (the Python `replace_pixels`);
/// returns how many pixels were replaced.
fn clean_in_house(p: &mut Projection, s: &CleanSettings) -> usize {
    if s.exclude_left == 0 && s.exclude_right == 0 {
        return 0;
    }
    let (mut min, mut max) = (f32::MAX, f32::MIN);
    for v in &p.mean {
        min = min.min(*v);
        max = max.max(*v);
    }
    if !(max > min) {
        return 0;
    }
    let edge = |k: usize| min + (max - min) * k as f32 / s.nbr_bins as f32;
    let thres_low = edge(s.exclude_left.min(s.nbr_bins));
    let thres_high = edge(s.nbr_bins.saturating_sub(s.exclude_right));
    let is_outlier = |v: f32| v <= thres_low || v > thres_high;
    if !p.mean.iter().any(|v| is_outlier(*v)) {
        return 0;
    }
    let filtered = median_filter_2d(&p.mean, p.width, p.height, s.correct_radius.max(1));
    let mut replaced = 0;
    for (v, m) in p.mean.iter_mut().zip(&filtered) {
        if is_outlier(*v) {
            *v = *m;
            replaced += 1;
        }
    }
    replaced
}

/// Tomopy `remove_outlier`: replace bright outliers by the 3×3 median.
fn clean_tomopy(p: &mut Projection, diff: f32) -> usize {
    let filtered = median_filter_2d(&p.mean, p.width, p.height, 1);
    let mut replaced = 0;
    for (v, m) in p.mean.iter_mut().zip(&filtered) {
        if *v - *m > diff {
            *v = *m;
            replaced += 1;
        }
    }
    replaced
}

/// Scipy `median_filter` size (1, 3, 3): every pixel becomes the 3×3 median.
fn clean_scipy(p: &mut Projection) {
    p.mean = median_filter_2d(&p.mean, p.width, p.height, 1);
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CleanStats {
    pub in_house_replaced: usize,
    pub tomopy_replaced: usize,
    pub scipy_applied: bool,
}

fn clean_projection(p: &mut Projection, s: &CleanSettings) -> CleanStats {
    let mut stats = CleanStats::default();
    if s.in_house {
        stats.in_house_replaced = clean_in_house(p, s);
    }
    if s.tomopy {
        stats.tomopy_replaced = clean_tomopy(p, s.tomopy_diff);
    }
    if s.scipy {
        clean_scipy(p);
        stats.scipy_applied = true;
    }
    let sum: f64 = p.mean.iter().map(|v| f64::from(*v)).sum();
    p.total_counts = sum * p.n_images_used.max(1) as f64;
    stats
}

/// The cleaning pass on background threads (rayon over the images of the
/// sample then open-beam stacks); `progress()` counts images done.
pub struct CleanJob {
    rx: Receiver<(LoadedStack, CleanStats)>,
    progress: Arc<AtomicUsize>,
    pub total: usize,
}

impl CleanJob {
    pub fn start(stack: Arc<LoadedStack>, settings: CleanSettings) -> Self {
        let (tx, rx) = channel();
        let progress = Arc::new(AtomicUsize::new(0));
        let thread_progress = Arc::clone(&progress);
        let total = stack.sample.len() + stack.ob.len();
        std::thread::spawn(move || {
            let run = |projections: &[Projection]| -> (Vec<Projection>, CleanStats) {
                let results: Vec<(Projection, CleanStats)> = projections
                    .par_iter()
                    .map(|p| {
                        let mut p = Projection::clone(p);
                        let stats = clean_projection(&mut p, &settings);
                        thread_progress.fetch_add(1, Ordering::Relaxed);
                        (p, stats)
                    })
                    .collect();
                let mut total = CleanStats::default();
                let mut cleaned = Vec::with_capacity(results.len());
                for (p, stats) in results {
                    total.in_house_replaced += stats.in_house_replaced;
                    total.tomopy_replaced += stats.tomopy_replaced;
                    total.scipy_applied |= stats.scipy_applied;
                    cleaned.push(p);
                }
                (cleaned, total)
            };
            let (sample, sample_stats) = run(&stack.sample);
            let (ob, ob_stats) = run(&stack.ob);
            let stats = CleanStats {
                in_house_replaced: sample_stats.in_house_replaced + ob_stats.in_house_replaced,
                tomopy_replaced: sample_stats.tomopy_replaced + ob_stats.tomopy_replaced,
                scipy_applied: sample_stats.scipy_applied || ob_stats.scipy_applied,
            };
            let mut metadata = stack.metadata.clone();
            metadata.retain(|(name, _)| name != "outlier_removal");
            metadata.push(("outlier_removal".to_owned(), settings.describe()));
            metadata.sort();
            let cleaned = LoadedStack {
                path: stack.path.clone(),
                sample,
                ob,
                metadata,
            };
            let _ = tx.send((cleaned, stats));
        });
        Self {
            rx,
            progress,
            total,
        }
    }

    pub fn done(&self) -> usize {
        self.progress.load(Ordering::Relaxed)
    }

    pub fn poll(&mut self) -> Option<(LoadedStack, CleanStats)> {
        self.rx.try_recv().ok()
    }
}

/// Histogram of `values` over `bins` equal bins spanning `[min, max]`;
/// values outside land in the first/last bin. `(min, max, counts)`.
pub fn histogram_range(values: &[f32], bins: usize, min: f64, max: f64) -> (f64, f64, Vec<u64>) {
    let mut counts = vec![0u64; bins.max(1)];
    if !(max > min) {
        return (min, max, counts);
    }
    let n = counts.len();
    let scale = n as f64 / (max - min);
    for v in values {
        let k = ((f64::from(*v) - min) * scale).max(0.0) as usize;
        counts[k.min(n - 1)] += 1;
    }
    (min, max, counts)
}

/// Histogram of `values` over `bins` equal bins between the data min/max.
pub fn histogram(values: &[f32], bins: usize) -> (f64, f64, Vec<u64>) {
    let (mut min, mut max) = (f64::MAX, f64::MIN);
    for v in values {
        let v = f64::from(*v);
        min = min.min(v);
        max = max.max(v);
    }
    histogram_range(values, bins, min, max)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projection(w: usize, h: usize, mean: Vec<f32>) -> Projection {
        Projection {
            name: "p".to_owned(),
            run_number: None,
            angle_deg: None,
            n_images_used: 1,
            height: h,
            width: w,
            mean,
            total_counts: 0.0,
        }
    }

    #[test]
    fn median_filter_removes_a_spike() {
        let mut data = vec![10.0f32; 25];
        data[12] = 1000.0; // center spike
        let filtered = median_filter_2d(&data, 5, 5, 1);
        assert_eq!(filtered[12], 10.0);
        // A constant field stays constant, including at borders.
        assert!(filtered.iter().all(|v| *v == 10.0));
    }

    #[test]
    fn tomopy_replaces_only_bright_outliers() {
        let mut data = vec![100.0f32; 25];
        data[6] = 500.0; // bright: replaced
        data[18] = 0.0; // dark: kept (tomopy only removes bright spots)
        let mut p = projection(5, 5, data);
        let replaced = clean_tomopy(&mut p, 20.0);
        assert_eq!(replaced, 1);
        assert_eq!(p.mean[6], 100.0);
        assert_eq!(p.mean[18], 0.0);
    }

    #[test]
    fn in_house_replaces_both_tails() {
        let mut data = vec![100.0f32; 25];
        data[6] = 1000.0;
        data[18] = -50.0;
        let mut p = projection(5, 5, data);
        let settings = CleanSettings {
            in_house: true,
            nbr_bins: 10,
            exclude_left: 1,
            exclude_right: 1,
            correct_radius: 1,
            ..CleanSettings::default()
        };
        let replaced = clean_in_house(&mut p, &settings);
        assert_eq!(replaced, 2);
        assert_eq!(p.mean[6], 100.0);
        assert_eq!(p.mean[18], 100.0);
        // With no bins excluded nothing happens.
        let mut p2 = projection(5, 5, vec![1.0; 25]);
        let none = CleanSettings {
            exclude_left: 0,
            exclude_right: 0,
            ..settings
        };
        assert_eq!(clean_in_house(&mut p2, &none), 0);
    }

    #[test]
    fn histogram_counts() {
        let values: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let (min, max, counts) = histogram(&values, 10);
        assert_eq!((min, max), (0.0, 99.0));
        assert_eq!(counts.iter().sum::<u64>(), 100);
        assert!(counts.iter().all(|c| *c == 10), "{counts:?}");
        // A fixed range keeps its edges; out-of-range values land in the
        // first/last bin.
        let (min, max, counts) = histogram_range(&values, 10, 0.0, 200.0);
        assert_eq!((min, max), (0.0, 200.0));
        assert_eq!(counts.iter().sum::<u64>(), 100);
        let (.., counts) = histogram_range(&[-5.0, 50.0, 500.0], 10, 0.0, 100.0);
        assert_eq!((counts[0], counts[5], counts[9]), (1, 1, 1));
    }
}
