//! Rotating the stack in 90° steps, so the rotation axis of the scan ends up
//! vertical — a requirement of the reconstruction.

use crate::combine::{LoadedStack, Projection};
use rayon::prelude::*;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

/// Rotate a row-major `height × width` image clockwise by `quarters` × 90°.
/// Returns `(new_width, new_height, values)`.
pub fn rotate_quarter(
    data: &[f32],
    width: usize,
    height: usize,
    quarters: usize,
) -> (usize, usize, Vec<f32>) {
    let q = quarters % 4;
    match q {
        0 => (width, height, data.to_vec()),
        2 => {
            let mut out = Vec::with_capacity(data.len());
            out.extend(data.iter().rev().copied());
            (width, height, out)
        }
        _ => {
            // 90° (q = 1) and 270° (q = 3) swap the dimensions.
            let (new_w, new_h) = (height, width);
            let mut out = vec![0.0f32; data.len()];
            for y in 0..new_h {
                for (x, value) in out[y * new_w..(y + 1) * new_w].iter_mut().enumerate() {
                    *value = if q == 1 {
                        data[(height - 1 - x) * width + y]
                    } else {
                        data[x * width + (width - 1 - y)]
                    };
                }
            }
            (new_w, new_h, out)
        }
    }
}

fn rotate_projection(p: &Projection, quarters: usize) -> Projection {
    let (w, h, mean) = rotate_quarter(&p.mean, p.width, p.height, quarters);
    Projection {
        name: p.name.clone(),
        run_number: p.run_number,
        angle_deg: p.angle_deg,
        n_images_used: p.n_images_used,
        height: h,
        width: w,
        mean,
        total_counts: p.total_counts,
    }
}

/// Rotate the whole stack (sample and open beams) from the given baseline on
/// a background thread.
pub struct RotateJob {
    rx: Receiver<LoadedStack>,
}

impl RotateJob {
    pub fn start(baseline: Arc<LoadedStack>, quarters: usize) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let quarters = quarters % 4;
            let rotate_all = |projections: &[Projection]| -> Vec<Projection> {
                projections
                    .par_iter()
                    .map(|p| rotate_projection(p, quarters))
                    .collect()
            };
            let mut metadata = baseline.metadata.clone();
            metadata.retain(|(name, _)| name != "rotation");
            if quarters > 0 {
                metadata.push(("rotation".to_owned(), format!("{}° clockwise", quarters * 90)));
            }
            metadata.sort();
            let _ = tx.send(LoadedStack {
                path: baseline.path.clone(),
                sample: rotate_all(&baseline.sample),
                ob: rotate_all(&baseline.ob),
                metadata,
                // A quarter turn changes the geometry: any stored center of
                // rotation is void.
                center_of_rotation: None,
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

    // 2 rows x 3 columns:
    //   a b c
    //   d e f
    const SRC: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

    #[test]
    fn quarter_turns() {
        let (w, h, out) = rotate_quarter(&SRC, 3, 2, 0);
        assert_eq!((w, h), (3, 2));
        assert_eq!(out, SRC);

        // 90° CW:  d a / e b / f c
        let (w, h, out) = rotate_quarter(&SRC, 3, 2, 1);
        assert_eq!((w, h), (2, 3));
        assert_eq!(out, [4.0, 1.0, 5.0, 2.0, 6.0, 3.0]);

        // 180°:    f e d / c b a
        let (w, h, out) = rotate_quarter(&SRC, 3, 2, 2);
        assert_eq!((w, h), (3, 2));
        assert_eq!(out, [6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);

        // 270° CW: c f / b e / a d
        let (w, h, out) = rotate_quarter(&SRC, 3, 2, 3);
        assert_eq!((w, h), (2, 3));
        assert_eq!(out, [3.0, 6.0, 2.0, 5.0, 1.0, 4.0]);

        // Four quarter turns are the identity.
        let (_, _, once) = rotate_quarter(&SRC, 3, 2, 1);
        let (_, _, back) = rotate_quarter(&once, 2, 3, 3);
        assert_eq!(back, SRC);
    }
}
