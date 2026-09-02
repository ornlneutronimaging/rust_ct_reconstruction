//! Exporting the pre-processed stack as TIFF images after any
//! pre-processing step.
//!
//! Every step of the pre-processing screen (crop, remove outliers,
//! normalization, …) can write the stack as it is at that point into a
//! folder named after the step (`crop/`, `remove_outliers/`,
//! `normalization/`, …) created under a folder the user picks. The sample
//! projections go directly into that folder as float32 TIFFs, one per
//! projection in angle order; before the normalization the open beams (and
//! dark currents, when any) go into `ob/` and `dc/` sub-folders since those
//! steps transform them too. A `provenance.txt` next to the images records
//! the step and the stack's metadata.

use crate::combine::LoadedStack;
use crate::normalize::write_projection_tiffs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

/// What a step export writes besides the sample projections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExportExtra {
    /// Sample projections only (steps after the normalization: the open
    /// beams are already consumed).
    SampleOnly,
    /// Sample, open beams and dark currents (steps before the normalization,
    /// which transform all of them).
    WithObDc,
}

/// The folder a step export ends up in: `<base>/<step>`, or the first free
/// `<base>/<step>_2`, `<base>/<step>_3`, … when it already exists (so an
/// export never mixes with the files of an earlier run of the step).
pub fn export_dir(base: &Path, step: &str) -> PathBuf {
    let first = base.join(step);
    if !first.exists() {
        return first;
    }
    (2..)
        .map(|n| base.join(format!("{step}_{n}")))
        .find(|dir| !dir.exists())
        .expect("an unbounded counter finds a free folder name")
}

fn write_provenance(dir: &Path, step: &str, stack: &LoadedStack) -> Result<(), String> {
    let mut text = format!(
        "pre-processing step: {step}\nsource: {}\nsample projections: {}\n",
        stack.path.display(),
        stack.sample.len()
    );
    if !stack.ob.is_empty() {
        text.push_str(&format!("open beams: {}\n", stack.ob.len()));
    }
    if !stack.dc.is_empty() {
        text.push_str(&format!("dark currents: {}\n", stack.dc.len()));
    }
    if let Some(p) = stack.sample.first() {
        text.push_str(&format!(
            "image size: {}x{} (height x width)\n",
            p.height, p.width
        ));
    }
    if let Some(cor) = stack.center_of_rotation {
        text.push_str(&format!("center of rotation: {cor}\n"));
    }
    if !stack.metadata.is_empty() {
        text.push_str("\nmetadata:\n");
        for (name, value) in &stack.metadata {
            text.push_str(&format!("  {name}: {value}\n"));
        }
    }
    text.push_str("\nfiles: NNNN_<projection name>.tif, float32, in stack order\n");
    let path = dir.join("provenance.txt");
    std::fs::write(&path, text).map_err(|e| format!("write {}: {e}", path.display()))
}

/// Write the stack into `<base>/<step>` (see [`export_dir`]); returns a
/// one-line summary of what was written and where.
pub fn export_step(
    base: &Path,
    step: &str,
    stack: &LoadedStack,
    extra: ExportExtra,
) -> Result<String, String> {
    if stack.sample.is_empty() {
        return Err("the stack has no sample projections to export".to_owned());
    }
    let dir = export_dir(base, step);
    write_projection_tiffs(&dir, &stack.sample)?;
    let mut parts = vec![format!("{} sample image(s)", stack.sample.len())];
    if extra == ExportExtra::WithObDc {
        if !stack.ob.is_empty() {
            write_projection_tiffs(&dir.join("ob"), &stack.ob)?;
            parts.push(format!("{} ob image(s) in ob/", stack.ob.len()));
        }
        if !stack.dc.is_empty() {
            write_projection_tiffs(&dir.join("dc"), &stack.dc)?;
            parts.push(format!("{} dc image(s) in dc/", stack.dc.len()));
        }
    }
    write_provenance(&dir, step, stack)?;
    Ok(format!("{} — {}", dir.display(), parts.join(", ")))
}

/// One export on a background thread (the stack can be many GB).
pub struct ExportJob {
    /// The step being exported, for the status line.
    pub step: &'static str,
    rx: Receiver<Result<String, String>>,
}

impl ExportJob {
    pub fn start(
        base: PathBuf,
        step: &'static str,
        stack: Arc<LoadedStack>,
        extra: ExportExtra,
    ) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(export_step(&base, step, &stack, extra));
        });
        Self { step, rx }
    }

    pub fn poll(&mut self) -> Option<Result<String, String>> {
        self.rx.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::combine::Projection;

    fn projection(name: &str, w: usize, h: usize) -> Projection {
        Projection {
            name: name.to_owned(),
            run_number: None,
            angle_deg: None,
            n_images_used: 1,
            height: h,
            width: w,
            mean: vec![1.0; w * h],
            total_counts: 0.0,
        }
    }

    fn stack(dir: &Path, with_ob: bool) -> LoadedStack {
        LoadedStack {
            path: dir.join("in.h5"),
            sample: vec![projection("a", 4, 3), projection("b", 4, 3)],
            ob: if with_ob {
                vec![projection("ob", 4, 3)]
            } else {
                Vec::new()
            },
            dc: Vec::new(),
            metadata: vec![("normalization".to_owned(), "x".to_owned())],
            center_of_rotation: None,
        }
    }

    fn scratch(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ct_export_test_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn export_dir_never_reuses_an_existing_folder() {
        let base = scratch("dir");
        assert_eq!(export_dir(&base, "crop"), base.join("crop"));
        std::fs::create_dir(base.join("crop")).unwrap();
        assert_eq!(export_dir(&base, "crop"), base.join("crop_2"));
        std::fs::create_dir(base.join("crop_2")).unwrap();
        assert_eq!(export_dir(&base, "crop"), base.join("crop_3"));
        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn export_writes_sample_and_optionally_ob() {
        let base = scratch("write");
        let st = stack(&base, true);
        export_step(&base, "crop", &st, ExportExtra::WithObDc).unwrap();
        let dir = base.join("crop");
        assert!(dir.join("0000_a.tif").is_file());
        assert!(dir.join("0001_b.tif").is_file());
        assert!(dir.join("ob").join("0000_ob.tif").is_file());
        assert!(dir.join("provenance.txt").is_file());
        let text = std::fs::read_to_string(dir.join("provenance.txt")).unwrap();
        assert!(text.contains("pre-processing step: crop"));
        assert!(text.contains("normalization: x"));

        export_step(&base, "normalization", &st, ExportExtra::SampleOnly).unwrap();
        let dir = base.join("normalization");
        assert!(dir.join("0000_a.tif").is_file());
        assert!(!dir.join("ob").exists());
        let _ = std::fs::remove_dir_all(&base);
    }
}
