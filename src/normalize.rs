//! Normalization of the stack — the mandatory pre-processing step.
//!
//! The core division (and the beam-fluctuation ROI correction) runs through
//! the NeuNorm Python library (https://github.com/ornlneutronimaging/NeuNorm),
//! handing the stacks over as `.npy` files. The sample-ROI normalization
//! (median of the normalized ROI anchored to 1) is not in NeuNorm yet and is
//! applied locally afterwards, followed by the notebook's clamp to [0, 1].
//! Both corrections share ONE ROI, selected with the sibling
//! `rust_roi_selector` application on the integrated sample image.

use crate::combine::{LoadedStack, Projection};
use crate::crop::{CropRect, read_npy, write_npy, write_npy_stack};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

/// The ROI selector of the sibling repo.
pub const ROI_SELECTOR_BIN: &str =
    "/SNS/VENUS/shared/software/git/rust_roi_selector/target/release/roi_selector";

/// The interpreter used to run NeuNorm.
pub const PYTHON: &str = "python3";

#[derive(Clone, Debug, Default)]
pub struct NormSettings {
    /// Beam fluctuation correction: NeuNorm divides every sample and OB
    /// image by the mean of its own ROI before the division.
    pub beam_fluctuation: bool,
    /// The ROI of the beam fluctuation correction.
    pub roi: Option<CropRect>,
}

impl NormSettings {
    pub fn needs_roi(&self) -> bool {
        self.beam_fluctuation
    }

    pub fn describe(&self) -> String {
        let mut text = "NeuNorm division by the mean OB".to_owned();
        if let Some(roi) = &self.roi
            && self.beam_fluctuation
        {
            text.push_str(&format!(
                ", beam fluctuation correction (ROI x={}, y={}, {}x{})",
                roi.x, roi.y, roi.width, roi.height
            ));
        }
        text.push_str(", clamped to [0, 1]");
        text
    }
}

pub(crate) fn scratch_dir(stack: &LoadedStack, tag: &str) -> Result<PathBuf, String> {
    let base = stack
        .path
        .parent()
        .filter(|p| p.is_dir())
        .map(Path::to_path_buf)
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join(format!(".ct_recon_{tag}_{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_ok() {
        return Ok(dir);
    }
    let dir = std::env::temp_dir().join(format!("ct_recon_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    Ok(dir)
}

/// One ROI-selector session on a background thread: the integrated (mean)
/// sample image is handed over, and the bounding rectangle of the returned
/// mask becomes the normalization ROI. `Ok(None)` when the selector was
/// closed without saving.
pub struct RoiJob {
    rx: Receiver<Result<Option<CropRect>, String>>,
}

impl RoiJob {
    pub fn start(stack: Arc<LoadedStack>) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(run_roi_selector(&stack));
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<Option<CropRect>, String>> {
        self.rx.try_recv().ok()
    }
}

fn run_roi_selector(stack: &LoadedStack) -> Result<Option<CropRect>, String> {
    let first = stack
        .sample
        .first()
        .ok_or("no sample projections in the stack")?;
    let (w, h) = (first.width, first.height);
    if let Some(bad) = stack.sample.iter().find(|p| (p.width, p.height) != (w, h)) {
        return Err(format!("projections have inconsistent sizes ({})", bad.name));
    }

    let dir = scratch_dir(stack, "roi")?;
    let image = dir.join("sample_stack.npy");
    let mask = dir.join("normalization_roi_mask.npy");
    let cleanup = || {
        let _ = std::fs::remove_file(&image);
        let _ = std::fs::remove_file(&mask);
        let _ = std::fs::remove_dir(&dir);
    };
    // The FULL stack goes over (3-D npy, one frame per projection), so the
    // region can be checked against every rotation angle; the selector opens
    // on the single-image view with a frame slider.
    if let Err(e) = write_npy(
        &image,
        &[stack.sample.len(), h, w],
        stack.sample.iter().map(|p| p.mean.as_slice()),
    ) {
        cleanup();
        return Err(e);
    }
    let result = std::process::Command::new(ROI_SELECTOR_BIN)
        .arg(&image)
        .arg("--called-from-python")
        .arg("--single-image")
        .arg("--output")
        .arg(&mask)
        .arg("--instructions")
        .arg(
            "Select a region AWAY from the sample: it must see only open beam at EVERY \
             rotation angle — use the slider (or the Integrated view) to check every \
             projection. It is used to normalize the beam intensity.",
        )
        .output();
    let output = match result {
        Err(e) => {
            cleanup();
            return Err(format!("cannot launch {ROI_SELECTOR_BIN}: {e}"));
        }
        Ok(out) => out,
    };
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        cleanup();
        return Err(format!("roi_selector failed ({}): {stderr}", output.status));
    }
    if !mask.is_file() {
        cleanup();
        return Ok(None);
    }
    let parsed = read_npy(&mask);
    cleanup();
    let (shape, values) = parsed?;
    if shape != [h, w] {
        return Err(format!(
            "mask shape {shape:?} does not match the image ({h}, {w})"
        ));
    }
    // Bounding rectangle of the selection.
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for (i, v) in values.iter().enumerate() {
        if *v > 0.5 {
            let (y, x) = (i / w, i % w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
    }
    if x0 == usize::MAX {
        return Err("the saved selection is empty — draw a region first".to_owned());
    }
    Ok(Some(CropRect {
        x: x0,
        y: y0,
        width: x1 - x0 + 1,
        height: y1 - y0 + 1,
    }))
}

const NEUNORM_SCRIPT: &str = r#"
import sys
import numpy as np
from NeuNorm.normalization import Normalization
from NeuNorm.roi import ROI

sample_file, ob_file, out_file = sys.argv[1:4]
o_norm = Normalization()
o_norm.load(data=np.load(sample_file))
o_norm.load(data=np.load(ob_file), data_type='ob')
if len(sys.argv) > 4:
    x0, y0, x1, y1 = (int(v) for v in sys.argv[4:8])
    o_norm.normalization(roi=ROI(x0=x0, y0=y0, x1=x1, y1=y1), force_mean_ob=True)
else:
    o_norm.normalization(force_mean_ob=True)
np.save(out_file, np.asarray(o_norm.get_normalized_data(), dtype=np.float32))
"#;

/// The normalization on a background thread: stacks handed to NeuNorm as
/// `.npy`, the sample-ROI anchor and the [0, 1] clamp applied locally on the
/// way back. Resolves to the normalized stack and a summary line.
pub struct NormJob {
    rx: Receiver<Result<(LoadedStack, String), String>>,
    /// Everything NeuNorm prints (stdout and stderr), appended live.
    pub output: Arc<std::sync::Mutex<String>>,
}

impl NormJob {
    pub fn start(stack: Arc<LoadedStack>, settings: NormSettings) -> Self {
        let (tx, rx) = channel();
        let output = Arc::new(std::sync::Mutex::new(String::new()));
        let output_thread = Arc::clone(&output);
        std::thread::spawn(move || {
            let _ = tx.send(run_normalization(&stack, &settings, &output_thread));
        });
        Self { rx, output }
    }

    pub fn poll(&mut self) -> Option<Result<(LoadedStack, String), String>> {
        self.rx.try_recv().ok()
    }
}

fn run_normalization(
    stack: &LoadedStack,
    settings: &NormSettings,
    captured: &Arc<std::sync::Mutex<String>>,
) -> Result<(LoadedStack, String), String> {
    if stack.ob.is_empty() {
        return Err("no open beam in this stack — normalization needs at least one".to_owned());
    }
    let first = stack
        .sample
        .first()
        .ok_or("no sample projections in the stack")?;
    let (w, h) = (first.width, first.height);
    if settings.needs_roi() && settings.roi.is_none() {
        return Err("the selected corrections need a ROI — select it first".to_owned());
    }
    if let Some(roi) = &settings.roi
        && (roi.x + roi.width > w || roi.y + roi.height > h)
    {
        return Err(format!(
            "the ROI ({}x{} at {},{}) does not fit the images ({w}x{h}) — reselect it",
            roi.width, roi.height, roi.x, roi.y
        ));
    }

    let dir = scratch_dir(stack, "norm")?;
    let sample_npy = dir.join("sample.npy");
    let ob_npy = dir.join("ob.npy");
    let out_npy = dir.join("normalized.npy");
    let script = dir.join("neunorm_run.py");
    let cleanup = || {
        for f in [&sample_npy, &ob_npy, &out_npy, &script] {
            let _ = std::fs::remove_file(f);
        }
        let _ = std::fs::remove_dir(&dir);
    };
    let run = || -> Result<Vec<f32>, String> {
        let sample_refs: Vec<&Projection> = stack.sample.iter().collect();
        write_npy_stack(&sample_npy, &sample_refs)?;
        let ob_refs: Vec<&Projection> = stack.ob.iter().collect();
        write_npy_stack(&ob_npy, &ob_refs)?;
        std::fs::write(&script, NEUNORM_SCRIPT)
            .map_err(|e| format!("write {}: {e}", script.display()))?;
        let mut args: Vec<String> = vec![
            script.display().to_string(),
            sample_npy.display().to_string(),
            ob_npy.display().to_string(),
            out_npy.display().to_string(),
        ];
        if settings.beam_fluctuation
            && let Some(roi) = &settings.roi
        {
            for v in [roi.x, roi.y, roi.x + roi.width - 1, roi.y + roi.height - 1] {
                args.push(v.to_string());
            }
        }
        // The exact command, in the log file and at the top of the captured
        // terminal output.
        let cmdline = format!("{PYTHON} {}", args.join(" "));
        crate::logger::log(format!("NeuNorm command: {cmdline}"));
        crate::recon_run::append_output(captured, &format!("$ {cmdline}"));
        let mut cmd = std::process::Command::new(PYTHON);
        cmd.args(&args);
        use std::io::BufRead;
        let mut child = cmd
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot launch {PYTHON}: {e}"))?;
        let stdout = child.stdout.take().expect("piped stdout");
        let stdout_buffer = Arc::clone(captured);
        let stdout_reader = std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout).lines().map_while(Result::ok) {
                crate::recon_run::append_output(&stdout_buffer, &line);
            }
        });
        let stderr = child.stderr.take().expect("piped stderr");
        let stderr_buffer = Arc::clone(captured);
        let stderr_reader = std::thread::spawn(move || {
            let mut text = String::new();
            for line in std::io::BufReader::new(stderr).lines().map_while(Result::ok) {
                crate::recon_run::append_output(&stderr_buffer, &line);
                text.push_str(&line);
                text.push('\n');
            }
            text
        });
        let status = child
            .wait()
            .map_err(|e| format!("waiting for NeuNorm: {e}"))?;
        let _ = stdout_reader.join();
        let stderr_text = stderr_reader.join().unwrap_or_default();
        if !status.success() {
            let tail: Vec<&str> = stderr_text.trim().lines().rev().take(4).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            return Err(format!("NeuNorm failed ({status}): {}", tail.join(" | ")));
        }
        let (shape, values) = read_npy(&out_npy)?;
        if shape != [stack.sample.len(), h, w] {
            return Err(format!(
                "NeuNorm returned shape {shape:?}, expected ({}, {h}, {w})",
                stack.sample.len()
            ));
        }
        Ok(values)
    };
    let result = run();
    cleanup();
    let values = result?;

    let mut sample = Vec::with_capacity(stack.sample.len());
    for (i, p) in stack.sample.iter().enumerate() {
        let mut mean = values[i * h * w..(i + 1) * h * w].to_vec();
        // The notebook clamps the transmission to [0, 1].
        for v in &mut mean {
            *v = v.clamp(0.0, 1.0);
        }
        let sum: f64 = mean.iter().map(|v| f64::from(*v)).sum();
        sample.push(Projection {
            name: p.name.clone(),
            run_number: p.run_number,
            angle_deg: p.angle_deg,
            n_images_used: p.n_images_used,
            height: h,
            width: w,
            mean,
            total_counts: sum * p.n_images_used.max(1) as f64,
        });
    }
    let mut metadata = stack.metadata.clone();
    metadata.retain(|(name, _)| name != "normalization");
    metadata.push(("normalization".to_owned(), settings.describe()));
    metadata.sort();
    let summary = format!(
        "{} projections normalized — {}",
        sample.len(),
        settings.describe()
    );
    Ok((
        LoadedStack {
            path: stack.path.clone(),
            sample,
            ob: stack.ob.clone(),
            metadata,
            center_of_rotation: stack.center_of_rotation,
        },
        summary,
    ))
}

/// The TIFF viewer of the sibling repo.
pub const TIFF_VIEWER_BIN: &str =
    "/SNS/VENUS/shared/software/git/rust_tiff_viewer/target/release/rust_tiff_viewer";

/// The 3-D volume viewer of the sibling repo.
pub const VOLUME_VIEWER_BIN: &str =
    "/SNS/VENUS/shared/software/git/rust_3d_visualization/target/release/volume_3d_viewer";

fn write_tiff_f32(path: &Path, width: usize, height: usize, data: &[f32]) -> Result<(), String> {
    let file =
        std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut encoder = tiff::encoder::TiffEncoder::new(std::io::BufWriter::new(file))
        .map_err(|e| format!("tiff encoder {}: {e}", path.display()))?;
    encoder
        .write_image::<tiff::encoder::colortype::Gray32Float>(width as u32, height as u32, data)
        .map_err(|e| format!("write {}: {e}", path.display()))
}

/// Visualize the normalized stack in the sibling TIFF viewer: the
/// projections are written as float32 TIFFs into a scratch folder and the
/// viewer opens on its single-image view; everything is cleaned up when the
/// viewer closes.
pub struct VisualizeJob {
    rx: Receiver<Result<(), String>>,
}

impl VisualizeJob {
    pub fn start(stack: Arc<LoadedStack>) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(run_viewer(&stack));
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<(), String>> {
        self.rx.try_recv().ok()
    }
}

fn run_viewer(stack: &LoadedStack) -> Result<(), String> {
    let base = scratch_dir(stack, "view")?;
    let dir = base.join("normalized");
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let cleanup = || {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let _ = std::fs::remove_file(entry.path());
        }
        let _ = std::fs::remove_dir(&dir);
        let _ = std::fs::remove_dir(&base);
    };
    let run = || -> Result<(), String> {
        for (i, p) in stack.sample.iter().enumerate() {
            let safe: String = p
                .name
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '.' { c } else { '_' })
                .collect();
            let file = dir.join(format!("{i:04}_{safe}.tif"));
            write_tiff_f32(&file, p.width, p.height, &p.mean)?;
        }
        let output = std::process::Command::new(TIFF_VIEWER_BIN)
            .arg(&dir)
            .arg("--single-image")
            .output()
            .map_err(|e| format!("cannot launch {TIFF_VIEWER_BIN}: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
            return Err(format!("rust_tiff_viewer failed ({}): {stderr}", output.status));
        }
        Ok(())
    };
    let result = run();
    cleanup();
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_describe_and_needs_roi() {
        let mut s = NormSettings::default();
        assert!(!s.needs_roi());
        s.beam_fluctuation = true;
        assert!(s.needs_roi());
        s.roi = Some(CropRect { x: 1, y: 2, width: 3, height: 4 });
        assert!(s.describe().contains("beam fluctuation"));
        assert!(s.describe().contains("x=1, y=2, 3x4"));
    }
}
