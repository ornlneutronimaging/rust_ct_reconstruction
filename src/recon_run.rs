//! The full reconstruction: ship the selected slice range to Python, run
//! the chosen algorithm (in slice-range jobs with stitching for
//! svmbir/mbirjax), and export the slices as TIFFs into the output folder.

use crate::combine::LoadedStack;
use crate::crop::write_npy;
use std::io::BufRead;
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};

/// The interpreter of the pixi environment that has every reconstruction
/// library installed (svmbir, mbirjax, astra, tomopy, algotom, tifffile).
pub const RECON_PYTHON: &str =
    "/SNS/VENUS/shared/software/git/all_ct_reconstruction_development/.pixi/envs/default/bin/python";

/// svmbir's on-disk system-matrix cache locations, first writable one wins
/// (same list as the Python configuration and the svmbir optimizer).
const SVMBIR_LIB_PATHS: [&str; 3] = ["/fastdata/", "/SNS/VENUS/shared/fastdata/", "/tmp/"];

/// Everything the run needs, resolved by the UI at click time.
pub struct RunSpec {
    /// `RECON_ALGORITHMS` key (svmbir, mbirjax, astra_fbp, ...).
    pub algo_key: String,
    /// The parameters JSON: the saved `<key>_config` or the defaults.
    pub params_json: String,
    /// Inclusive slice range to reconstruct.
    pub slice_from: usize,
    pub slice_to: usize,
    /// Slice-range jobs in absolute indices, end-exclusive (a single
    /// full-range job for the algorithms that are not split).
    pub jobs: Vec<(usize, usize)>,
    /// Overlap between consecutive jobs, cut in the middle when stitching.
    pub overlap: usize,
    /// The folder the `image_XXXX.tiff` slices are written into (created,
    /// including its parents).
    pub output_folder: PathBuf,
}

const RUN_SCRIPT: &str = r#"
import json
import os
import sys

import numpy as np
import tifffile

sino_file, spec_file = sys.argv[1:3]
with open(spec_file) as f:
    spec = json.load(f)
sino = np.load(sino_file)  # (n_angles, span, width)
angles = np.array(spec["angles_rad"], dtype=np.float32)
p = spec["params"]
algo = spec["algorithm"]
jobs = spec["jobs"]  # (start, end-exclusive) pairs, relative to the range
overlap = int(spec["overlap"])
out_dir = spec["output_folder"]
offset = int(spec["slice_offset"])


def report(msg):
    print(f"PROGRESS {msg}", flush=True)


def write_slices(volume, first_rel):
    for i in range(volume.shape[0]):
        idx = offset + first_rel + i
        name = os.path.join(out_dir, f"image_{idx:04d}.tiff")
        tifffile.imwrite(name, np.asarray(volume[i], dtype=np.float32))


if algo in ("svmbir", "mbirjax"):
    n_jobs = len(jobs)
    prev_tail = None  # the previous job's overlap slices, kept for blending
    for j, (a, b) in enumerate(jobs):
        report(f"job {j + 1}/{n_jobs}: reconstructing slices {offset + a} to {offset + b - 1}")
        s = np.ascontiguousarray(sino[:, a:b, :])
        if algo == "svmbir":
            import svmbir

            w = s.shape[2]
            rec = svmbir.recon(
                sino=s,
                angles=angles,
                num_rows=w,
                num_cols=w,
                center_offset=p["center_offset"],
                max_resolutions=int(p["max_resolutions"]),
                sharpness=p["sharpness"],
                snr_db=p["snr_db"],
                positivity=bool(p["positivity"]),
                max_iterations=int(p["max_iterations"]),
                num_threads=int(spec["num_threads"]),
                verbose=0,
                svmbir_lib_path=spec.get("lib_path"),
            )
            # Same orientation as the other algorithms (the notebook's .T).
            rec = np.transpose(np.asarray(rec, dtype=np.float32), (0, 2, 1))
        else:
            import mbirjax as mj

            model = mj.ParallelBeamModel(s.shape, angles)
            model.scale_recon_shape(row_scale=p["row_scale"], col_scale=p["col_scale"])
            model.set_params(
                sharpness=p["sharpness"],
                snr_db=p["snr_db"],
                det_channel_offset=p["det_channel_offset"],
                positivity_flag=bool(p["positivity"]),
            )
            rec, _ = model.recon(s, max_iterations=int(p["max_iterations"]))
            rec = np.asarray(np.swapaxes(np.asarray(rec, dtype=np.float32), 0, 2), dtype=np.float32)
        # Stitch on the fly: each overlap is a weighted average of the two
        # jobs, the weight ramping linearly from the earlier job to the
        # later one (the middle of the overlap is a perfect mean); the last
        # `overlap` slices are kept back to blend with the next job.
        head = overlap if j > 0 else 0
        keep = rec.shape[0] - (overlap if j < n_jobs - 1 else 0)
        if j > 0:
            ramp = ((np.arange(overlap) + 1.0) / (overlap + 1.0)).astype(np.float32)
            ramp = ramp[:, None, None]
            blended = prev_tail * (1.0 - ramp) + rec[:overlap] * ramp
            report(f"job {j + 1}/{n_jobs}: blending the overlap ({offset + a} to {offset + a + overlap - 1})")
            write_slices(blended, a)
        report(f"job {j + 1}/{n_jobs}: writing slices {offset + a + head} to {offset + a + keep - 1}")
        write_slices(rec[head:keep], a + head)
        prev_tail = np.array(rec[keep:], dtype=np.float32) if j < n_jobs - 1 else None
        del rec
else:
    report(f"reconstructing {sino.shape[1]} slices with {algo}")
    if algo == "astra_fbp":
        import algotom.rec.reconstruction as rec_mod

        rec = rec_mod.astra_reconstruction(
            sino,
            p["center"],
            angles=angles,
            ratio=p["ratio"],
            method=p["method"],
            num_iter=int(p["num_iter"]),
            filter_name=p["filter_name"],
            pad=p["pad"],
            apply_log=False,
            ncore=None,
        )
    elif algo == "tomopy_fbp":
        from tomopy import recon as tomopy_recon

        rec = tomopy_recon(
            tomo=sino,
            theta=angles,
            center=p["center"],
            sinogram_order=False,
            algorithm=p["algorithm"],
            filter_name=p["filter_name"],
        )
    elif algo == "algotom_fbp":
        import algotom.rec.reconstruction as rec_mod

        rec = rec_mod.fbp_reconstruction(
            sino,
            p["center"],
            angles=angles,
            ratio=p["ratio"],
            ramp_win=None,
            filter_name=p["filter_name"],
            pad=p["pad"],
            pad_mode=p["pad_mode"],
            apply_log=False,
            gpu=bool(p["gpu"]),
            ncore=None,
        )
    elif algo == "algotom_gridrec":
        import algotom.rec.reconstruction as rec_mod

        rec = rec_mod.gridrec_reconstruction(
            sino,
            p["center"],
            angles=angles,
            ratio=p["ratio"],
            filter_name=p["filter_name"],
            apply_log=False,
            pad=int(p["pad"]),
            filter_par=p["filter_par"],
            ncore=None,
        )
    else:
        raise SystemExit(f"unknown algorithm {algo}")
    rec = np.asarray(rec, dtype=np.float32)
    if rec.ndim == 2:
        rec = rec[None, ...]
    elif algo != "tomopy_fbp":
        # algotom's 3D output keeps the slicing axis at 1 (like the input);
        # the pipeline swaps it to the front before exporting.
        rec = np.ascontiguousarray(np.swapaxes(rec, 0, 1))
    report(f"writing {rec.shape[0]} slices")
    write_slices(rec, 0)
report("done")
"#;

/// Write (or replace) the `recon_settings` JSON in the checkpoint's
/// `/metadata` group, so loading the file later restores the
/// reconstruction setup (algorithm, slice range, split, output folder).
pub fn save_recon_settings(path: &Path, json: &str) -> Result<(), String> {
    use hdf5_metno::types::VarLenUnicode;
    let file = hdf5_metno::File::open_rw(path)
        .map_err(|e| format!("cannot open {} for writing: {e}", path.display()))?;
    let metadata = match file.group("metadata") {
        Ok(group) => group,
        Err(_) => file
            .create_group("metadata")
            .map_err(|e| format!("create metadata group: {e}"))?,
    };
    if metadata.dataset("recon_settings").is_ok() {
        metadata
            .unlink("recon_settings")
            .map_err(|e| format!("replace recon_settings: {e}"))?;
    }
    let value: VarLenUnicode = json.parse().unwrap_or_default();
    metadata
        .new_dataset::<VarLenUnicode>()
        .create("recon_settings")
        .and_then(|ds| ds.write_scalar(&value))
        .map_err(|e| format!("write recon_settings: {e}"))?;
    Ok(())
}

/// The full reconstruction on a background thread; `progress` mirrors the
/// Python side's progress lines and `poll` resolves to the output folder.
pub struct RunJob {
    rx: Receiver<Result<PathBuf, String>>,
    pub progress: Arc<Mutex<String>>,
    /// Everything the Python side prints (stdout except the progress
    /// lines, plus stderr), appended live.
    pub output: Arc<Mutex<String>>,
    /// PID of the running Python process (0 = not spawned yet or done).
    child_pid: Arc<std::sync::atomic::AtomicU32>,
    cancelled: Arc<std::sync::atomic::AtomicBool>,
}

impl RunJob {
    pub fn start(stack: Arc<LoadedStack>, spec: RunSpec) -> Self {
        let (tx, rx) = channel();
        let progress = Arc::new(Mutex::new("preparing the data…".to_owned()));
        let output = Arc::new(Mutex::new(String::new()));
        let child_pid = Arc::new(std::sync::atomic::AtomicU32::new(0));
        let cancelled = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let progress_thread = Arc::clone(&progress);
        let output_thread = Arc::clone(&output);
        let pid_thread = Arc::clone(&child_pid);
        let cancelled_thread = Arc::clone(&cancelled);
        std::thread::spawn(move || {
            let _ = tx.send(run(
                &stack,
                &spec,
                &progress_thread,
                &output_thread,
                &pid_thread,
                &cancelled_thread,
            ));
        });
        Self {
            rx,
            progress,
            output,
            child_pid,
            cancelled,
        }
    }

    pub fn poll(&mut self) -> Option<Result<PathBuf, String>> {
        self.rx.try_recv().ok()
    }

    /// Interrupt the reconstruction: the Python process is terminated and
    /// the job resolves to an error explaining the stop.
    pub fn cancel(&self) {
        use std::sync::atomic::Ordering;
        self.cancelled.store(true, Ordering::SeqCst);
        let pid = self.child_pid.load(Ordering::SeqCst);
        if pid != 0 {
            let _ = std::process::Command::new("kill")
                .arg(pid.to_string())
                .status();
        }
    }

    pub fn cancelling(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
    }
}

/// Append a line to a captured terminal output, trimming the front when
/// it grows past half a megabyte.
pub(crate) fn append_output(buffer: &Arc<Mutex<String>>, line: &str) {
    let mut text = buffer.lock().unwrap();
    text.push_str(line);
    text.push('\n');
    if text.len() > 512 * 1024 {
        let cut = text.len() - 256 * 1024;
        let cut = text[cut..].find('\n').map(|i| cut + i + 1).unwrap_or(cut);
        text.drain(..cut);
    }
}

fn run(
    stack: &LoadedStack,
    spec: &RunSpec,
    progress: &Arc<Mutex<String>>,
    output: &Arc<Mutex<String>>,
    child_pid: &Arc<std::sync::atomic::AtomicU32>,
    cancelled: &Arc<std::sync::atomic::AtomicBool>,
) -> Result<PathBuf, String> {
    use std::sync::atomic::Ordering;
    let first = stack
        .sample
        .first()
        .ok_or("no projections in the stack")?;
    let (w, h, n) = (first.width, first.height, stack.sample.len());
    let (from, to) = (spec.slice_from.min(h - 1), spec.slice_to.min(h - 1));
    let span = to - from + 1;
    let angles: Vec<f64> = stack
        .sample
        .iter()
        .map(|p| p.angle_deg.map(|a| a.to_radians()))
        .collect::<Option<Vec<f64>>>()
        .ok_or("some projections carry no angle — the reconstruction needs all of them")?;

    std::fs::create_dir_all(&spec.output_folder)
        .map_err(|e| format!("create {}: {e}", spec.output_folder.display()))?;
    let scratch = spec
        .output_folder
        .join(format!(".recon_scratch_{}", std::process::id()));
    std::fs::create_dir_all(&scratch).map_err(|e| format!("create {}: {e}", scratch.display()))?;
    let sino_npy = scratch.join("sino.npy");
    let spec_file = scratch.join("spec.json");
    let script = scratch.join("recon_run.py");
    let cleanup = || {
        for f in [&sino_npy, &spec_file, &script] {
            let _ = std::fs::remove_file(f);
        }
        let _ = std::fs::remove_dir(&scratch);
    };
    let result = (|| -> Result<PathBuf, String> {
        *progress.lock().unwrap() = format!("writing the {span}-slice sinogram…");
        let mut volume = Vec::with_capacity(n * span * w);
        for p in &stack.sample {
            volume.extend_from_slice(&p.mean[from * w..(to + 1) * w]);
        }
        write_npy(&sino_npy, &[n, span, w], volume.chunks(span * w))?;
        drop(volume);

        let lib_path = SVMBIR_LIB_PATHS
            .iter()
            .find(|p| {
                let path = Path::new(p);
                path.is_dir()
                    && std::fs::metadata(path)
                        .map(|m| !m.permissions().readonly())
                        .unwrap_or(false)
            })
            .copied();
        let jobs_rel: Vec<[usize; 2]> = spec
            .jobs
            .iter()
            .map(|(a, b)| [a - from, b - from])
            .collect();
        let doc = serde_json::json!({
            "algorithm": spec.algo_key,
            "angles_rad": angles,
            "params": serde_json::from_str::<serde_json::Value>(&spec.params_json)
                .map_err(|e| format!("bad parameters JSON: {e}"))?,
            "jobs": jobs_rel,
            "overlap": spec.overlap,
            "output_folder": spec.output_folder.display().to_string(),
            "slice_offset": from,
            "num_threads": std::thread::available_parallelism()
                .map(|n| n.get().min(60))
                .unwrap_or(8),
            "lib_path": lib_path,
        });
        std::fs::write(&spec_file, doc.to_string())
            .map_err(|e| format!("write {}: {e}", spec_file.display()))?;
        std::fs::write(&script, RUN_SCRIPT)
            .map_err(|e| format!("write {}: {e}", script.display()))?;

        if cancelled.load(Ordering::SeqCst) {
            return Err("reconstruction stopped by the user".to_owned());
        }
        *progress.lock().unwrap() = "starting the reconstruction…".to_owned();
        let mut child = std::process::Command::new(RECON_PYTHON)
            .arg(&script)
            .arg(&sino_npy)
            .arg(&spec_file)
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("cannot launch {RECON_PYTHON}: {e}"))?;
        child_pid.store(child.id(), Ordering::SeqCst);
        // A stop request may have arrived between the check and the spawn.
        if cancelled.load(Ordering::SeqCst) {
            let _ = child.kill();
        }
        let stdout = child.stdout.take().expect("piped stdout");
        let progress_lines = Arc::clone(progress);
        let stdout_buffer = Arc::clone(output);
        let stdout_reader = std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout).lines().map_while(Result::ok) {
                if let Some(msg) = line.strip_prefix("PROGRESS ") {
                    *progress_lines.lock().unwrap() = msg.to_owned();
                } else {
                    append_output(&stdout_buffer, &line);
                }
            }
        });
        let stderr = child.stderr.take().expect("piped stderr");
        let stderr_buffer = Arc::clone(output);
        let stderr_reader = std::thread::spawn(move || {
            let mut text = String::new();
            for line in std::io::BufReader::new(stderr).lines().map_while(Result::ok) {
                append_output(&stderr_buffer, &line);
                text.push_str(&line);
                text.push('\n');
            }
            text
        });
        let status = child
            .wait()
            .map_err(|e| format!("waiting for the reconstruction: {e}"))?;
        child_pid.store(0, Ordering::SeqCst);
        let _ = stdout_reader.join();
        let stderr_text = stderr_reader.join().unwrap_or_default();
        if cancelled.load(Ordering::SeqCst) {
            return Err(
                "reconstruction stopped by the user — the slices already written stay \
                 in the output folder"
                    .to_owned(),
            );
        }
        if !status.success() {
            let tail: Vec<&str> = stderr_text.trim().lines().rev().take(4).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            return Err(format!(
                "reconstruction failed ({status}): {}",
                tail.join(" | ")
            ));
        }
        Ok(spec.output_folder.clone())
    })();
    cleanup();
    result
}

#[cfg(test)]
mod tests {
    use super::save_recon_settings;

    #[test]
    fn recon_settings_write_and_replace() {
        let dir = std::env::temp_dir().join(format!("recon_settings_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("checkpoint.h5");
        hdf5_metno::File::create(&path).unwrap();
        save_recon_settings(&path, "{\"algorithm\":\"svmbir\"}").unwrap();
        // Saving again must replace, not fail.
        save_recon_settings(&path, "{\"algorithm\":\"mbirjax\",\"slice_from\":5}").unwrap();
        let file = hdf5_metno::File::open(&path).unwrap();
        let value: hdf5_metno::types::VarLenUnicode = file
            .dataset("metadata/recon_settings")
            .unwrap()
            .read_scalar()
            .unwrap();
        assert!(value.as_str().contains("mbirjax"));
        assert!(value.as_str().contains("\"slice_from\":5"));
        drop(file);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
