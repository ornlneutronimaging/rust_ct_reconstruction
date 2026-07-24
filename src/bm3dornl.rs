//! Ring artifact removal with the sibling bm3dornl GUI
//! (<https://github.com/ornlneutronimaging/bm3dornl>): the sample stack is
//! written to an exchange HDF5 file, the GUI is launched, the user denoises
//! the data there and exports the result into the same folder (HDF5 or
//! TIFF); when the tool closes the result is read back and replaces the
//! sample stack.

use crate::combine::{LoadedStack, Projection};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::time::SystemTime;

/// The bm3dornl GUI of the sibling repo.
pub const BM3DORNL_GUI_BIN: &str =
    "/SNS/VENUS/shared/software/git/rust_bm3dornl/src/rust_core/target/release/bm3dornl-gui";

/// Name of the exchange file the GUI is asked to open.
pub const INPUT_FILE: &str = "projections_for_bm3dornl.h5";

/// Dataset holding the stack inside the exchange file.
pub const INPUT_DATASET: &str = "projections";

/// The visible exchange folder next to the checkpoint (file dialogs often
/// hide dotted names, and the user has to browse into it from the tool).
fn exchange_dir(stack: &LoadedStack) -> Result<PathBuf, String> {
    let base = stack
        .path
        .parent()
        .filter(|p| p.is_dir())
        .map(Path::to_path_buf)
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join("bm3dornl_exchange");
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    Ok(dir)
}

/// Write the sample stack as `(n, height, width)` float32 into the exchange
/// HDF5 file the GUI opens.
fn write_input(path: &Path, stack: &LoadedStack) -> Result<(usize, usize, usize), String> {
    let first = stack.sample.first().ok_or("the stack has no projections")?;
    let (h, w, n) = (first.height, first.width, stack.sample.len());
    if let Some(bad) = stack.sample.iter().find(|p| (p.height, p.width) != (h, w)) {
        return Err(format!(
            "cannot export: {} is {}x{} while the first projection is {h}x{w}",
            bad.name, bad.height, bad.width
        ));
    }
    let _ = std::fs::remove_file(path);
    let file = hdf5_metno::File::create(path)
        .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    let mut flat = Vec::with_capacity(n * h * w);
    for p in &stack.sample {
        flat.extend_from_slice(&p.mean);
    }
    file.new_dataset::<f32>()
        .shape((n, h, w))
        .create(INPUT_DATASET)
        .and_then(|ds| ds.write_raw(&flat))
        .map_err(|e| format!("write {INPUT_DATASET}: {e}"))?;
    Ok((n, h, w))
}

/// Recursively look for a float 3-D dataset of exactly `want` shape — the
/// GUI writes the export wherever the user typed (default `/data`).
fn find_stack_dataset(
    group: &hdf5_metno::Group,
    want: (usize, usize, usize),
) -> Option<hdf5_metno::Dataset> {
    for name in group.member_names().unwrap_or_default() {
        if let Ok(ds) = group.dataset(&name) {
            if ds.shape() == [want.0, want.1, want.2] {
                return Some(ds);
            }
            continue;
        }
        if let Ok(sub) = group.group(&name)
            && let Some(ds) = find_stack_dataset(&sub, want)
        {
            return Some(ds);
        }
    }
    None
}

/// Read a multi-page TIFF (the GUI's TIFF export is one file with a page
/// per slice) as flat `(pages, height, width)` f32 values.
fn read_tiff_pages(path: &Path) -> Result<(Vec<f32>, usize, usize, usize), String> {
    use tiff::decoder::{Decoder, DecodingResult};
    let file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut decoder = Decoder::new(std::io::BufReader::new(file))
        .map_err(|e| format!("decode {}: {e}", path.display()))?;
    let mut flat = Vec::new();
    let (mut w, mut h, mut n) = (0usize, 0usize, 0usize);
    loop {
        let (pw, ph) = decoder
            .dimensions()
            .map_err(|e| format!("dimensions of {}: {e}", path.display()))?;
        let (pw, ph) = (pw as usize, ph as usize);
        if n == 0 {
            (w, h) = (pw, ph);
        } else if (pw, ph) != (w, h) {
            return Err(format!(
                "{}: page {n} is {pw}x{ph} while the first page is {w}x{h}",
                path.display()
            ));
        }
        let data = decoder
            .read_image()
            .map_err(|e| format!("read {}: {e}", path.display()))?;
        match data {
            DecodingResult::F32(v) => flat.extend_from_slice(&v),
            DecodingResult::F64(v) => flat.extend(v.into_iter().map(|x| x as f32)),
            DecodingResult::U16(v) => flat.extend(v.into_iter().map(|x| x as f32)),
            DecodingResult::U8(v) => flat.extend(v.into_iter().map(|x| x as f32)),
            _ => {
                return Err(format!(
                    "{}: unsupported TIFF sample format (expected float32)",
                    path.display()
                ));
            }
        }
        n += 1;
        if !decoder.more_images() {
            break;
        }
        decoder
            .next_image()
            .map_err(|e| format!("next page of {}: {e}", path.display()))?;
    }
    Ok((flat, n, h, w))
}

/// The newest HDF5/TIFF file in the exchange folder that is not the input
/// and was written after `since`.
fn newest_result(dir: &Path, since: SystemTime) -> Option<PathBuf> {
    let mut best: Option<(SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(dir).into_iter().flatten().flatten() {
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let ext = path
            .extension()
            .map(|e| e.to_string_lossy().to_ascii_lowercase())
            .unwrap_or_default();
        if name == INPUT_FILE || !matches!(ext.as_str(), "h5" | "hdf5" | "tif" | "tiff") {
            continue;
        }
        let Ok(modified) = entry.metadata().and_then(|m| m.modified()) else {
            continue;
        };
        if modified < since {
            continue;
        }
        if best.as_ref().is_none_or(|(t, _)| modified > *t) {
            best = Some((modified, path));
        }
    }
    best.map(|(_, p)| p)
}

/// Read the exported stack back — HDF5 (any dataset of the right shape) or
/// multi-page TIFF — as flat `(n, h, w)` f32 values.
fn read_result(path: &Path, want: (usize, usize, usize)) -> Result<Vec<f32>, String> {
    let ext = path
        .extension()
        .map(|e| e.to_string_lossy().to_ascii_lowercase())
        .unwrap_or_default();
    if matches!(ext.as_str(), "tif" | "tiff") {
        let (flat, n, h, w) = read_tiff_pages(path)?;
        if (n, h, w) != want {
            return Err(format!(
                "{}: {n} pages of {w}x{h}, expected {} of {}x{} — was the whole stack \
                 exported?",
                path.display(),
                want.0,
                want.2,
                want.1
            ));
        }
        return Ok(flat);
    }
    let file = hdf5_metno::File::open(path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let ds = find_stack_dataset(&file, want).ok_or_else(|| {
        format!(
            "{}: no {}x{}x{} dataset found — was the whole stack exported?",
            path.display(),
            want.0,
            want.1,
            want.2
        )
    })?;
    ds.read_raw::<f32>()
        .map_err(|e| format!("read {}: {e}", path.display()))
}

/// The stack with its sample projections replaced by the denoised values
/// (identity, angles and OB/DC untouched) and the treatment recorded in the
/// metadata.
fn stack_with_denoised(stack: &LoadedStack, flat: &[f32], result_name: &str) -> LoadedStack {
    let mut sample = Vec::with_capacity(stack.sample.len());
    for (i, p) in stack.sample.iter().enumerate() {
        let mean = flat[i * p.height * p.width..(i + 1) * p.height * p.width].to_vec();
        let sum: f64 = mean.iter().map(|v| f64::from(*v)).sum();
        sample.push(Projection {
            name: p.name.clone(),
            run_number: p.run_number,
            angle_deg: p.angle_deg,
            n_images_used: p.n_images_used,
            height: p.height,
            width: p.width,
            mean,
            total_counts: sum * p.n_images_used.max(1) as f64,
        });
    }
    let mut metadata = stack.metadata.clone();
    metadata.retain(|(name, _)| name != "ring_artifact_removal");
    metadata.push((
        "ring_artifact_removal".to_owned(),
        format!("bm3dornl GUI ({result_name})"),
    ));
    metadata.sort();
    LoadedStack {
        path: stack.path.clone(),
        sample,
        ob: stack.ob.clone(),
        dc: stack.dc.clone(),
        metadata,
        center_of_rotation: stack.center_of_rotation,
    }
}

/// One bm3dornl session: export, launch the GUI, wait for it to close, and
/// import the newest exported file. Resolves to `Ok(None)` when the tool
/// closed without exporting anything.
pub struct Bm3dJob {
    rx: Receiver<Result<Option<(LoadedStack, String)>, String>>,
    /// Shown in the instructions while the tool is open.
    pub input_path: PathBuf,
}

impl Bm3dJob {
    pub fn start(stack: Arc<LoadedStack>) -> Result<Self, String> {
        let dir = exchange_dir(&stack)?;
        let input_path = dir.join(INPUT_FILE);
        let (tx, rx) = channel();
        let thread_input = input_path.clone();
        std::thread::spawn(move || {
            let _ = tx.send(run_tool(&stack, &dir, &thread_input));
        });
        Ok(Self { rx, input_path })
    }

    pub fn poll(&mut self) -> Option<Result<Option<(LoadedStack, String)>, String>> {
        self.rx.try_recv().ok()
    }
}

fn run_tool(
    stack: &LoadedStack,
    dir: &Path,
    input_path: &Path,
) -> Result<Option<(LoadedStack, String)>, String> {
    let want = write_input(input_path, stack)?;
    let launched = SystemTime::now();
    let output = std::process::Command::new(BM3DORNL_GUI_BIN)
        .output()
        .map_err(|e| format!("cannot launch {BM3DORNL_GUI_BIN}: {e}"))?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        return Err(format!(
            "the bm3dornl tool failed ({}): {stderr}",
            output.status
        ));
    }
    let Some(result) = newest_result(dir, launched) else {
        let _ = std::fs::remove_file(input_path);
        let _ = std::fs::remove_dir(dir);
        return Ok(None);
    };
    let flat = read_result(&result, want)?;
    let name = result
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let denoised = stack_with_denoised(stack, &flat, &name);
    let summary = format!(
        "{} projections denoised by bm3dornl (imported {name})",
        denoised.sample.len()
    );
    let _ = std::fs::remove_file(input_path);
    let _ = std::fs::remove_file(&result);
    let _ = std::fs::remove_dir(dir);
    Ok(Some((denoised, summary)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projection(name: &str, h: usize, w: usize, value: f32) -> Projection {
        Projection {
            name: name.to_owned(),
            run_number: None,
            angle_deg: Some(0.0),
            n_images_used: 1,
            height: h,
            width: w,
            mean: vec![value; h * w],
            total_counts: f64::from(value) * (h * w) as f64,
        }
    }

    #[test]
    fn export_import_roundtrip() {
        let dir = std::env::temp_dir().join(format!("bm3d_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let stack = LoadedStack {
            path: dir.join("x.h5"),
            sample: vec![projection("a", 3, 4, 1.0), projection("b", 3, 4, 2.0)],
            ob: Vec::new(),
            dc: Vec::new(),
            metadata: Vec::new(),
            center_of_rotation: None,
        };
        let input = dir.join(INPUT_FILE);
        let want = write_input(&input, &stack).unwrap();
        assert_eq!(want, (2, 3, 4));
        // The input file itself reads back as a valid result (identity).
        let flat = read_result(&input, want).unwrap();
        assert_eq!(flat.len(), 2 * 3 * 4);
        assert_eq!(flat[0], 1.0);
        assert_eq!(flat[12], 2.0);
        let denoised = stack_with_denoised(&stack, &flat, "result.h5");
        assert_eq!(denoised.sample.len(), 2);
        assert!(denoised
            .metadata
            .iter()
            .any(|(k, v)| k == "ring_artifact_removal" && v.contains("bm3dornl")));
        // A wrong shape is rejected.
        assert!(read_result(&input, (3, 3, 4)).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
