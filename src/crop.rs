//! Cropping the loaded stack with the sibling `rust_crop_tiff` application:
//! the sample 3-D array is handed over as a NumPy `.npy` file, the tool
//! returns the rectangle drawn by the user (JSON on stdout in
//! `--called-from-app` mode), and the same crop is applied to the sample and
//! open-beam stacks in memory.

use crate::combine::{LoadedStack, Projection};
use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

/// The crop tool of the sibling repo.
pub const CROP_TIFF_BIN: &str =
    "/SNS/VENUS/shared/software/git/rust_crop_tiff/target/release/crop_tiff";

/// A rectangular crop: top-left corner and size, in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CropRect {
    pub x: usize,
    pub y: usize,
    pub width: usize,
    pub height: usize,
}

/// Indices of the projections handed to the crop tool: everything when the
/// stack has 100 projections or fewer, an evenly distributed ~10% subset
/// beyond that — enough to check the region against the whole scan without
/// writing (and having the tool load) many GB.
pub fn crop_subset_indices(n: usize, seed: u64) -> Vec<usize> {
    if n > 100 {
        subsample_indices(n, 0.10, seed)
    } else {
        (0..n).collect()
    }
}

/// Evenly distributed random subset of about `fraction` of `n` projections:
/// one random pick per stratum, so the subset covers the whole scan. At
/// least 20 (or all when fewer exist) so the crop tool's projections stay
/// meaningful.
pub fn subsample_indices(n: usize, fraction: f64, seed: u64) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let target = ((n as f64 * fraction).ceil() as usize)
        .max(20)
        .clamp(1, n);
    let mut state = seed | 1;
    let mut random_below = |m: usize| {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 33) as usize) % m.max(1)
    };
    let mut picked = Vec::with_capacity(target);
    for k in 0..target {
        let lo = k * n / target;
        let hi = (((k + 1) * n) / target).max(lo + 1);
        picked.push(lo + random_below(hi - lo));
    }
    picked.dedup();
    picked
}

/// Write an f32 array of any shape as a NumPy `.npy` file (version 1.0
/// header, float32 little-endian, C order). `planes` supplies the data in
/// chunks (e.g. one projection at a time).
pub fn write_npy<'a>(
    path: &Path,
    shape: &[usize],
    planes: impl Iterator<Item = &'a [f32]>,
) -> Result<(), String> {
    use std::io::Write;
    let file =
        std::fs::File::create(path).map_err(|e| format!("create {}: {e}", path.display()))?;
    let mut out = std::io::BufWriter::new(file);
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    let shape_txt = if dims.len() == 1 {
        format!("({},)", dims[0])
    } else {
        format!("({})", dims.join(", "))
    };
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': {shape_txt}, }}");
    // Header (magic + version + length + dict) padded to a multiple of 64
    // bytes, ending with a newline.
    let unpadded = 10 + dict.len() + 1;
    let padding = (64 - unpadded % 64) % 64;
    let header_len = (dict.len() + padding + 1) as u16;
    let mut write = |bytes: &[u8]| -> Result<(), String> {
        out.write_all(bytes)
            .map_err(|e| format!("write {}: {e}", path.display()))
    };
    write(b"\x93NUMPY\x01\x00")?;
    write(&header_len.to_le_bytes())?;
    write(dict.as_bytes())?;
    write(&vec![b' '; padding])?;
    write(b"\n")?;
    let mut total = 0usize;
    let mut buffer = Vec::new();
    for plane in planes {
        buffer.clear();
        for value in plane {
            buffer.extend_from_slice(&value.to_le_bytes());
        }
        write(&buffer)?;
        total += plane.len();
    }
    let expected: usize = shape.iter().product();
    if total != expected {
        return Err(format!(
            "internal: wrote {total} values for shape {shape:?} ({expected})"
        ));
    }
    out.flush()
        .map_err(|e| format!("flush {}: {e}", path.display()))
}

/// Write projections as a NumPy `.npy` file, shape `(n, height, width)` —
/// the 3-D-stack input form of the crop tool.
pub fn write_npy_stack(path: &Path, stack: &[&Projection]) -> Result<(), String> {
    let first = stack
        .first()
        .ok_or("cannot write an empty stack to .npy")?;
    let (h, w) = (first.height, first.width);
    if let Some(bad) = stack.iter().find(|p| (p.height, p.width) != (h, w)) {
        return Err(format!(
            "cannot write stack: {} is {}x{} while the first projection is {h}x{w}",
            bad.name, bad.height, bad.width
        ));
    }
    write_npy(
        path,
        &[stack.len(), h, w],
        stack.iter().map(|p| p.mean.as_slice()),
    )
}

/// Read a `.npy` file into `(shape, f32 values)`. Supports the common
/// little-endian numeric dtypes and C-order arrays (npy format 1.0/2.0).
pub fn read_npy(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let bad = |what: &str| format!("{}: {what}", path.display());
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" {
        return Err(bad("not a .npy file"));
    }
    let (header_start, header_len) = match bytes[6] {
        1 => (10, u16::from_le_bytes([bytes[8], bytes[9]]) as usize),
        2 => (
            12,
            u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize,
        ),
        v => return Err(bad(&format!("unsupported .npy version {v}"))),
    };
    let header = String::from_utf8_lossy(
        bytes
            .get(header_start..header_start + header_len)
            .ok_or_else(|| bad("truncated header"))?,
    )
    .into_owned();
    let field = |key: &str| -> Option<String> {
        let at = header.find(key)? + key.len();
        let rest = header[at..].trim_start().trim_start_matches(':').trim_start();
        Some(rest.to_owned())
    };
    let descr = field("'descr'")
        .and_then(|r| {
            let r = r.trim_start_matches(['\'', '"']);
            r.split(['\'', '"']).next().map(str::to_owned)
        })
        .ok_or_else(|| bad("no descr in header"))?;
    if field("'fortran_order'").is_some_and(|r| r.starts_with("True")) {
        return Err(bad("fortran-order arrays are not supported"));
    }
    let shape_txt = field("'shape'").ok_or_else(|| bad("no shape in header"))?;
    let inner = shape_txt
        .trim_start_matches('(')
        .split(')')
        .next()
        .unwrap_or_default();
    let shape: Vec<usize> = inner
        .split(',')
        .filter_map(|t| {
            let t = t.trim();
            (!t.is_empty()).then(|| t.parse().ok()).flatten()
        })
        .collect();
    let count: usize = shape.iter().product();
    let data = &bytes[header_start + header_len..];
    fn convert<const N: usize>(
        data: &[u8],
        count: usize,
        f: impl Fn([u8; N]) -> f32,
    ) -> Option<Vec<f32>> {
        (data.len() >= count * N).then(|| {
            data.chunks_exact(N)
                .take(count)
                .map(|c| f(c.try_into().expect("chunk size")))
                .collect()
        })
    }
    let values = match descr.as_str() {
        "<f4" => convert::<4>(data, count, f32::from_le_bytes),
        "<f8" => convert::<8>(data, count, |b| f64::from_le_bytes(b) as f32),
        "|u1" | "<u1" => convert::<1>(data, count, |b| b[0] as f32),
        "|i1" | "<i1" => convert::<1>(data, count, |b| b[0] as i8 as f32),
        "<u2" => convert::<2>(data, count, |b| u16::from_le_bytes(b) as f32),
        "<i2" => convert::<2>(data, count, |b| i16::from_le_bytes(b) as f32),
        "<u4" => convert::<4>(data, count, |b| u32::from_le_bytes(b) as f32),
        "<i4" => convert::<4>(data, count, |b| i32::from_le_bytes(b) as f32),
        "<u8" => convert::<8>(data, count, |b| u64::from_le_bytes(b) as f32),
        "<i8" => convert::<8>(data, count, |b| i64::from_le_bytes(b) as f32),
        other => return Err(bad(&format!("unsupported dtype '{other}'"))),
    };
    values
        .map(|v| (shape, v))
        .ok_or_else(|| bad("data shorter than the header's shape"))
}

fn crop_projection(p: &Projection, rect: &CropRect) -> Result<Projection, String> {
    if rect.x + rect.width > p.width || rect.y + rect.height > p.height {
        return Err(format!(
            "crop {},{} {}x{} does not fit in {} ({}x{})",
            rect.x, rect.y, rect.width, rect.height, p.name, p.width, p.height
        ));
    }
    let mut mean = Vec::with_capacity(rect.width * rect.height);
    for row in rect.y..rect.y + rect.height {
        let start = row * p.width + rect.x;
        mean.extend_from_slice(&p.mean[start..start + rect.width]);
    }
    let sum: f64 = mean.iter().map(|v| f64::from(*v)).sum();
    Ok(Projection {
        name: p.name.clone(),
        run_number: p.run_number,
        angle_deg: p.angle_deg,
        n_images_used: p.n_images_used,
        height: rect.height,
        width: rect.width,
        mean,
        total_counts: sum * p.n_images_used.max(1) as f64,
    })
}

/// The stack with `rect` applied to every sample projection AND every open
/// beam, with the crop recorded in the metadata.
pub fn apply_crop(stack: &LoadedStack, rect: &CropRect) -> Result<LoadedStack, String> {
    let sample: Result<Vec<Projection>, String> =
        stack.sample.iter().map(|p| crop_projection(p, rect)).collect();
    let ob: Result<Vec<Projection>, String> =
        stack.ob.iter().map(|p| crop_projection(p, rect)).collect();
    let dc: Result<Vec<Projection>, String> =
        stack.dc.iter().map(|p| crop_projection(p, rect)).collect();
    let mut metadata = stack.metadata.clone();
    metadata.retain(|(name, _)| name != "crop");
    metadata.push((
        "crop".to_owned(),
        format!("x={}, y={}, width={}, height={}", rect.x, rect.y, rect.width, rect.height),
    ));
    metadata.sort();
    Ok(LoadedStack {
        path: stack.path.clone(),
        sample: sample?,
        ob: ob?,
        dc: dc?,
        metadata,
        // The crop moves the x coordinates: a stored center of rotation is void.
        center_of_rotation: None,
    })
}

/// One crop-tool session on a background thread: write the sample stack to a
/// temp `.npy`, run the tool, apply the returned rectangle to sample and OB.
/// Resolves to `Ok(None)` when the tool was closed without saving.
pub struct CropJob {
    rx: Receiver<Result<Option<(CropRect, LoadedStack)>, String>>,
}

impl CropJob {
    pub fn start(stack: Arc<LoadedStack>, initial: Option<CropRect>) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(run_crop_tool(&stack, initial));
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<Option<(CropRect, LoadedStack)>, String>> {
        self.rx.try_recv().ok()
    }
}

fn run_crop_tool(
    stack: &LoadedStack,
    initial: Option<CropRect>,
) -> Result<Option<(CropRect, LoadedStack)>, String> {
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(1);
    let indices = crop_subset_indices(stack.sample.len(), seed);
    let subset: Vec<&Projection> = indices.iter().map(|&i| &stack.sample[i]).collect();

    // Scratch space next to the loaded HDF5 (a filesystem known to be big
    // and writable), falling back to the system temp directory.
    let base = stack
        .path
        .parent()
        .filter(|p| p.is_dir())
        .map(Path::to_path_buf)
        .unwrap_or_else(std::env::temp_dir);
    let mut dir = base.join(format!(".ct_recon_crop_{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_err() {
        dir = std::env::temp_dir().join(format!("ct_recon_crop_{}", std::process::id()));
        std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    }
    let npy = dir.join("sample_stack.npy");
    let crop_file = dir.join("crop.json");
    let cleanup = || {
        let _ = std::fs::remove_file(&npy);
        let _ = std::fs::remove_file(&crop_file);
        let _ = std::fs::remove_dir(&dir);
    };

    if let Err(first_error) = write_npy_stack(&npy, &subset) {
        // One retry in the system temp directory (e.g. read-only h5 folder).
        cleanup();
        let fallback = std::env::temp_dir().join(format!("ct_recon_crop_{}", std::process::id()));
        if std::fs::create_dir_all(&fallback).is_err()
            || write_npy_stack(&fallback.join("sample_stack.npy"), &subset).is_err()
        {
            return Err(first_error);
        }
        return run_crop_tool_in(&fallback, stack, &subset, initial);
    }
    run_crop_tool_in(&dir, stack, &subset, initial)
}

fn run_crop_tool_in(
    dir: &Path,
    stack: &LoadedStack,
    subset: &[&Projection],
    initial: Option<CropRect>,
) -> Result<Option<(CropRect, LoadedStack)>, String> {
    let npy = dir.join("sample_stack.npy");
    let crop_file = dir.join("crop.json");
    let cleanup = || {
        let _ = std::fs::remove_file(&npy);
        let _ = std::fs::remove_file(&crop_file);
        let _ = std::fs::remove_dir(dir);
    };
    let mut cmd = std::process::Command::new(CROP_TIFF_BIN);
    cmd.arg(&npy)
        .arg("--called-from-app")
        .arg("-o")
        .arg(&crop_file)
        .arg("--instructions")
        .arg(format!(
            "Draw the crop region for the CT reconstruction; it will also be applied to the \
             open beams. {}",
            if subset.len() == stack.sample.len() {
                format!("Showing all {} projections.", subset.len())
            } else {
                format!(
                    "Showing {} of {} projections (evenly sub-sampled).",
                    subset.len(),
                    stack.sample.len()
                )
            }
        ));
    if let Some(rect) = initial {
        cmd.arg("-c")
            .arg(format!("{},{},{},{}", rect.x, rect.y, rect.width, rect.height));
    }
    let result = cmd.output();
    let output = match result {
        Err(e) => {
            cleanup();
            return Err(format!("cannot launch {CROP_TIFF_BIN}: {e}"));
        }
        Ok(out) if !out.status.success() => {
            let stderr = String::from_utf8_lossy(&out.stderr).trim().to_owned();
            cleanup();
            return Err(format!("crop_tiff failed ({}): {stderr}", out.status));
        }
        Ok(out) => String::from_utf8_lossy(&out.stdout).trim().to_owned(),
    };
    cleanup();
    if output.is_empty() {
        return Ok(None);
    }
    let doc: serde_json::Value = serde_json::from_str(&output)
        .map_err(|e| format!("invalid crop JSON from crop_tiff: {e}"))?;
    let field = |name: &str| -> Result<usize, String> {
        doc.get(name)
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .ok_or_else(|| format!("no \"{name}\" in the crop JSON"))
    };
    let rect = CropRect {
        x: field("x")?,
        y: field("y")?,
        width: field("width")?,
        height: field("height")?,
    };
    let cropped = apply_crop(stack, &rect)?;
    Ok(Some((rect, cropped)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn projection(name: &str, w: usize, h: usize) -> Projection {
        Projection {
            name: name.to_owned(),
            run_number: None,
            angle_deg: Some(1.0),
            n_images_used: 1,
            height: h,
            width: w,
            mean: (0..w * h).map(|i| i as f32).collect(),
            total_counts: 0.0,
        }
    }

    #[test]
    fn crop_extracts_the_rectangle() {
        let p = projection("p", 4, 3); // rows: 0..4, 4..8, 8..12
        let rect = CropRect { x: 1, y: 1, width: 2, height: 2 };
        let cropped = crop_projection(&p, &rect).unwrap();
        assert_eq!(cropped.mean, vec![5.0, 6.0, 9.0, 10.0]);
        assert_eq!((cropped.width, cropped.height), (2, 2));
        assert_eq!(cropped.total_counts, 30.0);
        // Out of bounds is rejected.
        let bad = CropRect { x: 3, y: 0, width: 2, height: 2 };
        assert!(crop_projection(&p, &bad).is_err());
    }

    #[test]
    fn apply_crop_covers_sample_and_ob_and_records_metadata() {
        let stack = LoadedStack {
            path: std::path::PathBuf::from("/x.h5"),
            sample: vec![projection("s", 4, 3)],
            ob: vec![projection("ob", 4, 3)],
            dc: vec![projection("dc", 4, 3)],
            metadata: vec![("method".to_owned(), "mean".to_owned())],
            center_of_rotation: None,
        };
        let rect = CropRect { x: 0, y: 0, width: 2, height: 3 };
        let cropped = apply_crop(&stack, &rect).unwrap();
        assert_eq!(cropped.sample[0].width, 2);
        assert_eq!(cropped.ob[0].width, 2);
        assert_eq!(cropped.dc[0].width, 2);
        assert!(cropped
            .metadata
            .iter()
            .any(|(k, v)| k == "crop" && v.contains("width=2")));
    }

    #[test]
    fn subsampling_is_even_and_covers_the_scan() {
        let picked = subsample_indices(161, 0.10, 12345);
        assert!(picked.len() >= 20, "{}", picked.len());
        assert!(picked.len() <= 25);
        // Strictly increasing, spread across the whole range.
        for pair in picked.windows(2) {
            assert!(pair[1] > pair[0]);
        }
        assert!(picked[0] < 10);
        assert!(*picked.last().unwrap() >= 152);
        // Small stacks are passed whole.
        assert_eq!(subsample_indices(15, 0.10, 7), (0..15).collect::<Vec<_>>());
        assert!(subsample_indices(0, 0.10, 7).is_empty());
    }

    #[test]
    fn crop_tool_gets_everything_up_to_100_projections() {
        assert_eq!(crop_subset_indices(100, 42), (0..100).collect::<Vec<_>>());
        assert_eq!(crop_subset_indices(1, 42), vec![0]);
        assert!(crop_subset_indices(0, 42).is_empty());
        // Above the threshold only the ~10% subset goes to the tool.
        let picked = crop_subset_indices(400, 42);
        assert!(picked.len() >= 40, "{}", picked.len());
        assert!(picked.len() <= 50, "{}", picked.len());
    }

    #[test]
    fn npy_header_and_data() {
        let path = std::env::temp_dir().join(format!("ct_recon_npy_{}.npy", std::process::id()));
        let (a, b) = (projection("a", 3, 2), projection("b", 3, 2));
        write_npy_stack(&path, &[&a, &b]).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(&bytes[..8], b"\x93NUMPY\x01\x00");
        let header_len = u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        assert_eq!((10 + header_len) % 64, 0);
        let header = String::from_utf8_lossy(&bytes[10..10 + header_len]);
        assert!(header.contains("'shape': (2, 2, 3)"));
        assert_eq!(bytes.len(), 10 + header_len + 2 * 2 * 3 * 4);
        // First data value is 0.0f32, second 1.0f32.
        let at = 10 + header_len;
        assert_eq!(&bytes[at..at + 4], &0.0f32.to_le_bytes());
        assert_eq!(&bytes[at + 4..at + 8], &1.0f32.to_le_bytes());
    }
}
