//! Combining the images of each kept run folder into one mean projection,
//! and saving the stack (projections sorted by increasing angle) to HDF5.

use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, channel};

/// Rotation angle in degrees from a run folder name
/// `..._Ang_<deg>_<millideg>_<index>` — the two `_`-separated fields after
/// the `Ang` token, same rule as the Python pipeline (`111_246` → 111.246°).
pub fn angle_from_name(name: &str) -> Option<f64> {
    let parts: Vec<&str> = name.split('_').collect();
    let ang = parts.iter().rposition(|p| *p == "Ang")?;
    let deg: i64 = parts.get(ang + 1)?.parse().ok()?;
    let milli: i64 = parts.get(ang + 2)?.parse().ok()?;
    Some(deg as f64 + milli as f64 / 1000.0)
}

/// Which images of a run folder are combined.
#[derive(Clone, Debug)]
pub enum ImageSelection {
    /// Every image of the folder.
    All,
    /// The union of inclusive file-index ranges from the TOF Profile Viewer.
    FileIndexRanges(Vec<(usize, usize)>),
}

impl ImageSelection {
    pub fn pick<'a>(&self, images: &'a [PathBuf]) -> Vec<&'a PathBuf> {
        match self {
            ImageSelection::All => images.iter().collect(),
            ImageSelection::FileIndexRanges(ranges) => images
                .iter()
                .enumerate()
                .filter(|(i, _)| ranges.iter().any(|(lo, hi)| i >= lo && i <= hi))
                .map(|(_, p)| p)
                .collect(),
        }
    }

    pub fn describe(&self) -> String {
        match self {
            ImageSelection::All => "all images".to_owned(),
            ImageSelection::FileIndexRanges(ranges) => {
                let parts: Vec<String> = ranges
                    .iter()
                    .map(|(lo, hi)| format!("{lo}–{hi}"))
                    .collect();
                format!("file indices {}", parts.join(", "))
            }
        }
    }
}

/// One run to combine: its (sorted) images plus the identity carried into
/// the output. For white beam a run is a single image file. `angle_deg`
/// overrides the angle when the caller already knows it (e.g. from metadata
/// or an ASCII list); otherwise it is parsed from the name's `Ang` token.
#[derive(Clone, Debug)]
pub struct RunToCombine {
    pub name: String,
    pub run_number: Option<u32>,
    pub images: Vec<PathBuf>,
    pub angle_deg: Option<f64>,
}

/// The mean of the selected images of one run folder.
#[derive(Clone)]
pub struct Projection {
    pub name: String,
    pub run_number: Option<u32>,
    pub angle_deg: Option<f64>,
    pub n_images_used: usize,
    pub height: usize,
    pub width: usize,
    /// Row-major `height × width` mean image.
    pub mean: Vec<f32>,
    /// Total counts over the entire selected stack (statistics measure used
    /// to pick between folders sharing an angle).
    pub total_counts: f64,
}

pub struct CombineOutput {
    /// Sample projections sorted by increasing angle (unknown angles last,
    /// by name).
    pub sample: Vec<Projection>,
    pub ob: Vec<Projection>,
    /// Folders that could not be combined, with the reason.
    pub skipped: Vec<String>,
    /// How duplicate angles were resolved (merges or best-statistics picks).
    pub notes: Vec<String>,
}

/// First page of a TIFF as `(width, height, row-major f32 values)`.
pub fn read_tiff_f32(path: &Path) -> Result<(usize, usize, Vec<f32>), String> {
    use tiff::decoder::{Decoder, DecodingResult};
    let file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut decoder = Decoder::new(std::io::BufReader::new(file))
        .map_err(|e| format!("decode {}: {e}", path.display()))?;
    let (w, h) = decoder
        .dimensions()
        .map_err(|e| format!("dimensions of {}: {e}", path.display()))?;
    let (w, h) = (w as usize, h as usize);
    let data = decoder
        .read_image()
        .map_err(|e| format!("read {}: {e}", path.display()))?;
    let values: Vec<f32> = match data {
        DecodingResult::U8(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::U16(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::U32(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::U64(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::I8(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::I16(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::I32(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::I64(v) => v.into_iter().map(|x| x as f32).collect(),
        DecodingResult::F16(v) => v.into_iter().map(|x| x.to_f32()).collect(),
        DecodingResult::F32(v) => v,
        DecodingResult::F64(v) => v.into_iter().map(|x| x as f32).collect(),
    };
    // Multi-sample pixels (e.g. RGB) keep their first sample.
    let expected = w * h;
    if values.len() == expected {
        Ok((w, h, values))
    } else if expected > 0 && values.len() % expected == 0 {
        let spp = values.len() / expected;
        Ok((w, h, (0..expected).map(|i| values[i * spp]).collect()))
    } else {
        Err(format!(
            "{}: pixel count {} not compatible with {w}x{h}",
            path.display(),
            values.len()
        ))
    }
}

/// Mean of the selected images of one folder, accumulated in f64 so summing
/// thousands of 16-bit frames does not lose precision.
fn combine_run(
    run: &RunToCombine,
    selection: &ImageSelection,
    progress: &AtomicUsize,
) -> Result<Projection, String> {
    let picked = selection.pick(&run.images);
    if picked.is_empty() {
        return Err(format!("{}: no images in the selected range", run.name));
    }
    let mut sum: Vec<f64> = Vec::new();
    let (mut width, mut height) = (0usize, 0usize);
    for path in &picked {
        let (w, h, values) = read_tiff_f32(path)?;
        if sum.is_empty() {
            (width, height) = (w, h);
            sum = vec![0.0; w * h];
        } else if (w, h) != (width, height) {
            return Err(format!(
                "{}: image size {w}x{h} differs from {width}x{height}",
                path.display()
            ));
        }
        for (acc, v) in sum.iter_mut().zip(&values) {
            *acc += f64::from(*v);
        }
        progress.fetch_add(1, Ordering::Relaxed);
    }
    let n = picked.len();
    Ok(Projection {
        name: run.name.clone(),
        run_number: run.run_number,
        angle_deg: run.angle_deg.or_else(|| angle_from_name(&run.name)),
        n_images_used: n,
        height,
        width,
        mean: sum.iter().map(|s| (s / n as f64) as f32).collect(),
        total_counts: sum.iter().sum(),
    })
}

/// Resolve sample folders sharing the same angle: merge them into one
/// count-weighted mean when `combine` is set, otherwise keep only the folder
/// with the best statistics (highest total counts). Folders without an angle
/// are never grouped. Returns the projections plus a note per decision.
fn resolve_duplicate_angles(
    projections: Vec<Projection>,
    combine: bool,
) -> (Vec<Projection>, Vec<String>) {
    use std::collections::BTreeMap;
    let mut notes = Vec::new();
    let mut out = Vec::new();
    // Group by the exact milli-degree value the angle was parsed from.
    let mut groups: BTreeMap<i64, Vec<Projection>> = BTreeMap::new();
    for p in projections {
        match p.angle_deg {
            Some(angle) => groups
                .entry((angle * 1000.0).round() as i64)
                .or_default()
                .push(p),
            None => out.push(p),
        }
    }
    for (milli, mut group) in groups {
        if group.len() == 1 {
            out.append(&mut group);
            continue;
        }
        let angle = milli as f64 / 1000.0;
        if combine {
            let (h, w) = (group[0].height, group[0].width);
            if group.iter().any(|p| (p.height, p.width) != (h, w)) {
                notes.push(format!(
                    "angle {angle:.3}°: folders have different image sizes — kept separately"
                ));
                out.append(&mut group);
                continue;
            }
            let total_images: usize = group.iter().map(|p| p.n_images_used).sum();
            let mut mean = vec![0.0f64; h * w];
            for p in &group {
                let weight = p.n_images_used as f64;
                for (acc, v) in mean.iter_mut().zip(&p.mean) {
                    *acc += f64::from(*v) * weight;
                }
            }
            let names: Vec<&str> = group.iter().map(|p| p.name.as_str()).collect();
            notes.push(format!(
                "angle {angle:.3}°: combined {} folders ({})",
                group.len(),
                names.join(" + ")
            ));
            out.push(Projection {
                name: names.join(" + "),
                run_number: group[0].run_number,
                angle_deg: Some(angle),
                n_images_used: total_images,
                height: h,
                width: w,
                mean: mean
                    .iter()
                    .map(|s| (s / total_images as f64) as f32)
                    .collect(),
                total_counts: group.iter().map(|p| p.total_counts).sum(),
            });
        } else {
            group.sort_by(|a, b| b.total_counts.total_cmp(&a.total_counts));
            let dropped: Vec<String> = group[1..]
                .iter()
                .map(|p| format!("{} ({:.4e} counts)", p.name, p.total_counts))
                .collect();
            notes.push(format!(
                "angle {angle:.3}°: kept {} ({:.4e} counts), dropped {}",
                group[0].name,
                group[0].total_counts,
                dropped.join(", ")
            ));
            out.push(group.swap_remove(0));
        }
    }
    (out, notes)
}

/// The combine pass on background threads (one rayon task per folder);
/// `progress()` counts images read out of `total_images`.
pub struct CombineScan {
    rx: Receiver<CombineOutput>,
    progress: Arc<AtomicUsize>,
    pub total_images: usize,
}

impl CombineScan {
    pub fn start(
        sample: Vec<RunToCombine>,
        ob: Vec<RunToCombine>,
        selection: ImageSelection,
        combine_same_angle: bool,
    ) -> Self {
        let progress = Arc::new(AtomicUsize::new(0));
        let total_images = sample
            .iter()
            .chain(&ob)
            .map(|r| selection.pick(&r.images).len())
            .sum();
        let (tx, rx) = channel();
        let thread_progress = Arc::clone(&progress);
        std::thread::spawn(move || {
            let mut skipped = Vec::new();
            let mut run_all = |runs: Vec<RunToCombine>| -> Vec<Projection> {
                let results: Vec<Result<Projection, String>> = runs
                    .par_iter()
                    .map(|run| combine_run(run, &selection, &thread_progress))
                    .collect();
                let mut projections = Vec::new();
                for result in results {
                    match result {
                        Ok(p) => projections.push(p),
                        Err(e) => skipped.push(e),
                    }
                }
                projections
            };
            let sample = run_all(sample);
            let ob = run_all(ob);
            let (mut sample, notes) = resolve_duplicate_angles(sample, combine_same_angle);
            // Increasing angle; folders without an angle go last, by name.
            sample.sort_by(|a, b| match (a.angle_deg, b.angle_deg) {
                (Some(x), Some(y)) => x.total_cmp(&y),
                (Some(_), None) => std::cmp::Ordering::Less,
                (None, Some(_)) => std::cmp::Ordering::Greater,
                (None, None) => a.name.cmp(&b.name),
            });
            let _ = tx.send(CombineOutput {
                sample,
                ob,
                skipped,
                notes,
            });
        });
        Self {
            rx,
            progress,
            total_images,
        }
    }

    pub fn progress(&self) -> usize {
        self.progress.load(Ordering::Relaxed)
    }

    pub fn poll(&mut self) -> Option<CombineOutput> {
        self.rx.try_recv().ok()
    }
}

/// Everything recorded next to the data in the HDF5 file.
#[derive(Clone)]
pub struct SaveMeta {
    pub instrument: String,
    pub ipts: String,
    pub detector: String,
    pub sample_folder: String,
    pub ob_folder: String,
    pub combine_mode: String,
    pub selections_json: Option<String>,
    pub detector_offset_us: Option<f64>,
}

fn write_stack(
    file: &hdf5_metno::File,
    group_name: Option<&str>,
    projections: &[Projection],
) -> Result<(), String> {
    use hdf5_metno::types::VarLenUnicode;
    let group_owned;
    let group: &hdf5_metno::Group = match group_name {
        Some(name) => {
            group_owned = file
                .create_group(name)
                .map_err(|e| format!("create group {name}: {e}"))?;
            &group_owned
        }
        None => file,
    };
    let (h, w) = (projections[0].height, projections[0].width);
    if let Some(bad) = projections.iter().find(|p| (p.height, p.width) != (h, w)) {
        return Err(format!(
            "cannot stack: {} is {}x{} while the first projection is {h}x{w}",
            bad.name, bad.height, bad.width
        ));
    }
    let n = projections.len();
    let mut flat = Vec::with_capacity(n * h * w);
    for p in projections {
        flat.extend_from_slice(&p.mean);
    }
    let err = |name: &str, e: hdf5_metno::Error| format!("write {name}: {e}");
    group
        .new_dataset::<f32>()
        .shape((n, h, w))
        .create("projections")
        .and_then(|ds| ds.write_raw(&flat))
        .map_err(|e| err("projections", e))?;
    let angles: Vec<f64> = projections
        .iter()
        .map(|p| p.angle_deg.unwrap_or(f64::NAN))
        .collect();
    group
        .new_dataset::<f64>()
        .shape(n)
        .create("angles_deg")
        .and_then(|ds| ds.write_raw(&angles))
        .map_err(|e| err("angles_deg", e))?;
    // Radians alongside, for the reconstruction algorithms.
    let angles_rad: Vec<f64> = angles.iter().map(|a| a.to_radians()).collect();
    group
        .new_dataset::<f64>()
        .shape(n)
        .create("angles_rad")
        .and_then(|ds| ds.write_raw(&angles_rad))
        .map_err(|e| err("angles_rad", e))?;
    let runs: Vec<i64> = projections
        .iter()
        .map(|p| p.run_number.map(i64::from).unwrap_or(-1))
        .collect();
    group
        .new_dataset::<i64>()
        .shape(n)
        .create("run_numbers")
        .and_then(|ds| ds.write_raw(&runs))
        .map_err(|e| err("run_numbers", e))?;
    let used: Vec<u64> = projections.iter().map(|p| p.n_images_used as u64).collect();
    group
        .new_dataset::<u64>()
        .shape(n)
        .create("images_used")
        .and_then(|ds| ds.write_raw(&used))
        .map_err(|e| err("images_used", e))?;
    let names: Vec<VarLenUnicode> = projections
        .iter()
        .map(|p| p.name.parse().unwrap_or_default())
        .collect();
    group
        .new_dataset::<VarLenUnicode>()
        .shape(n)
        .create("folder_names")
        .and_then(|ds| ds.write_raw(&names))
        .map_err(|e| err("folder_names", e))?;
    Ok(())
}

/// Write the combined output: sample stack at the root (increasing angle),
/// OB stack under `/ob`, provenance under `/metadata`.
pub fn save_hdf5(path: &Path, output: &CombineOutput, meta: &SaveMeta) -> Result<String, String> {
    use hdf5_metno::types::VarLenUnicode;
    if output.sample.is_empty() {
        return Err("nothing to save: no combined sample projections".to_owned());
    }
    let file = hdf5_metno::File::create(path)
        .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    write_stack(&file, None, &output.sample)?;
    if !output.ob.is_empty() {
        write_stack(&file, Some("ob"), &output.ob)?;
    }
    let metadata = file
        .create_group("metadata")
        .map_err(|e| format!("create metadata group: {e}"))?;
    let put = |name: &str, value: &str| -> Result<(), String> {
        let v: VarLenUnicode = value.parse().unwrap_or_default();
        metadata
            .new_dataset::<VarLenUnicode>()
            .create(name)
            .and_then(|ds| ds.write_scalar(&v))
            .map_err(|e| format!("write metadata/{name}: {e}"))
    };
    put("method", "mean")?;
    // Routes the file when loaded again: this stage still needs the
    // pre-processing screen.
    put("processing_stage", "combined")?;
    put("instrument", &meta.instrument)?;
    put("ipts", &meta.ipts)?;
    put("detector", &meta.detector)?;
    put("sample_folder", &meta.sample_folder)?;
    put("ob_folder", &meta.ob_folder)?;
    put("combine_mode", &meta.combine_mode)?;
    if let Some(json) = &meta.selections_json {
        put("tof_selections_json", json)?;
    }
    if let Some(offset) = meta.detector_offset_us {
        metadata
            .new_dataset::<f64>()
            .create("detector_offset_us")
            .and_then(|ds| ds.write_scalar(&offset))
            .map_err(|e| format!("write metadata/detector_offset_us: {e}"))?;
    }
    Ok(format!(
        "{} — {} projections ({}x{}), {} ob",
        path.display(),
        output.sample.len(),
        output.sample[0].height,
        output.sample[0].width,
        output.ob.len()
    ))
}

/// A combined output as a pre-processing stack, without going through a
/// file — same content a save + load round trip would produce.
/// `pseudo_path` stands in for the (not saved) file: its name is shown as
/// the stack title and its folder is used for scratch space.
pub fn stack_from_output(
    output: &CombineOutput,
    meta: &SaveMeta,
    pseudo_path: PathBuf,
) -> LoadedStack {
    let mut metadata = vec![
        ("method".to_owned(), "mean".to_owned()),
        ("instrument".to_owned(), meta.instrument.clone()),
        ("ipts".to_owned(), meta.ipts.clone()),
        ("detector".to_owned(), meta.detector.clone()),
        ("sample_folder".to_owned(), meta.sample_folder.clone()),
        ("ob_folder".to_owned(), meta.ob_folder.clone()),
        ("combine_mode".to_owned(), meta.combine_mode.clone()),
    ];
    if let Some(json) = &meta.selections_json {
        metadata.push(("tof_selections_json".to_owned(), json.clone()));
    }
    if let Some(offset) = meta.detector_offset_us {
        metadata.push(("detector_offset_us".to_owned(), offset.to_string()));
    }
    metadata.sort();
    LoadedStack {
        path: pseudo_path,
        sample: output.sample.clone(),
        ob: output.ob.clone(),
        metadata,
        center_of_rotation: None,
    }
}

/// A stack loaded back from a previously saved HDF5 file.
pub struct LoadedStack {
    pub path: PathBuf,
    /// Projections in file order (increasing angle when saved by this app).
    pub sample: Vec<Projection>,
    pub ob: Vec<Projection>,
    /// `/metadata` entries as (name, value) strings, sorted by name.
    pub metadata: Vec<(String, String)>,
    /// `/center_of_rotation` (px), when the file carries one.
    pub center_of_rotation: Option<f64>,
}

fn read_stack(group: &hdf5_metno::Group, what: &str) -> Result<Vec<Projection>, String> {
    use hdf5_metno::types::VarLenUnicode;
    let err = |name: &str, e: hdf5_metno::Error| format!("read {what}/{name}: {e}");
    let ds = group
        .dataset("projections")
        .map_err(|e| err("projections", e))?;
    let shape = ds.shape();
    let [n, h, w] = shape.as_slice() else {
        return Err(format!(
            "{what}/projections has shape {shape:?}, expected (n, height, width)"
        ));
    };
    let (n, h, w) = (*n, *h, *w);
    let flat: Vec<f32> = ds.read_raw().map_err(|e| err("projections", e))?;
    let angles: Vec<f64> = group
        .dataset("angles_deg")
        .and_then(|d| d.read_raw())
        .map_err(|e| err("angles_deg", e))?;
    let runs: Vec<i64> = group
        .dataset("run_numbers")
        .and_then(|d| d.read_raw())
        .map_err(|e| err("run_numbers", e))?;
    let used: Vec<u64> = group
        .dataset("images_used")
        .and_then(|d| d.read_raw())
        .map_err(|e| err("images_used", e))?;
    let names: Vec<VarLenUnicode> = group
        .dataset("folder_names")
        .and_then(|d| d.read_raw())
        .map_err(|e| err("folder_names", e))?;
    if [angles.len(), runs.len(), used.len(), names.len()] != [n; 4] {
        return Err(format!("{what}: dataset lengths do not match {n} projections"));
    }
    Ok((0..n)
        .map(|i| {
            let mean = flat[i * h * w..(i + 1) * h * w].to_vec();
            let n_images_used = used[i] as usize;
            let sum: f64 = mean.iter().map(|v| f64::from(*v)).sum();
            Projection {
                name: names[i].as_str().to_owned(),
                run_number: u32::try_from(runs[i]).ok(),
                angle_deg: (!angles[i].is_nan()).then_some(angles[i]),
                n_images_used,
                height: h,
                width: w,
                mean,
                total_counts: sum * n_images_used.max(1) as f64,
            }
        })
        .collect())
}

/// Read back a file written by [`save_hdf5`].
pub fn load_hdf5(path: &Path) -> Result<LoadedStack, String> {
    use hdf5_metno::types::VarLenUnicode;
    let file = hdf5_metno::File::open(path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let sample = read_stack(&file, "/")?;
    if sample.is_empty() {
        return Err(format!("no projections in {}", path.display()));
    }
    let ob = match file.group("ob") {
        Ok(group) => read_stack(&group, "ob")?,
        Err(_) => Vec::new(),
    };
    let mut metadata = Vec::new();
    if let Ok(group) = file.group("metadata") {
        for name in group.member_names().unwrap_or_default() {
            let Ok(ds) = group.dataset(&name) else { continue };
            let value = ds
                .read_scalar::<VarLenUnicode>()
                .map(|v| v.as_str().to_owned())
                .or_else(|_| ds.read_scalar::<f64>().map(|v| v.to_string()));
            if let Ok(value) = value {
                metadata.push((name, value));
            }
        }
    }
    metadata.sort();
    let center_of_rotation = file
        .dataset("center_of_rotation")
        .and_then(|ds| ds.read_scalar::<f64>())
        .ok();
    Ok(LoadedStack {
        path: path.to_path_buf(),
        sample,
        ob,
        metadata,
        center_of_rotation,
    })
}

/// Write a [`LoadedStack`] back to HDF5 in the exact layout [`load_hdf5`]
/// reads, carrying its metadata plus `extra` entries (which override same-
/// named ones) — the pre-processing checkpoint file.
pub fn save_stack_hdf5(
    path: &Path,
    stack: &LoadedStack,
    center_of_rotation: Option<f64>,
    extra: &[(String, String)],
) -> Result<String, String> {
    use hdf5_metno::types::VarLenUnicode;
    if stack.sample.is_empty() {
        return Err("nothing to save: the stack has no projections".to_owned());
    }
    let file = hdf5_metno::File::create(path)
        .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    write_stack(&file, None, &stack.sample)?;
    if !stack.ob.is_empty() {
        write_stack(&file, Some("ob"), &stack.ob)?;
    }
    if let Some(cor) = center_of_rotation {
        file.new_dataset::<f64>()
            .create("center_of_rotation")
            .and_then(|ds| ds.write_scalar(&cor))
            .map_err(|e| format!("write center_of_rotation: {e}"))?;
    }
    let metadata = file
        .create_group("metadata")
        .map_err(|e| format!("create metadata group: {e}"))?;
    let mut entries: Vec<(String, String)> = stack
        .metadata
        .iter()
        .filter(|(name, _)| {
            name != "processing_stage" && !extra.iter().any(|(e, _)| e == name)
        })
        .cloned()
        .collect();
    entries.extend(extra.iter().cloned());
    // Routes the file when loaded again: a checkpoint goes straight to the
    // reconstruction evaluation.
    entries.push(("processing_stage".to_owned(), "preprocessed".to_owned()));
    entries.sort();
    for (name, value) in &entries {
        let v: VarLenUnicode = value.parse().unwrap_or_default();
        metadata
            .new_dataset::<VarLenUnicode>()
            .create(name.as_str())
            .and_then(|ds| ds.write_scalar(&v))
            .map_err(|e| format!("write metadata/{name}: {e}"))?;
    }
    Ok(format!(
        "{} — {} projections ({}x{}), {} ob",
        path.display(),
        stack.sample.len(),
        stack.sample[0].height,
        stack.sample[0].width,
        stack.ob.len()
    ))
}

/// Saving a checkpoint on a background thread.
pub struct StackSaveJob {
    rx: Receiver<Result<String, String>>,
}

impl StackSaveJob {
    pub fn start(
        path: PathBuf,
        stack: Arc<LoadedStack>,
        center_of_rotation: Option<f64>,
        extra: Vec<(String, String)>,
    ) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(save_stack_hdf5(&path, &stack, center_of_rotation, &extra));
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<String, String>> {
        self.rx.try_recv().ok()
    }
}

/// Loading on a background thread (the stack can be many GB).
pub struct LoadJob {
    rx: Receiver<Result<LoadedStack, String>>,
}

impl LoadJob {
    pub fn start(path: PathBuf) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(load_hdf5(&path));
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<LoadedStack, String>> {
        self.rx.try_recv().ok()
    }
}

/// Saving on a background thread (the stack can be hundreds of MB).
pub struct SaveJob {
    rx: Receiver<Result<String, String>>,
}

impl SaveJob {
    pub fn start(path: PathBuf, output: Arc<CombineOutput>, meta: SaveMeta) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(save_hdf5(&path, &output, &meta));
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<String, String>> {
        self.rx.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn angle_parsing() {
        assert_eq!(
            angle_from_name("20260418_Run_19085_01_cell_nCT_5_260C_5_200AngsMin_Ang_000_000_1"),
            Some(0.0)
        );
        assert_eq!(
            angle_from_name("20250911_Run_12159_beetle_Ang_111_246_3"),
            Some(111.246)
        );
        assert_eq!(
            angle_from_name("20250911_Run_12158_beetle_Ang_180_000_2"),
            Some(180.0)
        );
        // OB folders have no Ang token.
        assert_eq!(angle_from_name("20250909_OB__0_832C_0_700AngsMin"), None);
        assert_eq!(angle_from_name("Ang_12"), None);
    }

    fn projection(name: &str, angle: Option<f64>, n: usize, mean: f32, counts: f64) -> Projection {
        Projection {
            name: name.to_owned(),
            run_number: None,
            angle_deg: angle,
            n_images_used: n,
            height: 1,
            width: 2,
            mean: vec![mean, mean],
            total_counts: counts,
        }
    }

    #[test]
    fn duplicate_angles_merged_with_weighted_mean() {
        let (out, notes) = resolve_duplicate_angles(
            vec![
                projection("a", Some(10.0), 30, 2.0, 100.0),
                projection("b", Some(10.0), 10, 6.0, 50.0),
                projection("c", Some(20.0), 5, 1.0, 10.0),
            ],
            true,
        );
        assert_eq!(out.len(), 2);
        let merged = out.iter().find(|p| p.angle_deg == Some(10.0)).unwrap();
        // (2.0 * 30 + 6.0 * 10) / 40 = 3.0
        assert_eq!(merged.mean, vec![3.0, 3.0]);
        assert_eq!(merged.n_images_used, 40);
        assert_eq!(merged.total_counts, 150.0);
        assert_eq!(merged.name, "a + b");
        assert_eq!(notes.len(), 1);
    }

    #[test]
    fn duplicate_angles_keep_best_statistics() {
        let (out, notes) = resolve_duplicate_angles(
            vec![
                projection("weak", Some(10.0), 30, 2.0, 100.0),
                projection("strong", Some(10.0), 10, 6.0, 500.0),
                projection("no_angle_1", None, 5, 1.0, 1.0),
                projection("no_angle_2", None, 5, 1.0, 1.0),
            ],
            false,
        );
        // Best-statistics folder kept; angle-less folders never grouped.
        assert_eq!(out.len(), 3);
        let kept = out.iter().find(|p| p.angle_deg == Some(10.0)).unwrap();
        assert_eq!(kept.name, "strong");
        assert!(notes[0].contains("kept strong"));
        assert!(notes[0].contains("dropped weak"));
    }

    #[test]
    fn save_load_roundtrip() {
        let output = CombineOutput {
            sample: vec![
                projection("p0", Some(0.0), 3, 1.5, 100.0),
                projection("p1", Some(90.5), 3, 2.5, 200.0),
                projection("p2", None, 1, 0.5, 10.0),
            ],
            ob: vec![projection("ob0", None, 2, 4.0, 50.0)],
            skipped: Vec::new(),
            notes: Vec::new(),
        };
        let meta = SaveMeta {
            instrument: "VENUS".to_owned(),
            ipts: "IPTS-1".to_owned(),
            detector: "IkonXL".to_owned(),
            sample_folder: "/a".to_owned(),
            ob_folder: "/b".to_owned(),
            combine_mode: "test".to_owned(),
            selections_json: None,
            detector_offset_us: Some(12.5),
        };
        let path = std::env::temp_dir().join(format!(
            "ct_recon_roundtrip_{}.h5",
            std::process::id()
        ));
        save_hdf5(&path, &output, &meta).unwrap();
        let loaded = load_hdf5(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.sample.len(), 3);
        assert_eq!(loaded.ob.len(), 1);
        assert_eq!(loaded.sample[0].name, "p0");
        assert_eq!(loaded.sample[1].angle_deg, Some(90.5));
        assert_eq!(loaded.sample[2].angle_deg, None);
        assert_eq!(loaded.sample[1].mean, vec![2.5, 2.5]);
        assert_eq!(loaded.sample[0].n_images_used, 3);
        assert!(loaded
            .metadata
            .iter()
            .any(|(k, v)| k == "instrument" && v == "VENUS"));
        assert!(loaded
            .metadata
            .iter()
            .any(|(k, v)| k == "detector_offset_us" && v == "12.5"));
        assert!(load_hdf5(Path::new("/nonexistent.h5")).is_err());
    }

    #[test]
    fn stack_checkpoint_roundtrip() {
        let stack = LoadedStack {
            path: PathBuf::from("/x.h5"),
            sample: vec![projection("p0", Some(0.0), 1, 0.5, 10.0)],
            ob: Vec::new(),
            metadata: vec![("normalization".to_owned(), "NeuNorm".to_owned())],
            center_of_rotation: None,
        };
        let path = std::env::temp_dir().join(format!(
            "ct_recon_checkpoint_{}.h5",
            std::process::id()
        ));
        save_stack_hdf5(
            &path,
            &stack,
            Some(261.34),
            &[("extra_note".to_owned(), "checkpoint".to_owned())],
        )
        .unwrap();
        let loaded = load_hdf5(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(loaded.sample.len(), 1);
        assert!(loaded.ob.is_empty());
        // The center of rotation survives as a numeric dataset, and the
        // angles are stored in degrees and radians.
        assert_eq!(loaded.center_of_rotation, Some(261.34));
        assert!(loaded.metadata.iter().any(|(k, _)| k == "normalization"));
        assert!(loaded
            .metadata
            .iter()
            .any(|(k, v)| k == "processing_stage" && v == "preprocessed"));
        assert!(loaded
            .metadata
            .iter()
            .any(|(k, v)| k == "extra_note" && v == "checkpoint"));
    }

    #[test]
    fn selection_picks_union_of_ranges() {
        let images: Vec<PathBuf> = (0..10).map(|i| PathBuf::from(format!("{i}.tif"))).collect();
        let all = ImageSelection::All;
        assert_eq!(all.pick(&images).len(), 10);
        let ranges = ImageSelection::FileIndexRanges(vec![(1, 3), (8, 9), (2, 4)]);
        let picked = ranges.pick(&images);
        let names: Vec<String> = picked
            .iter()
            .map(|p| p.display().to_string())
            .collect();
        assert_eq!(names, ["1.tif", "2.tif", "3.tif", "4.tif", "8.tif", "9.tif"]);
    }
}
