//! White-beam workflow data discovery: the CCD detector in use and where its
//! sample / open-beam folders live under the experiment.
//!
//! Unlike TOF (folder per projection full of TOF-binned images), a
//! white-beam sample folder holds one TIFF file per projection, with the run
//! number (`Run_<n>`) and the angle (`Ang_<deg>_<millideg>`) in the file
//! name, e.g. `20260604_Run_21775_Trex_CT_300_000s_2_800AngsMin_Ang_000_000_1.tiff`.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, channel};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WbDetector {
    IkonXl,
    Qhy,
    Scmos,
}

impl WbDetector {
    pub const ALL: [WbDetector; 3] = [WbDetector::IkonXl, WbDetector::Qhy, WbDetector::Scmos];

    pub fn label(self) -> &'static str {
        match self {
            WbDetector::IkonXl => "IkonXL",
            WbDetector::Qhy => "QHY",
            WbDetector::Scmos => "sCMOS",
        }
    }

    /// Subdirectory of `<ipts>/images` this detector writes to. Only the
    /// IkonXL location is confirmed; adjust the others once their layouts
    /// are pinned down.
    fn images_subdir(self) -> &'static str {
        match self {
            WbDetector::IkonXl => "ikonxl",
            WbDetector::Qhy => "qhy",
            WbDetector::Scmos => "scmos",
        }
    }

    /// Where the CT sample folders live, e.g.
    /// `/SNS/VENUS/IPTS-36573/images/ikonxl/raw/ct`.
    pub fn ct_root(self, ipts: &Path) -> PathBuf {
        ipts.join("images").join(self.images_subdir()).join("raw/ct")
    }

    /// Where the open-beam folders live, e.g.
    /// `/SNS/VENUS/IPTS-36573/images/ikonxl/ob`.
    pub fn ob_root(self, ipts: &Path) -> PathBuf {
        ipts.join("images").join(self.images_subdir()).join("ob")
    }
}

/// The TIFF files directly inside `dir`, sorted by name, split into the
/// highest revision of each projection (used) and the older revisions of
/// retaken projections (kept visible but excluded automatically).
pub fn tiff_files_in(dir: &Path) -> Result<(Vec<PathBuf>, Vec<PathBuf>), String> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("cannot list {}: {e}", dir.display()))?;
    let files: Vec<PathBuf> = entries
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_file() && crate::tof::is_image(p))
        .collect();
    Ok(keep_only_highest_revision(files))
}

/// The trailing revision number of a file stem (`..._R2` → 2, none → 0) and
/// the stem without it — retakes of the same projection share that base.
fn revision_of(stem: &str) -> (String, u32) {
    if let Some((base, last)) = stem.rsplit_once('_')
        && let Some(digits) = last.strip_prefix('R')
        && !digits.is_empty()
        && digits.chars().all(|c| c.is_ascii_digit())
        && let Ok(revision) = digits.parse()
    {
        return (base.to_owned(), revision);
    }
    (stem.to_owned(), 0)
}

/// The revision base of a file: retakes of the same projection (`x.tiff`,
/// `x_R1.tiff`, `x_R2.tiff`) all map to `x`.
pub fn revision_base(path: &Path) -> String {
    let stem = path
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    revision_of(&stem).0
}

/// When a projection was re-acquired (`x.tiff`, `x_R1.tiff`, `x_R2.tiff` —
/// same run number and angle), only the highest revision is used, like the
/// Python `_keep_only_highest_R_value`. Returns `(used, superseded)`, both
/// sorted.
pub fn keep_only_highest_revision(files: Vec<PathBuf>) -> (Vec<PathBuf>, Vec<PathBuf>) {
    use std::collections::HashMap;
    let mut best: HashMap<String, (u32, PathBuf)> = HashMap::new();
    let mut superseded = Vec::new();
    for file in files {
        let stem = file
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default();
        let (base, revision) = revision_of(&stem);
        match best.get_mut(&base) {
            Some((kept, kept_file)) if *kept >= revision => superseded.push(file),
            Some((kept, kept_file)) => {
                superseded.push(std::mem::replace(kept_file, file));
                *kept = revision;
            }
            None => {
                best.insert(base, (revision, file));
            }
        }
    }
    let mut kept: Vec<PathBuf> = best.into_values().map(|(_, f)| f).collect();
    kept.sort();
    superseded.sort();
    (kept, superseded)
}

/// The three ways to obtain the projection angle of each image (same options
/// as the Python `how_to_retrieve_angle_value`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum AngleSource {
    NamingConvention,
    Metadata,
    AsciiFile,
}

impl AngleSource {
    pub const ALL: [AngleSource; 3] = [
        AngleSource::NamingConvention,
        AngleSource::Metadata,
        AngleSource::AsciiFile,
    ];

    pub fn label(self) -> &'static str {
        match self {
            AngleSource::NamingConvention => "setup naming convention",
            AngleSource::Metadata => "use angle value from metadata file",
            AngleSource::AsciiFile => "import list of angles from ASCII file",
        }
    }
}

/// The `_`-separated fields of a file name (no extension), with a trailing
/// revision token (`R1`, `R002`, …) dropped — the pieces offered as angle
/// (degree / decimals) candidates by the naming convention.
pub fn name_fields(file: &Path) -> Vec<String> {
    let stem = file
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut fields: Vec<String> = stem.split('_').map(str::to_owned).collect();
    if fields
        .last()
        .is_some_and(|last| last.starts_with('R') && last[1..].chars().all(|c| c.is_ascii_digit()))
    {
        fields.pop();
    }
    fields
}

/// Default pair of angle fields: the two just before the trailing file
/// counter (`..._Ang_045_030_12` → the `045` / `030` fields).
pub fn default_angle_fields(n_fields: usize) -> Option<(usize, usize)> {
    (n_fields >= 3).then(|| (n_fields - 3, n_fields - 2))
}

/// Angle of one file per the naming convention: `float("<field_i>.<field_j>")`,
/// e.g. fields `045` and `030` give 45.030°.
pub fn angle_from_fields(file: &Path, i: usize, j: usize) -> Option<f64> {
    let fields = name_fields(file);
    let a = fields.get(i)?;
    let b = fields.get(j)?;
    format!("{a}.{b}").parse().ok()
}

/// The `<value>` of a `<label>:<value>` ASCII metadata tag.
fn tag_value(text: &str) -> Option<&str> {
    text.split(':')
        .nth(1)
        .map(|v| v.split_whitespace().next().unwrap_or(v).trim())
}

/// Angle stored in the image's TIFF metadata.
///
/// On VENUS IkonXL files, tag 65050 (`MotDeviceStr:smallrot6`) names the
/// rotation stage and the matching `MotRot<N>` tag (65061 + N, e.g.
/// `MotRot6:111.246094` at 65067) holds the angle. Tag 65039, the location
/// the Python pipeline reads, is tried as a fallback for older data.
pub fn angle_from_tiff_metadata(path: &Path) -> Result<f64, String> {
    use tiff::tags::Tag;
    let file =
        std::fs::File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut decoder = tiff::decoder::Decoder::new(std::io::BufReader::new(file))
        .map_err(|e| format!("decode {}: {e}", path.display()))?;
    let mut get = |tag: u16| decoder.get_tag_ascii_string(Tag::Unknown(tag)).ok();

    if let Some(device) = get(65050) {
        let name = tag_value(&device).unwrap_or_default();
        let digits: String = name
            .chars()
            .rev()
            .take_while(char::is_ascii_digit)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect();
        if let Ok(n) = digits.parse::<u16>()
            && let Some(rot) = get(65061 + n)
            && let Some(angle) = tag_value(&rot).and_then(|v| v.parse().ok())
        {
            return Ok(angle);
        }
    }
    if let Some(text) = get(65039)
        && let Some(angle) = tag_value(&text).and_then(|v| v.parse().ok())
    {
        return Ok(angle);
    }
    Err(format!(
        "no rotation angle in the metadata of {} (no usable MotDeviceStr/MotRot or 65039 tag)",
        path.display()
    ))
}

/// Angles from an ASCII file: whitespace/newline-separated floats, folded
/// into [0, 360).
pub fn angles_from_ascii(path: &Path) -> Result<Vec<f64>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let mut angles = Vec::new();
    for token in text.split_whitespace() {
        let value: f64 = token
            .parse()
            .map_err(|e| format!("bad angle '{token}' in {}: {e}", path.display()))?;
        angles.push(value.rem_euclid(360.0));
    }
    if angles.is_empty() {
        return Err(format!("no angle values in {}", path.display()));
    }
    Ok(angles)
}

/// Run numbers from a manual exclusion list like `1,2,5-10` →
/// {1, 2, 5, 6, 7, 8, 9, 10}. Whitespace is ignored; an empty text is an
/// empty set.
pub fn parse_run_list(text: &str) -> Result<std::collections::HashSet<u32>, String> {
    let mut runs = std::collections::HashSet::new();
    for part in text.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.split_once('-') {
            None => {
                let n: u32 = part
                    .parse()
                    .map_err(|_| format!("'{part}' is not a run number"))?;
                runs.insert(n);
            }
            Some((a, b)) => {
                let a: u32 = a
                    .trim()
                    .parse()
                    .map_err(|_| format!("'{part}' is not a run range"))?;
                let b: u32 = b
                    .trim()
                    .parse()
                    .map_err(|_| format!("'{part}' is not a run range"))?;
                if a > b {
                    return Err(format!("'{part}': range start is after its end"));
                }
                runs.extend(a..=b);
            }
        }
    }
    Ok(runs)
}

/// Integrated intensity (sum of every pixel) of one image.
#[derive(Clone, Debug)]
pub struct ImageIntensity {
    pub path: PathBuf,
    pub run_number: Option<u32>,
    pub intensity: f64,
}

/// Integrated intensities of every sample image, on background threads (each
/// image is read fully once — tens of MB per CCD frame).
pub struct IntensityScan {
    rx: Receiver<Vec<Result<ImageIntensity, String>>>,
    progress: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub total: usize,
}

impl IntensityScan {
    pub fn start(files: Vec<PathBuf>) -> Self {
        use rayon::prelude::*;
        let (tx, rx) = channel();
        let progress = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let thread_progress = std::sync::Arc::clone(&progress);
        let total = files.len();
        std::thread::spawn(move || {
            let results: Vec<Result<ImageIntensity, String>> = files
                .par_iter()
                .map(|path| {
                    let result =
                        crate::combine::read_tiff_f32(path).map(|(_, _, values)| ImageIntensity {
                            path: path.clone(),
                            run_number: crate::tof::run_number(
                                &path.file_name().unwrap_or_default().to_string_lossy(),
                            ),
                            intensity: values.iter().map(|v| f64::from(*v)).sum(),
                        });
                    thread_progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    result
                })
                .collect();
            let _ = tx.send(results);
        });
        Self {
            rx,
            progress,
            total,
        }
    }

    pub fn done(&self) -> usize {
        self.progress.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn poll(&mut self) -> Option<Vec<Result<ImageIntensity, String>>> {
        self.rx.try_recv().ok()
    }
}

/// Shortest angular distance between two angles on a circle of
/// `max_coverage` degrees (360 for a full turn, 180 for a half turn).
fn angular_distance(a: f64, b: f64, max_coverage: f64) -> f64 {
    let d = (a - b).abs();
    d.min(max_coverage - d)
}

/// Pick `n` projections maximizing angular coverage — the farthest-point
/// sampling of the Python pipeline (start from the lowest angle, greedily add
/// the candidate farthest from everything selected; wraparound at 180° or
/// 360° depending on the data) — with the angles closest to 0° and 180°
/// always kept. Returns indices into `angles`, sorted by angle.
pub fn select_coverage(angles: &[f64], n: usize) -> Vec<usize> {
    if angles.is_empty() {
        return Vec::new();
    }
    let n = n.clamp(1, angles.len());
    let mut order: Vec<usize> = (0..angles.len()).collect();
    order.sort_by(|&a, &b| angles[a].total_cmp(&angles[b]));
    if n >= angles.len() {
        return order;
    }
    let max_coverage = if angles.iter().cloned().fold(f64::MIN, f64::max) > 180.0 {
        360.0
    } else {
        180.0
    };

    // Farthest-point sampling over the sorted angles, seeded with the lowest
    // one (the angle closest to 0°).
    let mut selected: Vec<usize> = vec![order[0]];
    let mut unselected: Vec<usize> = order[1..].to_vec();
    while selected.len() < n && !unselected.is_empty() {
        let (pos, _) = unselected
            .iter()
            .enumerate()
            .map(|(pos, &candidate)| {
                let min_dist = selected
                    .iter()
                    .map(|&sel| angular_distance(angles[candidate], angles[sel], max_coverage))
                    .fold(f64::MAX, f64::min)
                    // Tie-break on the raw distance to keep the result stable.
                    - angles[candidate] * 1e-12;
                (pos, min_dist)
            })
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .expect("unselected is not empty");
        selected.push(unselected.swap_remove(pos));
    }

    // 180° must always survive the down-selection (0° is the seed).
    if let Some(&at_180) = order
        .iter()
        .min_by(|&&a, &&b| (angles[a] - 180.0).abs().total_cmp(&(angles[b] - 180.0).abs()))
        && !selected.contains(&at_180)
    {
        selected.push(at_180);
    }
    selected.sort_by(|&a, &b| angles[a].total_cmp(&angles[b]));
    selected
}

/// Reading the metadata angle of every sample image on a background thread
/// (one TIFF header per file on the network filesystem).
pub struct MetaAnglesScan {
    rx: Receiver<Vec<Result<f64, String>>>,
    progress: std::sync::Arc<std::sync::atomic::AtomicUsize>,
    pub total: usize,
}

impl MetaAnglesScan {
    pub fn start(files: Vec<PathBuf>) -> Self {
        let (tx, rx) = channel();
        let progress = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let thread_progress = std::sync::Arc::clone(&progress);
        let total = files.len();
        std::thread::spawn(move || {
            let results: Vec<Result<f64, String>> = files
                .iter()
                .map(|f| {
                    let r = angle_from_tiff_metadata(f);
                    thread_progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    r
                })
                .collect();
            let _ = tx.send(results);
        });
        Self {
            rx,
            progress,
            total,
        }
    }

    pub fn done(&self) -> usize {
        self.progress.load(std::sync::atomic::Ordering::Relaxed)
    }

    pub fn poll(&mut self) -> Option<Vec<Result<f64, String>>> {
        self.rx.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_roots() {
        let ipts = Path::new("/SNS/VENUS/IPTS-36573");
        assert_eq!(
            WbDetector::IkonXl.ct_root(ipts),
            Path::new("/SNS/VENUS/IPTS-36573/images/ikonxl/raw/ct")
        );
        assert_eq!(
            WbDetector::IkonXl.ob_root(ipts),
            Path::new("/SNS/VENUS/IPTS-36573/images/ikonxl/ob")
        );
        assert_eq!(
            WbDetector::Qhy.ct_root(ipts),
            Path::new("/SNS/VENUS/IPTS-36573/images/qhy/raw/ct")
        );
    }

    #[test]
    fn labels() {
        let labels: Vec<&str> = WbDetector::ALL.iter().map(|d| d.label()).collect();
        assert_eq!(labels, ["IkonXL", "QHY", "sCMOS"]);
    }

    #[test]
    fn naming_convention_fields_and_angle() {
        let file = Path::new(
            "/x/20260604_Run_21775_Trex_CT_300_000s_2_800AngsMin_Ang_045_030_12.tiff",
        );
        let fields = name_fields(file);
        assert_eq!(fields.last().map(String::as_str), Some("12"));
        let (i, j) = default_angle_fields(fields.len()).unwrap();
        assert_eq!((fields[i].as_str(), fields[j].as_str()), ("045", "030"));
        assert_eq!(angle_from_fields(file, i, j), Some(45.03));
        // A trailing revision token is dropped before the fields are counted.
        let rev = Path::new("/x/sample_Ang_045_030_12_R2.tiff");
        let fields = name_fields(rev);
        assert_eq!(fields.last().map(String::as_str), Some("12"));
        let (i, j) = default_angle_fields(fields.len()).unwrap();
        assert_eq!(angle_from_fields(rev, i, j), Some(45.03));
        assert_eq!(default_angle_fields(2), None);
    }

    #[test]
    fn highest_revision_wins() {
        let files: Vec<PathBuf> = [
            "a_Ang_037_329_189.tiff",
            "a_Ang_037_329_189_R1.tiff",
            "a_Ang_037_329_189_R2.tiff",
            "a_Ang_000_000_1.tiff",
            "a_Ang_180_000_2_R1.tiff",
        ]
        .iter()
        .map(|f| PathBuf::from(format!("/x/{f}")))
        .collect();
        let (kept, superseded) = keep_only_highest_revision(files);
        let names: Vec<String> = kept
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            names,
            [
                "a_Ang_000_000_1.tiff",
                "a_Ang_037_329_189_R2.tiff",
                "a_Ang_180_000_2_R1.tiff",
            ]
        );
        let old: Vec<String> = superseded
            .iter()
            .map(|p| p.file_name().unwrap().to_string_lossy().into_owned())
            .collect();
        assert_eq!(
            old,
            ["a_Ang_037_329_189.tiff", "a_Ang_037_329_189_R1.tiff"]
        );
        assert_eq!(
            revision_base(Path::new("/x/a_Ang_037_329_189_R1.tiff")),
            "a_Ang_037_329_189"
        );
    }

    #[test]
    fn run_list_parsing() {
        let runs = parse_run_list("1,2,5-10").unwrap();
        let mut sorted: Vec<u32> = runs.into_iter().collect();
        sorted.sort();
        assert_eq!(sorted, vec![1, 2, 5, 6, 7, 8, 9, 10]);
        assert_eq!(parse_run_list("").unwrap().len(), 0);
        assert_eq!(parse_run_list(" 3 , 7 - 9 ").unwrap().len(), 4);
        assert!(parse_run_list("a,b").is_err());
        assert!(parse_run_list("10-5").is_err());
    }

    #[test]
    fn coverage_selection_keeps_0_and_180() {
        let angles: Vec<f64> = (0..181).map(|i| i as f64).collect();
        let picked = select_coverage(&angles, 10);
        let values: Vec<f64> = picked.iter().map(|&i| angles[i]).collect();
        assert!(values.contains(&0.0));
        assert!(values.contains(&180.0));
        assert!(picked.len() >= 10);
        // Sorted by angle, decent spread: no two picks closer than ~5°.
        for pair in values.windows(2) {
            assert!(pair[1] > pair[0]);
            assert!(pair[1] - pair[0] >= 5.0, "{values:?}");
        }
        // Asking for everything returns everything.
        assert_eq!(select_coverage(&angles, 500).len(), angles.len());
        assert!(select_coverage(&[], 5).is_empty());
    }

    #[test]
    fn ascii_angle_list() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("ct_recon_angles_{}.txt", std::process::id()));
        std::fs::write(&path, "0.0 90.5\n181.25\n-90\n365\n").unwrap();
        let angles = angles_from_ascii(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(angles, vec![0.0, 90.5, 181.25, 270.0, 5.0]);
        assert!(angles_from_ascii(Path::new("/nonexistent.txt")).is_err());
    }
}
