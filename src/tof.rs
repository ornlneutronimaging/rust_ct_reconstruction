//! TOF workflow data discovery: the detector in use and the sample /
//! open-beam folder trees under the IPTS autoreduce images directory.
//!
//! On disk (e.g. VENUS): `<ipts>/shared/autoreduce/images/tpx1/raw/ct/`
//! holds one folder per sample; a sample folder holds one folder per
//! projection (angle); a projection folder holds the TOF-binned TIFF images.
//! Open-beam runs follow the same folder-of-image-folders shape under
//! `<ipts>/shared/autoreduce/images/tpx1/ob/`.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Detector {
    Tpx1UntilJuly2025,
    Tpx1FromAugust2025,
    Tpx3,
}

impl Detector {
    pub const ALL: [Detector; 3] = [
        Detector::Tpx1UntilJuly2025,
        Detector::Tpx1FromAugust2025,
        Detector::Tpx3,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Detector::Tpx1UntilJuly2025 => "tpx1 - until July 2025",
            Detector::Tpx1FromAugust2025 => "tpx1 - from August 2025",
            Detector::Tpx3 => "tpx3",
        }
    }

    /// Subdirectory of `shared/autoreduce/images` this detector writes to.
    /// The until-July-2025 tpx1 layout is assumed identical for now — adjust
    /// here once its actual folder structure is pinned down.
    fn images_subdir(self) -> &'static str {
        match self {
            Detector::Tpx1UntilJuly2025 | Detector::Tpx1FromAugust2025 => "tpx1",
            Detector::Tpx3 => "tpx3",
        }
    }

    /// Where the CT sample folders live, e.g.
    /// `/SNS/VENUS/IPTS-36202/shared/autoreduce/images/tpx1/raw/ct`.
    pub fn ct_root(self, ipts: &Path) -> PathBuf {
        ipts.join("shared/autoreduce/images")
            .join(self.images_subdir())
            .join("raw/ct")
    }

    /// Where the open-beam folders live, e.g.
    /// `/SNS/VENUS/IPTS-36202/shared/autoreduce/images/tpx1/ob`.
    pub fn ob_root(self, ipts: &Path) -> PathBuf {
        ipts.join("shared/autoreduce/images")
            .join(self.images_subdir())
            .join("ob")
    }
}

/// One folder of images (a projection run or an OB run): every image file
/// directly inside it, full paths, sorted.
#[derive(Clone, Debug)]
pub struct ImageFolder {
    pub name: String,
    pub path: PathBuf,
    pub images: Vec<PathBuf>,
}

pub fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("tif") || e.eq_ignore_ascii_case("tiff"))
}

/// Immediate subdirectories of `root`, sorted by name.
pub fn list_subdirs(root: &Path) -> Result<Vec<PathBuf>, String> {
    let dir = std::fs::read_dir(root).map_err(|e| format!("cannot list {}: {e}", root.display()))?;
    let mut subdirs: Vec<PathBuf> = dir
        .flatten()
        .filter(|item| item.file_type().is_ok_and(|t| t.is_dir()))
        .map(|item| item.path())
        .collect();
    subdirs.sort();
    Ok(subdirs)
}

fn images_in(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("cannot list {}: {e}", dir.display()))?;
    let mut images: Vec<PathBuf> = entries
        .flatten()
        .map(|item| item.path())
        .filter(|p| p.is_file() && is_image(p))
        .collect();
    images.sort();
    Ok(images)
}

fn folder_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

enum ScanMsg {
    Progress { done: usize, total: usize },
    Done(Result<Vec<ImageFolder>, String>),
}

/// A background inventory of a selected folder: one [`ImageFolder`] per
/// subfolder (per projection for a sample, per run for OBs). Runs on a
/// thread — hundreds of folders holding thousands of images each on the
/// network filesystem is too slow for the UI thread.
pub struct FolderScan {
    pub root: PathBuf,
    rx: Receiver<ScanMsg>,
    pub done: usize,
    pub total: usize,
}

impl FolderScan {
    pub fn start(root: PathBuf) -> Self {
        let (tx, rx) = channel();
        let thread_root = root.clone();
        std::thread::spawn(move || scan_thread(thread_root, tx));
        Self {
            root,
            rx,
            done: 0,
            total: 0,
        }
    }

    /// Drain progress messages; `Some` once the scan has finished.
    pub fn poll(&mut self) -> Option<Result<Vec<ImageFolder>, String>> {
        loop {
            match self.rx.try_recv() {
                Ok(ScanMsg::Progress { done, total }) => {
                    self.done = done;
                    self.total = total;
                }
                Ok(ScanMsg::Done(result)) => return Some(result),
                Err(_) => return None,
            }
        }
    }
}

fn scan_thread(root: PathBuf, tx: Sender<ScanMsg>) {
    let result = (|| {
        let subdirs = list_subdirs(&root)?;
        // A folder whose images sit directly inside (no per-projection
        // subfolders) is inventoried as a single entry.
        if subdirs.is_empty() {
            let images = images_in(&root)?;
            if images.is_empty() {
                return Ok(Vec::new());
            }
            return Ok(vec![ImageFolder {
                name: folder_name(&root),
                path: root.clone(),
                images,
            }]);
        }
        let total = subdirs.len();
        let mut folders = Vec::with_capacity(total);
        for (i, dir) in subdirs.into_iter().enumerate() {
            let images = images_in(&dir)?;
            folders.push(ImageFolder {
                name: folder_name(&dir),
                path: dir,
                images,
            });
            let _ = tx.send(ScanMsg::Progress {
                done: i + 1,
                total,
            });
        }
        Ok(folders)
    })();
    let _ = tx.send(ScanMsg::Done(result));
}

/// Lightweight description of an [`ImageFolder`] (no image list), enough for
/// the preprocessing pass.
#[derive(Clone, Debug)]
pub struct FolderSummary {
    pub name: String,
    pub path: PathBuf,
    pub n_images: usize,
}

impl ImageFolder {
    pub fn summary(&self) -> FolderSummary {
        FolderSummary {
            name: self.name.clone(),
            path: self.path.clone(),
            n_images: self.images.len(),
        }
    }
}

/// The run number embedded in a run folder name, e.g.
/// `20260418_Run_19085_01_cell_..._Ang_000_000_1` → `19085` (same
/// `Run_(\d+)` rule as the Python pipeline).
pub fn run_number(folder_name: &str) -> Option<u32> {
    let start = folder_name.find("Run_")? + "Run_".len();
    let digits: String = folder_name[start..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// `<nexus_dir>/<instrument>_<run>.nxs.h5`, e.g.
/// `/SNS/VENUS/IPTS-37118/nexus/VENUS_19085.nxs.h5`.
pub fn nexus_file_path(nexus_dir: &Path, instrument: &str, run: u32) -> PathBuf {
    nexus_dir.join(format!("{instrument}_{run}.nxs.h5"))
}

/// Proton charge of a run in Coulombs: `entry/proton_charge[0]` in the NeXus
/// file is in picocoulombs (same conversion as the Python pipeline).
pub fn read_proton_charge_c(nexus: &Path) -> Result<f64, String> {
    let file = hdf5_metno::File::open(nexus)
        .map_err(|e| format!("cannot open {}: {e}", nexus.display()))?;
    let values: Vec<f64> = file
        .dataset("entry/proton_charge")
        .and_then(|ds| ds.read_raw())
        .map_err(|e| format!("cannot read proton charge from {}: {e}", nexus.display()))?;
    values
        .first()
        .map(|pc| pc / 1e12)
        .ok_or_else(|| format!("empty proton charge in {}", nexus.display()))
}

/// One run after preprocessing: empty-folder rejection and NeXus lookup.
#[derive(Clone, Debug)]
pub struct RunInfo {
    pub name: String,
    pub path: PathBuf,
    pub n_images: usize,
    pub run_number: Option<u32>,
    pub nexus: Option<PathBuf>,
    pub proton_charge_c: Option<f64>,
    /// `true` when the folder holds no images and is excluded from processing.
    pub rejected_empty: bool,
}

#[derive(Clone, Debug, Default)]
pub struct PreprocessResult {
    pub sample: Vec<RunInfo>,
    pub ob: Vec<RunInfo>,
}

enum PreprocessMsg {
    Progress { done: usize, total: usize },
    Done(PreprocessResult),
}

/// The preprocessing pass on a background thread: reject empty run folders
/// and read each remaining run's proton charge from its NeXus file (one HDF5
/// open per run on the network filesystem).
pub struct PreprocessScan {
    rx: Receiver<PreprocessMsg>,
    pub done: usize,
    pub total: usize,
}

impl PreprocessScan {
    pub fn start(
        sample: Vec<FolderSummary>,
        ob: Vec<FolderSummary>,
        nexus_dir: PathBuf,
        instrument: String,
    ) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let total = sample.len() + ob.len();
            let mut done = 0;
            let mut inspect = |folders: Vec<FolderSummary>| -> Vec<RunInfo> {
                folders
                    .into_iter()
                    .map(|f| {
                        let rejected_empty = f.n_images == 0;
                        let run = run_number(&f.name);
                        let nexus = run.map(|n| nexus_file_path(&nexus_dir, &instrument, n));
                        let proton_charge_c = if rejected_empty {
                            None
                        } else {
                            nexus.as_deref().and_then(|p| read_proton_charge_c(p).ok())
                        };
                        done += 1;
                        let _ = tx.send(PreprocessMsg::Progress { done, total });
                        RunInfo {
                            name: f.name,
                            path: f.path,
                            n_images: f.n_images,
                            run_number: run,
                            nexus,
                            proton_charge_c,
                            rejected_empty,
                        }
                    })
                    .collect()
            };
            let result = PreprocessResult {
                sample: inspect(sample),
                ob: inspect(ob),
            };
            let _ = tx.send(PreprocessMsg::Done(result));
        });
        Self { rx, done: 0, total: 0 }
    }

    /// Drain progress messages; `Some` once the pass has finished.
    pub fn poll(&mut self) -> Option<PreprocessResult> {
        loop {
            match self.rx.try_recv() {
                Ok(PreprocessMsg::Progress { done, total }) => {
                    self.done = done;
                    self.total = total;
                }
                Ok(PreprocessMsg::Done(result)) => return Some(result),
                Err(_) => return None,
            }
        }
    }
}

/// The TOF Profile Viewer application (sibling repo), used to choose how the
/// TOF images are combined. Its `--called-from-marimo` mode prints the
/// selections as JSON on stdout and closes.
pub const TOF_PROFILE_VIEWER_BIN: &str =
    "/SNS/VENUS/shared/software/git/rust_tof_profile_viewer/target/release/tof_profile_viewer";

/// One selection exported by the TOF Profile Viewer: the same range on every
/// axis it could express it in (an axis is `None` when no TOF spectra were
/// available to convert to it).
#[derive(Clone, Debug)]
pub struct CombineRange {
    pub enabled: bool,
    pub file_index: Option<(f64, f64)>,
    pub tof_us: Option<(f64, f64)>,
    pub lambda_angstrom: Option<(f64, f64)>,
    pub energy_ev: Option<(f64, f64)>,
}

/// The parsed selections document, plus the raw JSON for the combine step.
#[derive(Clone, Debug)]
pub struct CombineSpec {
    pub folder: String,
    pub distance_m: Option<f64>,
    pub ranges: Vec<CombineRange>,
    pub has_bins: bool,
    pub raw: String,
}

pub fn parse_selections(json: &str) -> Result<CombineSpec, String> {
    let doc: serde_json::Value =
        serde_json::from_str(json).map_err(|e| format!("invalid selections JSON: {e}"))?;
    let selections = doc
        .get("selections")
        .and_then(|v| v.as_array())
        .ok_or("no \"selections\" array in the viewer output")?;
    let pair = |sel: &serde_json::Value, key: &str| -> Option<(f64, f64)> {
        let arr = sel.get(key)?.as_array()?;
        Some((arr.first()?.as_f64()?, arr.get(1)?.as_f64()?))
    };
    let ranges = selections
        .iter()
        .map(|sel| CombineRange {
            enabled: sel.get("enabled").and_then(|v| v.as_bool()).unwrap_or(true),
            file_index: pair(sel, "file_index"),
            tof_us: pair(sel, "tof_us"),
            lambda_angstrom: pair(sel, "lambda_angstrom"),
            energy_ev: pair(sel, "energy_ev"),
        })
        .collect();
    // Only a bins table with segments counts: it is what `--selections` can
    // hand back to the viewer, which rejects a document without segments.
    let has_bins = doc
        .get("bins")
        .and_then(|b| b.get("segments"))
        .and_then(|s| s.as_array())
        .is_some_and(|a| !a.is_empty());
    Ok(CombineSpec {
        folder: doc
            .get("folder")
            .and_then(|v| v.as_str())
            .unwrap_or_default()
            .to_owned(),
        distance_m: doc.get("distance_m").and_then(|v| v.as_f64()),
        ranges,
        has_bins,
        raw: json.to_owned(),
    })
}

/// A TOF Profile Viewer session on a background thread; resolves to the JSON
/// it printed on export, or `Ok(None)` when it was closed without exporting.
pub struct ViewerJob {
    rx: Receiver<Result<Option<String>, String>>,
}

impl ViewerJob {
    pub fn launch(folder: PathBuf, previous_selections: Option<&str>) -> Self {
        let (tx, rx) = channel();
        // Manual bins of the previous session are handed back via a temp
        // file (`--selections` imports only the bins table).
        let selections_file = previous_selections.and_then(|json| {
            let path = std::env::temp_dir().join(format!(
                "ct_reconstruction_tof_selections_{}.json",
                std::process::id()
            ));
            std::fs::write(&path, json).ok().map(|()| path)
        });
        std::thread::spawn(move || {
            let mut cmd = std::process::Command::new(TOF_PROFILE_VIEWER_BIN);
            cmd.arg("--called-from-marimo").arg(&folder);
            if let Some(path) = &selections_file {
                cmd.arg("--selections").arg(path);
            }
            let result = match cmd.output() {
                Err(e) => Err(format!("cannot launch {TOF_PROFILE_VIEWER_BIN}: {e}")),
                Ok(out) if !out.status.success() => Err(format!(
                    "tof_profile_viewer failed ({}): {}",
                    out.status,
                    String::from_utf8_lossy(&out.stderr).trim()
                )),
                Ok(out) => {
                    let stdout = String::from_utf8_lossy(&out.stdout).trim().to_owned();
                    Ok((!stdout.is_empty()).then_some(stdout))
                }
            };
            if let Some(path) = &selections_file {
                let _ = std::fs::remove_file(path);
            }
            let _ = tx.send(result);
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<Option<String>, String>> {
        self.rx.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_selections_document() {
        let json = r#"{
            "folder": "/data/run1",
            "distance_m": 25.0,
            "selections": [
                {"enabled": true, "file_index": [10, 20], "tof_us": [2000, 4000],
                 "lambda_angstrom": [3.9, 7.9], "energy_ev": null},
                {"enabled": false, "file_index": null, "tof_us": [100, 200],
                 "lambda_angstrom": null, "energy_ev": [0.1, 0.4]}
            ],
            "bins": {"axis": "tof",
                     "segments": [{"min": 0, "max": 10, "step": 1}]}
        }"#;
        let spec = parse_selections(json).unwrap();
        assert_eq!(spec.folder, "/data/run1");
        assert_eq!(spec.distance_m, Some(25.0));
        assert_eq!(spec.ranges.len(), 2);
        assert!(spec.ranges[0].enabled);
        assert_eq!(spec.ranges[0].file_index, Some((10.0, 20.0)));
        assert_eq!(spec.ranges[0].energy_ev, None);
        assert!(!spec.ranges[1].enabled);
        assert!(spec.has_bins);
        assert!(parse_selections("{}").is_err());
        assert!(parse_selections("not json").is_err());
        // No bins / empty segments: nothing --selections could hand back.
        let no_bins = r#"{"selections": [], "bins": {"axis": "tof", "segments": []}}"#;
        assert!(!parse_selections(no_bins).unwrap().has_bins);
        assert!(!parse_selections(r#"{"selections": []}"#).unwrap().has_bins);
    }

    #[test]
    fn run_number_from_folder_name() {
        assert_eq!(
            run_number("20260418_Run_19085_01_cell_nCT_5_260C_5_200AngsMin_Ang_000_000_1"),
            Some(19085)
        );
        assert_eq!(run_number("20250911_Run_12157_beetle_Ang_0_000_1"), Some(12157));
        assert_eq!(run_number("no_run_here"), None);
        assert_eq!(run_number("Run_"), None);
    }

    #[test]
    fn nexus_path_format() {
        assert_eq!(
            nexus_file_path(Path::new("/SNS/VENUS/IPTS-37118/nexus"), "VENUS", 19085),
            Path::new("/SNS/VENUS/IPTS-37118/nexus/VENUS_19085.nxs.h5")
        );
    }

    #[test]
    fn detector_roots() {
        let ipts = Path::new("/SNS/VENUS/IPTS-36202");
        assert_eq!(
            Detector::Tpx1FromAugust2025.ct_root(ipts),
            Path::new("/SNS/VENUS/IPTS-36202/shared/autoreduce/images/tpx1/raw/ct")
        );
        assert_eq!(
            Detector::Tpx1FromAugust2025.ob_root(ipts),
            Path::new("/SNS/VENUS/IPTS-36202/shared/autoreduce/images/tpx1/ob")
        );
        assert_eq!(
            Detector::Tpx3.ct_root(ipts),
            Path::new("/SNS/VENUS/IPTS-36202/shared/autoreduce/images/tpx3/raw/ct")
        );
    }

    #[test]
    fn image_extensions() {
        assert!(is_image(Path::new("a/b/img_00001.tif")));
        assert!(is_image(Path::new("a/b/IMG.TIFF")));
        assert!(!is_image(Path::new("a/b/notes.txt")));
        assert!(!is_image(Path::new("a/b/no_extension")));
    }
}
