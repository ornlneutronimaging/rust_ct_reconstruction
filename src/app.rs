//! The egui/eframe application shell.
//!
//! The UI is a small state machine: the setup screen (instrument → IPTS →
//! acquisition mode) gates everything else, and the [`Session`] it produces
//! decides which workflow screen the rest of the application shows.

use crate::combine::{
    CombineOutput, CombineScan, ImageSelection, RunToCombine, SaveJob, SaveMeta,
};
use crate::config;
use crate::instrument::Instrument;
use crate::ipts::{self, IptsEntry, IptsScan};
use crate::logger;
pub use crate::session::{Mode, Session};
use crate::tof::{
    self, CombineSpec, Detector, FolderScan, ImageFolder, PreprocessResult, PreprocessScan,
    RunInfo, ViewerJob,
};
use crate::white_beam::{self, AngleSource, WbDetector};

use egui::{Align, Color32, Layout, RichText};
use sha2::{Digest, Sha256};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// How much of the log file the viewer shows, and how often it auto-refreshes.
const LOG_TAIL_BYTES: u64 = 64 * 1024;
const LOG_REFRESH_EVERY: Duration = Duration::from_secs(2);

/// Development convenience: start with the admin debug mode already on, so
/// the setup screen opens prefilled from the default config
/// (`config/config_jean.h5`). Set to `false` for production.
const START_WITH_DEBUG_ON: bool = true;

/// SHA-256 of the admin password; the plaintext is never stored, the typed
/// password is hashed and compared.
const ADMIN_PASSWORD_SHA256: &str =
    "b8b22aedc372aa891df895be9a7626e6d9ddc6d39ba85d202ca68de8c52ad782";

/// Imaging team logo, embedded in the binary and shown in the top-right
/// corner (same asset and placement as the jupyter notebooks portal).
const LOGO_BYTES: &[u8] = include_bytes!("../logos/ImagingLogo.png");
const LOGO_MAX_HEIGHT: f32 = 64.0;

fn load_logo(ctx: &egui::Context) -> Option<egui::TextureHandle> {
    let img = image::load_from_memory(LOGO_BYTES).ok()?;
    let rgba = img.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    let pixels = rgba.into_raw();
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
    Some(ctx.load_texture("imaging_logo", color_image, egui::TextureOptions::LINEAR))
}

fn password_matches(input: &str) -> bool {
    let digest = Sha256::digest(input.as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    hex == ADMIN_PASSWORD_SHA256
}

enum Screen {
    Setup,
    Workflow { session: Session, view: WorkflowView },
}

/// Mode-specific state and UI of the workflow screen.
enum WorkflowView {
    WhiteBeam(WhiteBeamView),
    Tof(TofView),
}

/// The white-beam workflow: the CCD detector drives where the sample and
/// open-beam folders are looked for; a sample folder holds one image per
/// projection, and several folders can contribute to the same dataset.
struct WhiteBeamView {
    detector: WbDetector,
    sample: MultiFolderPick,
    ob: MultiFolderPick,

    // Projection angle retrieval.
    angle_source: AngleSource,
    /// Example file the naming-convention checkboxes were built from, and
    /// which of its fields are checked (exactly 2 = degree + decimals).
    nc_example: Option<PathBuf>,
    nc_fields: Vec<String>,
    nc_checked: Vec<bool>,
    ascii_path: Option<PathBuf>,
    ascii_angles: Option<Vec<f64>>,
    ascii_error: Option<String>,
    /// Result of the metadata "test on first image" button.
    metadata_test: Option<Result<f64, String>>,
}

impl WhiteBeamView {
    fn new(session: &Session) -> Self {
        Self::with_detector(WbDetector::IkonXl, session)
    }

    fn with_detector(detector: WbDetector, session: &Session) -> Self {
        let sample = MultiFolderPick::new("sample", detector.ct_root(&session.ipts.path));
        let ob = MultiFolderPick::new("open beam", detector.ob_root(&session.ipts.path));
        logger::log(format!(
            "white beam detector: {} — sample root {} ({}), OB root {} ({})",
            detector.label(),
            sample.root.display(),
            match &sample.candidates {
                Ok(dirs) => format!("{} folders", dirs.len()),
                Err(_) => "unreadable".to_owned(),
            },
            ob.root.display(),
            match &ob.candidates {
                Ok(dirs) => format!("{} folders", dirs.len()),
                Err(_) => "unreadable".to_owned(),
            },
        ));
        Self {
            detector,
            sample,
            ob,
            angle_source: AngleSource::NamingConvention,
            nc_example: None,
            nc_fields: Vec::new(),
            nc_checked: Vec::new(),
            ascii_path: None,
            ascii_angles: None,
            ascii_error: None,
            metadata_test: None,
        }
    }

    /// First selected sample image — the naming-convention example and the
    /// metadata test subject.
    fn first_sample_image(&self) -> Option<&PathBuf> {
        self.sample
            .selected
            .iter()
            .flat_map(|(_, files)| files.iter())
            .next()
    }

    fn total_sample_images(&self) -> usize {
        self.sample.total_files()
    }
}

/// Selection of any number of folders under a fixed root; what is kept is
/// the list of TIFF files inside each selected folder.
struct MultiFolderPick {
    /// "sample" or "open beam" — used in headings and log lines.
    kind: &'static str,
    root: PathBuf,
    candidates: Result<Vec<PathBuf>, String>,
    /// Selected folders (sorted by name) and the TIFF files inside each.
    selected: Vec<(PathBuf, Vec<PathBuf>)>,
    error: Option<String>,
}

impl MultiFolderPick {
    fn new(kind: &'static str, root: PathBuf) -> Self {
        let candidates = tof::list_subdirs(&root);
        Self {
            kind,
            root,
            candidates,
            selected: Vec::new(),
            error: None,
        }
    }

    fn is_selected(&self, dir: &Path) -> bool {
        self.selected.iter().any(|(d, _)| d == dir)
    }

    fn total_files(&self) -> usize {
        self.selected.iter().map(|(_, files)| files.len()).sum()
    }

    /// Add (recording its TIFF files) or remove one folder.
    fn toggle(&mut self, dir: PathBuf) {
        if let Some(i) = self.selected.iter().position(|(d, _)| d == &dir) {
            logger::log(format!("{} folder removed: {}", self.kind, dir.display()));
            self.selected.remove(i);
            return;
        }
        match white_beam::tiff_files_in(&dir) {
            Ok(files) => {
                logger::log(format!(
                    "{} folder selected: {} ({} tiff images)",
                    self.kind,
                    dir.display(),
                    files.len()
                ));
                self.selected.push((dir, files));
                self.selected.sort_by(|(a, _), (b, _)| a.cmp(b));
                self.error = None;
            }
            Err(e) => {
                logger::error(format!("{} folder rejected: {e}", self.kind));
                self.error = Some(e);
            }
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        let heading = match self.kind {
            "sample" => "Sample (projections)",
            _ => "Open beam",
        };
        ui.label(RichText::new(heading).size(16.0).strong());
        ui.label(RichText::new(self.root.display().to_string()).weak().size(11.0));
        ui.add_space(4.0);

        let mut toggled: Option<PathBuf> = None;
        match &self.candidates {
            Err(e) => {
                ui.colored_label(Color32::LIGHT_RED, e);
            }
            Ok(dirs) if dirs.is_empty() => {
                ui.label(RichText::new("no folders found").weak());
            }
            Ok(dirs) => {
                egui::ScrollArea::vertical()
                    .id_salt((self.kind, "multi_candidates"))
                    .max_height(150.0)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        for dir in dirs {
                            let name = dir
                                .file_name()
                                .map(|n| n.to_string_lossy().into_owned())
                                .unwrap_or_else(|| dir.display().to_string());
                            let mut checked = self.is_selected(dir);
                            if ui.checkbox(&mut checked, name).changed() {
                                toggled = Some(dir.clone());
                            }
                        }
                    });
            }
        }
        if let Some(dir) = toggled {
            self.toggle(dir);
        }

        if ui.button("📂 Browse…").clicked() {
            let mut dialog = rfd::FileDialog::new()
                .set_title(format!("Select {} folder(s)", self.kind));
            let start = if self.root.is_dir() {
                Some(self.root.clone())
            } else {
                self.root.parent().filter(|p| p.is_dir()).map(PathBuf::from)
            };
            if let Some(dir) = start {
                dialog = dialog.set_directory(dir);
            }
            for dir in dialog.pick_folders().unwrap_or_default() {
                if !self.is_selected(&dir) {
                    self.toggle(dir);
                }
            }
        }

        if let Some(e) = &self.error {
            ui.colored_label(Color32::LIGHT_RED, e);
        }
        ui.add_space(6.0);
        if self.selected.is_empty() {
            ui.label(RichText::new("no folder selected yet").weak());
        } else {
            ui.label(
                RichText::new(format!(
                    "{} folder(s) — {} tiff images",
                    self.selected.len(),
                    self.total_files()
                ))
                .strong(),
            );
            for (dir, files) in &self.selected {
                let name = dir
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| dir.display().to_string());
                let text = RichText::new(format!("{name} — {} tiff", files.len())).size(12.0);
                if files.is_empty() {
                    ui.colored_label(Color32::from_rgb(240, 180, 60), text.text());
                } else {
                    ui.label(text.weak());
                }
            }
        }
    }
}

/// The TOF workflow: detector choice drives where the sample and open-beam
/// folders are looked for; selecting a folder inventories the images of each
/// of its subfolders (one per projection / OB run).
struct TofView {
    detector: Detector,
    sample: FolderPick,
    ob: FolderPick,
    /// The (sample, OB) pair whose selection summary was already written to
    /// the log, so the summary is logged (and preprocessing started) once per
    /// completed pair.
    summary_logged: Option<(PathBuf, PathBuf)>,
    /// Preprocessing pass in flight (empty-run rejection + proton charges).
    preprocess: Option<PreprocessScan>,
    preprocessed: Option<PreprocessResult>,
    /// Proton-charge selection band [min, max] in C — runs outside it are
    /// excluded from the next step. Defaults to median ±10% once
    /// preprocessing finishes.
    pc_range: Option<(f64, f64)>,
    /// Fixed value scale of the range slider, derived from the data.
    pc_bounds: (f64, f64),
    /// Which slider handle the current drag moves (`true` = upper).
    pc_drag_upper: Option<bool>,
    /// Runs manually removed from the next step (by run name), on top of the
    /// proton-charge band filter.
    removed_sample: HashSet<String>,
    removed_ob: HashSet<String>,
    /// TOF Profile Viewer session in flight (combine-range selection).
    combine_job: Option<ViewerJob>,
    combine_spec: Option<CombineSpec>,
    combine_error: Option<String>,
    /// `true`: combine every image of each run, no TOF range selection.
    combine_all: bool,
    /// `true`: folders sharing an angle are merged; `false`: only the one
    /// with the best statistics (highest total counts) is used.
    merge_same_angle: bool,
    /// Mean-combining pass in flight, and its result (shared with the save
    /// thread, the stacks can be hundreds of MB).
    process: Option<CombineScan>,
    processed: Option<std::sync::Arc<CombineOutput>>,
    save_job: Option<SaveJob>,
    save_status: Option<Result<String, String>>,
}

impl TofView {
    fn new(session: &Session) -> Self {
        Self::with_detector(Detector::Tpx1FromAugust2025, session)
    }

    fn with_detector(detector: Detector, session: &Session) -> Self {
        let sample = FolderPick::new("sample", detector.ct_root(&session.ipts.path));
        let ob = FolderPick::new("open beam", detector.ob_root(&session.ipts.path));
        logger::log(format!(
            "TOF detector: {} — sample root {} ({}), OB root {} ({})",
            detector.label(),
            sample.root.display(),
            match &sample.candidates {
                Ok(dirs) => format!("{} folders", dirs.len()),
                Err(_) => "unreadable".to_owned(),
            },
            ob.root.display(),
            match &ob.candidates {
                Ok(dirs) => format!("{} folders", dirs.len()),
                Err(_) => "unreadable".to_owned(),
            },
        ));
        Self {
            detector,
            sample,
            ob,
            summary_logged: None,
            preprocess: None,
            preprocessed: None,
            pc_range: None,
            pc_bounds: (0.0, 1.0),
            pc_drag_upper: None,
            removed_sample: HashSet::new(),
            removed_ob: HashSet::new(),
            combine_job: None,
            combine_spec: None,
            combine_error: None,
            // "Combine all images" is the default; picking TOF ranges with
            // the profile viewer is the opt-in refinement.
            combine_all: true,
            merge_same_angle: false,
            process: None,
            processed: None,
            save_job: None,
            save_status: None,
        }
    }

    /// The first sample run going to the next step (band + manual filters),
    /// used to visualize the TOF profile.
    fn first_kept_sample_run(&self) -> Option<&RunInfo> {
        let result = self.preprocessed.as_ref()?;
        let range = self.pc_range?;
        result
            .sample
            .iter()
            .find(|r| pc_in_range(r, range) && !self.removed_sample.contains(&r.name))
    }
}

/// Selection of one folder (sample or open beam) under a fixed root, plus the
/// background inventory of the images inside each of its subfolders.
struct FolderPick {
    /// "sample" or "open beam" — used in headings and log lines.
    kind: &'static str,
    root: PathBuf,
    candidates: Result<Vec<PathBuf>, String>,
    selected: Option<PathBuf>,
    scan: Option<FolderScan>,
    folders: Option<Vec<ImageFolder>>,
    error: Option<String>,
}

impl FolderPick {
    fn new(kind: &'static str, root: PathBuf) -> Self {
        let candidates = tof::list_subdirs(&root);
        Self {
            kind,
            root,
            candidates,
            selected: None,
            scan: None,
            folders: None,
            error: None,
        }
    }

    fn select(&mut self, path: PathBuf) {
        logger::log(format!("{} folder selected: {}", self.kind, path.display()));
        self.selected = Some(path.clone());
        self.folders = None;
        self.error = None;
        self.scan = Some(FolderScan::start(path));
    }

    fn poll(&mut self, ctx: &egui::Context) {
        let Some(scan) = &mut self.scan else { return };
        match scan.poll() {
            Some(Ok(folders)) => {
                let images: usize = folders.iter().map(|f| f.images.len()).sum();
                logger::log(format!(
                    "{} inventory of {}: {} folders, {} images",
                    self.kind,
                    scan.root.display(),
                    folders.len(),
                    images
                ));
                self.folders = Some(folders);
                self.scan = None;
            }
            Some(Err(e)) => {
                logger::error(format!("{} inventory failed: {e}", self.kind));
                self.error = Some(e);
                self.scan = None;
            }
            None => ctx.request_repaint_after(Duration::from_millis(200)),
        }
    }

    fn ui(&mut self, ui: &mut egui::Ui) {
        let heading = match self.kind {
            "sample" => "Sample (projections)",
            _ => "Open beam",
        };
        ui.label(RichText::new(heading).size(16.0).strong());
        ui.label(RichText::new(self.root.display().to_string()).weak().size(11.0));
        ui.add_space(4.0);

        let mut clicked = None;
        match &self.candidates {
            Err(e) => {
                ui.colored_label(Color32::LIGHT_RED, e);
            }
            Ok(dirs) if dirs.is_empty() => {
                ui.label(RichText::new("no folders found").weak());
            }
            Ok(dirs) => {
                egui::ScrollArea::vertical()
                    .id_salt((self.kind, "candidates"))
                    .max_height(150.0)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        ui.with_layout(Layout::top_down_justified(Align::Min), |ui| {
                            for dir in dirs {
                                let name = dir
                                    .file_name()
                                    .map(|n| n.to_string_lossy().into_owned())
                                    .unwrap_or_else(|| dir.display().to_string());
                                let is_selected = self.selected.as_deref() == Some(dir);
                                if ui.selectable_label(is_selected, name).clicked() {
                                    clicked = Some(dir.clone());
                                }
                            }
                        });
                    });
            }
        }
        if let Some(dir) = clicked {
            self.select(dir);
        }

        if ui.button("📂 Browse…").clicked() {
            let mut dialog = rfd::FileDialog::new().set_title(format!("Select the {} folder", self.kind));
            let start = if self.root.is_dir() {
                Some(self.root.clone())
            } else {
                self.root.parent().filter(|p| p.is_dir()).map(PathBuf::from)
            };
            if let Some(dir) = start {
                dialog = dialog.set_directory(dir);
            }
            if let Some(path) = dialog.pick_folder() {
                self.select(path);
            }
        }

        ui.add_space(6.0);
        if let Some(scan) = &self.scan {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(format!("inventorying folders… {}/{}", scan.done, scan.total));
            });
        }
        if let Some(e) = &self.error {
            ui.colored_label(Color32::LIGHT_RED, e);
        }
        if let Some(folders) = &self.folders {
            let images: usize = folders.iter().map(|f| f.images.len()).sum();
            ui.label(
                RichText::new(format!("{} folders — {} images", folders.len(), images))
                    .strong(),
            );
            egui::ScrollArea::vertical()
                .id_salt((self.kind, "inventory"))
                .max_height(180.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for folder in folders {
                        ui.label(
                            RichText::new(format!(
                                "{} — {} images",
                                folder.name,
                                folder.images.len()
                            ))
                            .weak()
                            .size(12.0),
                        );
                    }
                });
        }
    }
}

/// State of the IPTS discovery for one instrument.
enum Scan {
    Running(IptsScan),
    Done(Vec<IptsEntry>),
    Failed(String),
}

pub struct CtApp {
    screen: Screen,
    instrument: Instrument,
    /// One scan (or its cached result) per instrument, started on demand the
    /// first time the instrument is shown.
    scans: HashMap<Instrument, Scan>,
    selected: Option<IptsEntry>,
    /// Acquisition mode picked with the two large buttons; `Next` needs it.
    mode: Option<Mode>,
    filter: String,
    manual: String,
    manual_error: Option<String>,
    /// Scroll the IPTS list to the selected entry on the next frame it is
    /// shown — set when the selection comes from outside the list (manual
    /// entry, debug config), which may leave it outside the visible window.
    scroll_to_selected: bool,

    // Admin section (bottom of the setup screen).
    admin_unlocked: bool,
    admin_password: String,
    admin_error: Option<String>,
    debug_mode: bool,
    /// Config file the debug mode reads; starts at the default
    /// (`config/config_jean.h5`) and can be changed with the Browse button.
    config_path: PathBuf,
    debug_info: Option<String>,
    debug_error: Option<String>,

    /// Loaded lazily on the first frame (needs the egui context).
    logo: Option<egui::TextureHandle>,

    // Log viewer (right side panel).
    log_view_open: bool,
    log_auto_refresh: bool,
    log_text: String,
    log_last_read: Option<Instant>,
    log_error: Option<String>,
}

impl Default for CtApp {
    fn default() -> Self {
        Self::new()
    }
}

impl CtApp {
    pub fn new() -> Self {
        let mut app = Self {
            screen: Screen::Setup,
            instrument: Instrument::Venus,
            scans: HashMap::new(),
            selected: None,
            mode: None,
            filter: String::new(),
            manual: String::new(),
            manual_error: None,
            scroll_to_selected: false,
            admin_unlocked: false,
            admin_password: String::new(),
            admin_error: None,
            debug_mode: false,
            config_path: config::default_config_path(),
            debug_info: None,
            debug_error: None,
            logo: None,
            log_view_open: false,
            log_auto_refresh: true,
            log_text: String::new(),
            log_last_read: None,
            log_error: None,
        };
        if START_WITH_DEBUG_ON {
            app.debug_mode = true;
            app.enable_debug();
        }
        app
    }

    /// Make sure a scan exists for the current instrument and fold a finished
    /// background scan into its cached result.
    fn poll_scan(&mut self, ctx: &egui::Context) {
        let scan = self
            .scans
            .entry(self.instrument)
            .or_insert_with(|| Scan::Running(IptsScan::start(self.instrument)));
        if let Scan::Running(job) = scan {
            match job.try_finish() {
                Some(Ok(entries)) => {
                    logger::log(format!(
                        "IPTS scan of {}: {} readable experiments",
                        self.instrument.root(),
                        entries.len()
                    ));
                    *scan = Scan::Done(entries);
                }
                Some(Err(e)) => {
                    logger::error(format!("IPTS scan failed: {e}"));
                    *scan = Scan::Failed(e);
                }
                None => ctx.request_repaint_after(Duration::from_millis(150)),
            }
        }
    }

    /// The log viewer: a resizable right panel showing the tail of the log
    /// file, refreshed manually or automatically every couple of seconds.
    fn log_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if !self.log_view_open {
            return;
        }
        if self.log_auto_refresh {
            if self
                .log_last_read
                .is_none_or(|t| t.elapsed() >= LOG_REFRESH_EVERY)
            {
                self.refresh_log();
            }
            ctx.request_repaint_after(LOG_REFRESH_EVERY);
        } else if self.log_last_read.is_none() {
            self.refresh_log();
        }
        egui::Panel::right("log_panel")
            .resizable(true)
            .default_size(420.0)
            .min_size(280.0)
            .show(ui, |ui| {
                ui.add_space(4.0);
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Application log").strong());
                    if ui.button("🔄 Refresh").clicked() {
                        self.refresh_log();
                    }
                    ui.checkbox(&mut self.log_auto_refresh, "auto");
                });
                ui.label(
                    RichText::new(logger::log_path().display().to_string())
                        .weak()
                        .size(11.0),
                );
                ui.separator();
                if let Some(e) = &self.log_error {
                    ui.colored_label(Color32::LIGHT_RED, e);
                }
                egui::ScrollArea::both()
                    .stick_to_bottom(true)
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        ui.label(RichText::new(&self.log_text).monospace().size(11.0));
                    });
            });
    }

    fn refresh_log(&mut self) {
        self.log_last_read = Some(Instant::now());
        match logger::read_tail(LOG_TAIL_BYTES) {
            Ok(text) => {
                self.log_text = text;
                self.log_error = None;
            }
            Err(e) => self.log_error = Some(e),
        }
    }

    fn setup_ui(&mut self, ui: &mut egui::Ui) -> bool {
        top_right_bar(ui, self.logo.as_ref(), &mut self.log_view_open, LOGO_MAX_HEIGHT);
        ui.vertical_centered(|ui| {
            ui.add_space(16.0);
            ui.label(RichText::new("CT Reconstruction").size(32.0).strong());
            ui.add_space(2.0);
            ui.label(
                RichText::new("Select the instrument, the experiment, and the acquisition mode")
                    .weak(),
            );
            ui.add_space(20.0);

            self.instrument_row(ui);
            ui.add_space(20.0);
            self.ipts_section(ui);
            ui.add_space(28.0);
            self.mode_buttons(ui);
        });
        self.next_button(ui)
    }

    fn instrument_row(&mut self, ui: &mut egui::Ui) {
        const W: f32 = 170.0;
        const H: f32 = 54.0;
        const GAP: f32 = 16.0;
        ui.label(RichText::new("Instrument").size(18.0).strong());
        ui.add_space(6.0);
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = GAP;
            ui.add_space(((ui.available_width() - (2.0 * W + GAP)) / 2.0).max(0.0));
            for inst in Instrument::ALL {
                let selected = inst == self.instrument;
                let text = RichText::new(inst.name()).size(22.0).strong();
                let button = egui::Button::new(text)
                    .min_size(egui::vec2(W, H))
                    .selected(selected);
                if ui.add(button).clicked() && !selected {
                    // The IPTS list and selection belong to the previous
                    // instrument — drop them, keep its scan cached.
                    logger::log(format!("instrument selected: {}", inst.name()));
                    self.instrument = inst;
                    self.selected = None;
                    self.filter.clear();
                    self.manual_error = None;
                    // MARS is not a TOF instrument.
                    if inst == Instrument::Mars && self.mode != Some(Mode::WhiteBeam) {
                        logger::log("mode forced to White Beam (TOF not available on MARS)");
                        self.mode = Some(Mode::WhiteBeam);
                    }
                }
            }
        });
        ui.add_space(4.0);
        ui.label(RichText::new(self.instrument.description()).weak().size(13.0));
    }

    fn ipts_section(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Experiment (IPTS)").size(18.0).strong());
        ui.add_space(6.0);

        let mut clicked: Option<IptsEntry> = None;
        let mut manual_requested = false;
        let mut scrolled = false;
        let scan = self.scans.get(&self.instrument);

        ui.group(|ui| {
            ui.set_width(460.0);
            match scan {
                None | Some(Scan::Running(_)) => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label(format!("Scanning {} for readable experiments…", self.instrument.root()));
                    });
                }
                Some(Scan::Failed(e)) => {
                    ui.colored_label(Color32::LIGHT_RED, e);
                }
                Some(Scan::Done(entries)) => {
                    ui.horizontal(|ui| {
                        ui.label("Filter:");
                        ui.add(
                            egui::TextEdit::singleline(&mut self.filter)
                                .hint_text("type digits to narrow the list")
                                .desired_width(200.0),
                        );
                    });
                    ui.add_space(4.0);
                    let needle = self.filter.trim().to_ascii_uppercase();
                    let shown: Vec<&IptsEntry> = entries
                        .iter()
                        .filter(|e| needle.is_empty() || e.name.contains(&needle))
                        .collect();
                    ui.label(
                        RichText::new(format!(
                            "{} of {} experiments you can read",
                            shown.len(),
                            entries.len()
                        ))
                        .weak()
                        .size(12.0),
                    );
                    ui.add_space(2.0);
                    egui::ScrollArea::vertical()
                        .max_height(190.0)
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            ui.with_layout(Layout::top_down_justified(Align::Min), |ui| {
                                for entry in shown {
                                    let is_selected =
                                        self.selected.as_ref().is_some_and(|s| s.name == entry.name);
                                    let response = ui.selectable_label(is_selected, &entry.name);
                                    if is_selected && self.scroll_to_selected {
                                        response.scroll_to_me(Some(Align::Center));
                                        scrolled = true;
                                    }
                                    if response.clicked() {
                                        clicked = Some(entry.clone());
                                    }
                                }
                            });
                        });
                }
            }

            ui.add_space(8.0);
            ui.separator();
            ui.horizontal(|ui| {
                ui.label("Manual entry:");
                let edit = ui.add(
                    egui::TextEdit::singleline(&mut self.manual)
                        .hint_text("e.g. IPTS-36967 or 36967")
                        .desired_width(180.0),
                );
                let entered = edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                if ui.button("Use").clicked() || entered {
                    manual_requested = true;
                }
            });
        });

        if scrolled {
            self.scroll_to_selected = false;
        }
        if let Some(entry) = clicked {
            logger::log(format!("IPTS selected: {} ({})", entry.name, entry.path.display()));
            self.selected = Some(entry);
            self.manual_error = None;
        }
        if manual_requested {
            match ipts::manual_entry(self.instrument, &self.manual) {
                Ok(entry) => {
                    logger::log(format!(
                        "IPTS selected manually: {} ({})",
                        entry.name,
                        entry.path.display()
                    ));
                    self.selected = Some(entry);
                    self.manual_error = None;
                    self.scroll_to_selected = true;
                }
                Err(e) => {
                    logger::error(format!("manual IPTS entry rejected: {e}"));
                    self.manual_error = Some(e);
                }
            }
        }

        if let Some(e) = &self.manual_error {
            ui.add_space(6.0);
            ui.colored_label(Color32::LIGHT_RED, e);
        }
        ui.add_space(8.0);
        match &self.selected {
            Some(entry) => {
                ui.label(
                    RichText::new(format!("Selected: {}  ({})", entry.name, entry.path.display()))
                        .size(15.0)
                        .strong(),
                );
            }
            None => {
                ui.label(RichText::new("No experiment selected yet").weak());
            }
        }
    }

    /// The two large acquisition-mode buttons; clicking selects the mode, it
    /// does not navigate — that is what the `Next` button is for.
    fn mode_buttons(&mut self, ui: &mut egui::Ui) {
        const W: f32 = 260.0;
        const H: f32 = 110.0;
        const GAP: f32 = 40.0;
        ui.horizontal(|ui| {
            ui.spacing_mut().item_spacing.x = GAP;
            ui.add_space(((ui.available_width() - (2.0 * W + GAP)) / 2.0).max(0.0));
            for mode in [Mode::WhiteBeam, Mode::Tof] {
                let available = mode != Mode::Tof || self.instrument != Instrument::Mars;
                let button = egui::Button::new(RichText::new(mode.label()).size(26.0).strong())
                    .min_size(egui::vec2(W, H))
                    .selected(self.mode == Some(mode));
                let response = ui
                    .add_enabled(available, button)
                    .on_disabled_hover_text("TOF is not available on MARS");
                if response.clicked() && self.mode != Some(mode) {
                    logger::log(format!("mode selected: {}", mode.label()));
                    self.mode = Some(mode);
                }
            }
        });
    }

    /// `Next` in the bottom-right corner, enabled once the instrument, the
    /// experiment and the mode are all selected; returns `true` when clicked.
    fn next_button(&mut self, ui: &mut egui::Ui) -> bool {
        let mut go = false;
        let ready = self.selected.is_some() && self.mode.is_some();
        ui.with_layout(Layout::bottom_up(Align::Max), |ui| {
            ui.add_space(8.0);
            if next_button_widget(ui, ready).clicked() {
                go = true;
            }
            if !ready {
                let missing = match (self.selected.is_some(), self.mode.is_some()) {
                    (false, false) => "select an experiment and a mode",
                    (false, true) => "select an experiment",
                    _ => "select a mode",
                };
                ui.label(RichText::new(missing).weak().size(12.0));
            }
        });
        go
    }

    /// Load the debug config and prefill the setup screen from it. Turns the
    /// toggle back off when the config file itself cannot be read.
    fn enable_debug(&mut self) {
        let path = self.config_path.clone();
        match config::read(&path) {
            Ok(cfg) => {
                self.instrument = cfg.instrument;
                self.filter.clear();
                self.manual_error = None;
                match ipts::manual_entry(cfg.instrument, &cfg.ipts) {
                    Ok(entry) => {
                        self.selected = Some(entry);
                        self.debug_error = None;
                        self.scroll_to_selected = true;
                    }
                    Err(e) => {
                        logger::error(format!("debug config IPTS rejected: {e}"));
                        self.selected = None;
                        self.debug_error = Some(e);
                    }
                }
                self.mode = Some(cfg.mode);
                if cfg.instrument == Instrument::Mars && cfg.mode == Mode::Tof {
                    logger::error("debug config asks for TOF on MARS — forcing White Beam");
                    self.mode = Some(Mode::WhiteBeam);
                }
                logger::log(format!(
                    "debug mode ON — prefilled from {}: {} / {} / {}",
                    path.display(),
                    cfg.instrument.name(),
                    cfg.ipts,
                    cfg.mode.label()
                ));
                self.debug_info = Some(format!(
                    "{} -> {} / {} / {}",
                    path.display(),
                    cfg.instrument.name(),
                    cfg.ipts,
                    cfg.mode.label()
                ));
            }
            Err(e) => {
                logger::error(format!("debug config load failed: {e}"));
                self.debug_mode = false;
                self.debug_info = None;
                self.debug_error = Some(e);
            }
        }
    }

    fn disable_debug(&mut self) {
        logger::log("debug mode OFF");
        self.debug_info = None;
        self.debug_error = None;
    }

    fn admin_panel(&mut self, ui: &mut egui::Ui) {
        ui.collapsing(RichText::new("🔧 Admin").size(14.0), |ui| {
            if !self.admin_unlocked {
                ui.horizontal(|ui| {
                    ui.label("Password:");
                    let edit = ui.add(
                        egui::TextEdit::singleline(&mut self.admin_password)
                            .password(true)
                            .desired_width(140.0),
                    );
                    let entered =
                        edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                    if ui.button("Unlock").clicked() || entered {
                        if password_matches(&self.admin_password) {
                            logger::log("admin section unlocked");
                            self.admin_unlocked = true;
                            self.admin_error = None;
                        } else {
                            logger::error("admin unlock failed (wrong password)");
                            self.admin_error = Some("wrong password".to_owned());
                        }
                        self.admin_password.clear();
                    }
                });
                if let Some(e) = &self.admin_error {
                    ui.colored_label(Color32::LIGHT_RED, e);
                }
            } else {
                ui.horizontal(|ui| {
                    let label = if self.debug_mode {
                        "Debug mode: ON"
                    } else {
                        "Debug mode: OFF"
                    };
                    let toggle =
                        ui.toggle_value(&mut self.debug_mode, RichText::new(label).strong());
                    if toggle.changed() {
                        if self.debug_mode {
                            self.enable_debug();
                        } else {
                            self.disable_debug();
                        }
                    }
                    if ui.button("🔒 Lock").clicked() {
                        logger::log("admin section locked");
                        self.admin_unlocked = false;
                        self.admin_error = None;
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Config:");
                    let name = self
                        .config_path
                        .file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| self.config_path.display().to_string());
                    ui.label(RichText::new(name).monospace())
                        .on_hover_text(self.config_path.display().to_string());
                    if ui.button("📂 Browse…").clicked() {
                        let mut dialog = rfd::FileDialog::new()
                            .add_filter("HDF5 config", &["h5", "hdf5"])
                            .set_title("Select a debug config file");
                        if let Some(dir) = self.config_path.parent().filter(|d| d.is_dir()) {
                            dialog = dialog.set_directory(dir);
                        }
                        if let Some(path) = dialog.pick_file() {
                            logger::log(format!("debug config changed: {}", path.display()));
                            self.config_path = path;
                            // A new file while debug is on takes effect right away.
                            if self.debug_mode {
                                self.enable_debug();
                            }
                        }
                    }
                });
                if let Some(info) = &self.debug_info {
                    ui.label(RichText::new(info).weak().size(12.0));
                }
                if let Some(e) = &self.debug_error {
                    ui.colored_label(Color32::LIGHT_RED, e);
                }
            }
        });
    }
}

/// The custom-painted `Next` button: a rounded accent pill with a
/// double-chevron arrow, grayed out while the selection is incomplete.
fn next_button_widget(ui: &mut egui::Ui, enabled: bool) -> egui::Response {
    use egui::{Align2, CursorIcon, FontId, Pos2, Sense, Stroke, vec2};
    let (rect, response) = ui.allocate_exact_size(
        vec2(190.0, 56.0),
        if enabled { Sense::click() } else { Sense::hover() },
    );
    if !ui.is_rect_visible(rect) {
        return response;
    }
    let (fill, content) = if !enabled {
        (Color32::from_gray(45), Color32::from_gray(110))
    } else if response.is_pointer_button_down_on() {
        (Color32::from_rgb(0, 86, 160), Color32::WHITE)
    } else if response.hovered() {
        (Color32::from_rgb(36, 140, 235), Color32::WHITE)
    } else {
        (Color32::from_rgb(0, 110, 200), Color32::WHITE)
    };
    let painter = ui.painter();
    painter.rect_filled(rect, 14.0, fill);
    painter.text(
        rect.center() + vec2(-18.0, 0.0),
        Align2::CENTER_CENTER,
        "Next",
        FontId::proportional(22.0),
        content,
    );
    // Double chevron » to the right of the label.
    let stroke = Stroke::new(3.0, content);
    let h = 9.0;
    for dx in [0.0, 13.0] {
        let x = rect.center().x + 32.0 + dx;
        let y = rect.center().y;
        painter.line_segment([Pos2::new(x, y - h), Pos2::new(x + h, y)], stroke);
        painter.line_segment([Pos2::new(x + h, y), Pos2::new(x, y + h)], stroke);
    }
    if enabled {
        response.on_hover_cursor(CursorIcon::PointingHand)
    } else {
        response
    }
}

/// Top-right corner of the current row: the imaging team logo (at the given
/// height) and the toggle that opens/closes the log viewer side panel.
fn top_right_bar(
    ui: &mut egui::Ui,
    logo: Option<&egui::TextureHandle>,
    log_open: &mut bool,
    logo_height: f32,
) {
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
        if let Some(tex) = logo {
            ui.add(egui::Image::from_texture(tex).max_height(logo_height));
        }
        if ui.selectable_label(*log_open, "📜 Log").clicked() {
            *log_open = !*log_open;
        }
    });
}

/// The workflow screen: shared header, then the mode-specific view; returns
/// `true` when the user wants to go back to the setup screen.
fn workflow_ui(
    ui: &mut egui::Ui,
    session: &Session,
    view: &mut WorkflowView,
    logo: Option<&egui::TextureHandle>,
    log_open: &mut bool,
) -> bool {
    let mut back = false;
    // One compact header row: back, session recap, log toggle and a small
    // logo — the vertical space belongs to the workflow itself.
    ui.horizontal(|ui| {
        if ui.button("↩ Back").clicked() {
            back = true;
        }
        ui.label(
            RichText::new(format!(
                "{} — {} — {}",
                session.instrument.name(),
                session.ipts.name,
                session.mode.label()
            ))
            .size(15.0)
            .strong(),
        );
        top_right_bar(ui, logo, log_open, 28.0);
    });
    ui.add_space(6.0);
    match view {
        WorkflowView::WhiteBeam(wb_view) => {
            egui::ScrollArea::vertical()
                .id_salt("wb_workflow_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| white_beam_ui(ui, session, wb_view));
        }
        WorkflowView::Tof(tof_view) => {
            egui::ScrollArea::vertical()
                .id_salt("tof_workflow_scroll")
                .auto_shrink([false, false])
                .show(ui, |ui| tof_ui(ui, session, tof_view));
        }
    }
    back
}

fn white_beam_ui(ui: &mut egui::Ui, session: &Session, view: &mut WhiteBeamView) {
    // Detector — it decides where the sample and OB folders are looked for,
    // so changing it rebuilds both pickers.
    ui.horizontal(|ui| {
        ui.label(RichText::new("Detector:").strong());
        let mut detector = view.detector;
        egui::ComboBox::from_id_salt("wb_detector")
            .selected_text(detector.label())
            .show_ui(ui, |ui| {
                for d in WbDetector::ALL {
                    ui.selectable_value(&mut detector, d, d.label());
                }
            });
        if detector != view.detector {
            logger::log(format!("white beam detector changed: {}", detector.label()));
            *view = WhiteBeamView::with_detector(detector, session);
        }
        ui.label(
            RichText::new("one image per projection")
                .weak()
                .size(11.0),
        );
    });
    ui.add_space(10.0);
    ui.separator();
    ui.add_space(10.0);

    ui.columns(2, |cols| {
        view.sample.ui(&mut cols[0]);
        view.ob.ui(&mut cols[1]);
    });

    ui.add_space(10.0);
    egui::CollapsingHeader::new(RichText::new("Projection angles").strong())
        .default_open(true)
        .show(ui, |ui| {
            angle_source_ui(ui, view);
        });
}

/// How the projection angle of each image is obtained: naming convention
/// (pick 2 file-name fields), TIFF metadata, or an imported ASCII list.
fn angle_source_ui(ui: &mut egui::Ui, view: &mut WhiteBeamView) {
    let mut source = view.angle_source;
    for option in AngleSource::ALL {
        ui.radio_value(&mut source, option, option.label());
    }
    if source != view.angle_source {
        logger::log(format!("angle retrieval method: {}", source.label()));
        view.angle_source = source;
    }
    ui.add_space(6.0);

    match view.angle_source {
        AngleSource::NamingConvention => {
            let Some(example) = view.first_sample_image().cloned() else {
                ui.label(
                    RichText::new("select at least one sample folder to set up the convention")
                        .weak(),
                );
                return;
            };
            // (Re)build the checkboxes when the example file changes.
            if view.nc_example.as_ref() != Some(&example) {
                view.nc_fields = white_beam::name_fields(&example);
                view.nc_checked = vec![false; view.nc_fields.len()];
                if let Some((i, j)) = white_beam::default_angle_fields(view.nc_fields.len()) {
                    view.nc_checked[i] = true;
                    view.nc_checked[j] = true;
                }
                view.nc_example = Some(example.clone());
            }
            ui.horizontal(|ui| {
                ui.label(RichText::new("File name:").strong());
                ui.label(
                    RichText::new(
                        example
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default(),
                    )
                    .monospace()
                    .size(12.0),
                );
            });
            ui.label(
                RichText::new(
                    "check the 2 fields giving the angle value (degrees . decimals)",
                )
                .weak()
                .size(12.0),
            );
            ui.horizontal_wrapped(|ui| {
                for (field, checked) in view.nc_fields.iter().zip(view.nc_checked.iter_mut()) {
                    ui.checkbox(checked, field.as_str());
                }
            });
            let picked: Vec<usize> = view
                .nc_checked
                .iter()
                .enumerate()
                .filter(|(_, c)| **c)
                .map(|(i, _)| i)
                .collect();
            match picked.as_slice() {
                [i, j] => match white_beam::angle_from_fields(&example, *i, *j) {
                    Some(angle) => {
                        ui.label(
                            RichText::new(format!(
                                "angle value: {}.{} = {angle:.3}°",
                                view.nc_fields[*i], view.nc_fields[*j]
                            ))
                            .strong(),
                        );
                    }
                    None => {
                        ui.colored_label(
                            Color32::LIGHT_RED,
                            format!(
                                "'{}.{}' is not a number — pick numeric fields",
                                view.nc_fields[*i], view.nc_fields[*j]
                            ),
                        );
                    }
                },
                _ => {
                    ui.colored_label(Color32::LIGHT_RED, "select 2 and only 2 fields!");
                }
            }
        }
        AngleSource::Metadata => {
            ui.label(
                RichText::new(
                    "the angle will be read from each image's TIFF metadata (tag 65039)",
                )
                .color(Color32::from_rgb(120, 200, 120)),
            );
            let first = view.first_sample_image().cloned();
            match &first {
                None => {
                    ui.label(RichText::new("select a sample folder to test it").weak());
                }
                Some(file) => {
                    if ui.button("Test on the first image").clicked() {
                        let result = white_beam::angle_from_tiff_metadata(file);
                        match &result {
                            Ok(angle) => logger::log(format!(
                                "metadata angle test: {} -> {angle:.3} deg",
                                file.display()
                            )),
                            Err(e) => logger::error(format!("metadata angle test failed: {e}")),
                        }
                        view.metadata_test = Some(result);
                    }
                }
            }
            match &view.metadata_test {
                Some(Ok(angle)) => {
                    ui.label(RichText::new(format!("first image angle: {angle:.3}°")).strong());
                }
                Some(Err(e)) => {
                    ui.colored_label(Color32::LIGHT_RED, e);
                }
                None => {}
            }
        }
        AngleSource::AsciiFile => {
            ui.horizontal(|ui| {
                if ui.button("📂 Select ASCII file…").clicked() {
                    let mut dialog = rfd::FileDialog::new()
                        .set_title("Select the ASCII file containing the list of angles")
                        .add_filter("Text files", &["txt"])
                        .add_filter("All files", &["*"]);
                    if let Some(dir) = view.sample.root.parent().filter(|p| p.is_dir()) {
                        dialog = dialog.set_directory(dir);
                    }
                    if let Some(path) = dialog.pick_file() {
                        match white_beam::angles_from_ascii(&path) {
                            Ok(angles) => {
                                logger::log(format!(
                                    "angle list imported: {} ({} angles)",
                                    path.display(),
                                    angles.len()
                                ));
                                view.ascii_angles = Some(angles);
                                view.ascii_error = None;
                            }
                            Err(e) => {
                                logger::error(format!("angle list rejected: {e}"));
                                view.ascii_angles = None;
                                view.ascii_error = Some(e);
                            }
                        }
                        view.ascii_path = Some(path);
                    }
                }
                if let Some(path) = &view.ascii_path {
                    ui.label(
                        RichText::new(path.display().to_string())
                            .weak()
                            .size(11.0),
                    );
                }
            });
            if let Some(e) = &view.ascii_error {
                ui.colored_label(Color32::LIGHT_RED, e);
            }
            if let Some(angles) = &view.ascii_angles {
                let n_images = view.total_sample_images();
                let preview: Vec<String> =
                    angles.iter().take(5).map(|a| format!("{a:.3}")).collect();
                ui.label(format!(
                    "{} angles ({}{})",
                    angles.len(),
                    preview.join(", "),
                    if angles.len() > 5 { ", …" } else { "" }
                ));
                if n_images == 0 {
                    ui.label(RichText::new("select the sample folder(s) to validate").weak());
                } else if angles.len() == n_images {
                    ui.label(
                        RichText::new(format!(
                            "matches the {n_images} sample images ✔",
                        ))
                        .color(Color32::from_rgb(120, 200, 120)),
                    );
                } else {
                    ui.colored_label(
                        Color32::LIGHT_RED,
                        format!(
                            "{} angles but {n_images} sample images — they must match",
                            angles.len()
                        ),
                    );
                }
            }
        }
    }
}

fn tof_ui(ui: &mut egui::Ui, session: &Session, view: &mut TofView) {
    let ctx = ui.ctx().clone();
    view.sample.poll(&ctx);
    view.ob.poll(&ctx);

    // Both selections made and inventoried: log the selection summary, once
    // per (sample, OB) pair.
    if let (Some(sample_folders), Some(sample_path), Some(ob_path)) = (
        view.sample.folders.as_ref(),
        view.sample.selected.as_ref(),
        view.ob.selected.as_ref(),
    ) && view.ob.folders.is_some()
    {
        let pair = (sample_path.clone(), ob_path.clone());
        if view.summary_logged.as_ref() != Some(&pair) {
            logger::log(format!("Number of projections: {}", sample_folders.len()));
            logger::log(format!("Sample folder: {}", pair.0.display()));
            logger::log(format!("OB folder: {}", pair.1.display()));
            logger::log(format!(
                "Nexus folder: {}",
                session.ipts.path.join("nexus").display()
            ));
            logger::log(format!("Detector: {}", view.detector.label()));
            logger::log("preprocessing: rejecting empty runs, reading proton charges from NeXus…");
            view.preprocessed = None;
            view.removed_sample.clear();
            view.removed_ob.clear();
            view.process = None;
            view.processed = None;
            view.save_status = None;
            view.preprocess = Some(PreprocessScan::start(
                sample_folders.iter().map(ImageFolder::summary).collect(),
                view.ob
                    .folders
                    .as_ref()
                    .map(|f| f.iter().map(ImageFolder::summary).collect())
                    .unwrap_or_default(),
                session.ipts.path.join("nexus"),
                session.instrument.name().to_owned(),
            ));
            view.summary_logged = Some(pair);
        }
    }

    // Fold a finished preprocessing pass into the view, logging its findings
    // and defaulting the proton-charge selection to median ±10%.
    if let Some(scan) = &mut view.preprocess {
        if let Some(result) = scan.poll() {
            log_preprocess_result(&result);
            if let Some((range, bounds)) = default_pc_selection(&result) {
                logger::log(format!(
                    "proton charge selection default (sample median ±10%): {:.3} – {:.3} C",
                    range.0, range.1
                ));
                view.pc_range = Some(range);
                view.pc_bounds = bounds;
            } else {
                view.pc_range = None;
            }
            view.preprocessed = Some(result);
            view.preprocess = None;
        } else {
            ctx.request_repaint_after(Duration::from_millis(200));
        }
    }

    // Detector — it decides where the sample and OB folders are looked for,
    // so changing it rebuilds both pickers.
    ui.horizontal(|ui| {
        ui.label(RichText::new("Detector:").strong());
        let mut detector = view.detector;
        egui::ComboBox::from_id_salt("tof_detector")
            .selected_text(detector.label())
            .show_ui(ui, |ui| {
                for d in Detector::ALL {
                    ui.selectable_value(&mut detector, d, d.label());
                }
            });
        if detector != view.detector {
            logger::log(format!("TOF detector changed: {}", detector.label()));
            *view = TofView::with_detector(detector, session);
        }
    });
    ui.add_space(10.0);
    ui.separator();
    ui.add_space(10.0);

    ui.columns(2, |cols| {
        view.sample.ui(&mut cols[0]);
        view.ob.ui(&mut cols[1]);
    });

    ui.add_space(10.0);
    if let Some(scan) = &view.preprocess {
        ui.separator();
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(format!(
                "preprocessing: reading NeXus proton charges… {}/{}",
                scan.done, scan.total
            ));
        });
    }
    if view.preprocessed.is_some() {
        ui.separator();
        ui.add_space(6.0);
        egui::CollapsingHeader::new(RichText::new("Proton charge selection").strong())
            .default_open(true)
            .show(ui, |ui| {
                preprocess_summary_ui(ui, view.preprocessed.as_ref().unwrap());
                proton_charge_section(ui, view);
            });
        ui.add_space(4.0);
        egui::CollapsingHeader::new(RichText::new("Runs going to the next step").strong())
            .default_open(true)
            .show(ui, |ui| {
                runs_selection_ui(ui, view);
            });
        ui.add_space(4.0);
        egui::CollapsingHeader::new(RichText::new("Combine images (TOF range)").strong())
            .default_open(true)
            .show(ui, |ui| {
                combine_section_ui(ui, &ctx, view);
            });
        ui.add_space(4.0);
        egui::CollapsingHeader::new(RichText::new("Process & combine (save to HDF5)").strong())
            .default_open(true)
            .show(ui, |ui| {
                process_section_ui(ui, &ctx, session, view);
            });
    }
}

/// The union of the enabled file-index ranges of the viewer selections, or
/// everything in combine-all mode; `None` when nothing usable is selected.
fn image_selection(view: &TofView) -> Option<ImageSelection> {
    if view.combine_all {
        return Some(ImageSelection::All);
    }
    let spec = view.combine_spec.as_ref()?;
    let ranges: Vec<(usize, usize)> = spec
        .ranges
        .iter()
        .filter(|r| r.enabled)
        .filter_map(|r| r.file_index)
        .map(|(a, b)| {
            let (a, b) = (a.round().max(0.0) as usize, b.round().max(0.0) as usize);
            (a.min(b), a.max(b))
        })
        .collect();
    (!ranges.is_empty()).then_some(ImageSelection::FileIndexRanges(ranges))
}

/// Mean-combine every kept run folder and offer to save the projections
/// (sorted by increasing angle) plus the OB stack into an HDF5 file.
fn process_section_ui(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    session: &Session,
    view: &mut TofView,
) {
    // Fold finished background work into the view.
    if let Some(scan) = &mut view.process {
        if let Some(output) = scan.poll() {
            let angles: Vec<f64> = output.sample.iter().filter_map(|p| p.angle_deg).collect();
            logger::log(format!(
                "combined {} sample projections (angles {}) and {} ob images with mean",
                output.sample.len(),
                match (angles.first(), angles.last()) {
                    (Some(a), Some(b)) => format!("{a:.3} deg -> {b:.3} deg, increasing"),
                    _ => "unknown".to_owned(),
                },
                output.ob.len()
            ));
            for note in &output.notes {
                logger::log(format!("duplicate angle: {note}"));
            }
            for e in &output.skipped {
                logger::error(format!("combine skipped: {e}"));
            }
            view.processed = Some(std::sync::Arc::new(output));
            view.process = None;
        } else {
            ctx.request_repaint_after(Duration::from_millis(300));
        }
    }
    if let Some(job) = &mut view.save_job {
        match job.poll() {
            Some(Ok(msg)) => {
                logger::log(format!("saved combined data: {msg}"));
                view.save_status = Some(Ok(msg));
                view.save_job = None;
            }
            Some(Err(e)) => {
                logger::error(format!("saving combined data failed: {e}"));
                view.save_status = Some(Err(e));
                view.save_job = None;
            }
            None => ctx.request_repaint_after(Duration::from_millis(300)),
        }
    }

    let Some(result) = &view.preprocessed else {
        return;
    };
    let Some(range) = view.pc_range else {
        return;
    };
    fn kept<'a>(
        runs: &'a [RunInfo],
        removed: &HashSet<String>,
        range: (f64, f64),
    ) -> Vec<&'a RunInfo> {
        runs.iter()
            .filter(|r| pc_in_range(r, range) && !removed.contains(&r.name))
            .collect()
    }
    let kept_sample = kept(&result.sample, &view.removed_sample, range);
    let kept_ob = kept(&result.ob, &view.removed_ob, range);

    let selection = image_selection(view);
    match &selection {
        None => {
            ui.label(
                RichText::new(
                    "select TOF range(s) above (or switch to 'combine all images') to enable",
                )
                .weak(),
            );
        }
        Some(sel) => {
            ui.label(format!(
                "{} sample folders (one projection per angle) and {} ob folders — mean of {}",
                kept_sample.len(),
                kept_ob.len(),
                sel.describe()
            ));
        }
    }

    if ui
        .checkbox(
            &mut view.merge_same_angle,
            "Combine folders having the same angle",
        )
        .on_hover_text(
            "off: when several folders share a projection angle, only the one with the \
             best statistics (highest total counts over its stack) is used",
        )
        .changed()
    {
        logger::log(format!(
            "duplicate angles policy: {}",
            if view.merge_same_angle {
                "combine folders with the same angle"
            } else {
                "keep the folder with the best statistics"
            }
        ));
    }

    let busy = view.process.is_some();
    let ready = selection.is_some() && !kept_sample.is_empty() && !busy;
    if ui
        .add_enabled(ready, egui::Button::new("▶ Combine all kept folders (mean)"))
        .clicked()
        && let Some(sel) = selection
    {
        let to_combine = |kept: &[&RunInfo]| -> Vec<RunToCombine> {
            kept.iter()
                .filter_map(|r| {
                    let folders = match r.path.parent() == view.sample.selected.as_deref() {
                        true => view.sample.folders.as_deref(),
                        false => view.ob.folders.as_deref(),
                    }?;
                    folders.iter().find(|f| f.name == r.name).map(|f| RunToCombine {
                        name: r.name.clone(),
                        run_number: r.run_number,
                        images: f.images.clone(),
                    })
                })
                .collect()
        };
        let sample_runs = to_combine(&kept_sample);
        let ob_runs = to_combine(&kept_ob);
        logger::log(format!(
            "combining (mean of {}): {} sample folders, {} ob folders",
            sel.describe(),
            sample_runs.len(),
            ob_runs.len()
        ));
        view.processed = None;
        view.save_status = None;
        view.process = Some(CombineScan::start(
            sample_runs,
            ob_runs,
            sel,
            view.merge_same_angle,
        ));
    }

    if let Some(scan) = &view.process {
        let done = scan.progress();
        let frac = (done as f32 / scan.total_images.max(1) as f32).min(1.0);
        ui.add(
            egui::ProgressBar::new(frac)
                .text(format!("{done}/{} images", scan.total_images)),
        );
    }

    if let Some(output) = &view.processed {
        let angles: Vec<f64> = output.sample.iter().filter_map(|p| p.angle_deg).collect();
        let dims = output
            .sample
            .first()
            .map(|p| format!("{}x{}", p.height, p.width))
            .unwrap_or_else(|| "?".to_owned());
        ui.label(
            RichText::new(format!(
                "combined: {} projections ({dims}), angles {} — {} ob images",
                output.sample.len(),
                match (angles.first(), angles.last()) {
                    (Some(a), Some(b)) => format!("{a:.3}° to {b:.3}°"),
                    _ => "unknown".to_owned(),
                },
                output.ob.len()
            ))
            .strong(),
        );
        for note in &output.notes {
            ui.label(RichText::new(note).weak().size(12.0));
        }
        for e in &output.skipped {
            ui.colored_label(Color32::from_rgb(240, 180, 60), format!("skipped: {e}"));
        }
        let saving = view.save_job.is_some();
        if ui
            .add_enabled(!saving, egui::Button::new("💾 Save to HDF5…"))
            .clicked()
        {
            let default_name = view
                .sample
                .selected
                .as_deref()
                .and_then(|p| p.file_name())
                .map(|n| format!("{}_combined.h5", n.to_string_lossy()))
                .unwrap_or_else(|| "ct_combined.h5".to_owned());
            let mut dialog = rfd::FileDialog::new()
                .set_title("Save the combined projections")
                .add_filter("HDF5", &["h5", "hdf5"])
                .set_file_name(default_name);
            let shared = session.ipts.path.join("shared");
            if shared.is_dir() {
                dialog = dialog.set_directory(shared);
            }
            if let Some(path) = dialog.save_file() {
                let meta = SaveMeta {
                    instrument: session.instrument.name().to_owned(),
                    ipts: session.ipts.name.clone(),
                    detector: view.detector.label().to_owned(),
                    sample_folder: view
                        .sample
                        .selected
                        .as_deref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                    ob_folder: view
                        .ob
                        .selected
                        .as_deref()
                        .map(|p| p.display().to_string())
                        .unwrap_or_default(),
                    combine_mode: if view.combine_all {
                        "all images".to_owned()
                    } else {
                        image_selection(view)
                            .map(|s| s.describe())
                            .unwrap_or_default()
                    },
                    selections_json: (!view.combine_all)
                        .then(|| view.combine_spec.as_ref().map(|s| s.raw.clone()))
                        .flatten(),
                    detector_offset_us: result.detector_offset_us,
                };
                logger::log(format!("saving combined data to {}", path.display()));
                view.save_status = None;
                view.save_job = Some(SaveJob::start(
                    path,
                    std::sync::Arc::clone(output),
                    meta,
                ));
            }
        }
        if saving {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("writing HDF5…");
            });
        }
    }
    match &view.save_status {
        Some(Ok(msg)) => {
            ui.colored_label(Color32::from_rgb(120, 200, 120), format!("saved: {msg}"));
        }
        Some(Err(e)) => {
            ui.colored_label(Color32::LIGHT_RED, format!("save failed: {e}"));
        }
        None => {}
    }
}

/// Choosing how the TOF images are combined (file index / TOF / lambda /
/// energy ranges): the first kept sample projection is opened in the TOF
/// Profile Viewer, whose exported selections come back here.
fn combine_section_ui(ui: &mut egui::Ui, ctx: &egui::Context, view: &mut TofView) {
    // Fold a finished viewer session into the view.
    if let Some(job) = &mut view.combine_job {
        match job.poll() {
            Some(Ok(Some(json))) => {
                match tof::parse_selections(&json) {
                    Ok(spec) => {
                        log_combine_spec(&spec);
                        view.combine_spec = Some(spec);
                        view.combine_error = None;
                    }
                    Err(e) => {
                        logger::error(format!("TOF combine selections rejected: {e}"));
                        view.combine_error = Some(e);
                    }
                }
                view.combine_job = None;
            }
            Some(Ok(None)) => {
                logger::log("TOF Profile Viewer closed without exporting selections");
                view.combine_job = None;
            }
            Some(Err(e)) => {
                logger::error(format!("TOF Profile Viewer failed: {e}"));
                view.combine_error = Some(e);
                view.combine_job = None;
            }
            None => ctx.request_repaint_after(Duration::from_millis(300)),
        }
    }

    // Combine everything, or select TOF range(s) in the profile viewer.
    let mut combine_all = view.combine_all;
    ui.horizontal(|ui| {
        ui.radio_value(&mut combine_all, false, "Select TOF range(s)");
        ui.radio_value(
            &mut combine_all,
            true,
            "Combine all images (no TOF selection)",
        );
    });
    if combine_all != view.combine_all {
        logger::log(if combine_all {
            "combine mode: all images (no TOF selection)"
        } else {
            "combine mode: selected TOF range(s)"
        });
        view.combine_all = combine_all;
    }
    if view.combine_all {
        ui.label(
            RichText::new("every image of each run will be combined")
                .weak()
                .size(12.0),
        );
        return;
    }
    ui.add_space(4.0);

    let first_run = view.first_kept_sample_run().cloned();
    match (&view.combine_job, &first_run) {
        (Some(_), _) => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(
                    "TOF Profile Viewer is open — make the selections there and press its \
                     export button to bring them back",
                );
            });
        }
        (None, None) => {
            ui.label(
                RichText::new("no sample run left to visualize — adjust the selection above")
                    .weak(),
            );
        }
        (None, Some(run)) => {
            let label = if view.combine_spec.is_some() {
                "📈 Reopen the TOF Profile Viewer"
            } else {
                "📈 Open the TOF Profile Viewer (first projection)"
            };
            if ui.button(label).clicked() {
                let offset_us = view
                    .preprocessed
                    .as_ref()
                    .and_then(|r| r.detector_offset_us);
                logger::log(format!(
                    "opening TOF Profile Viewer on first projection: {} (detector offset: {})",
                    run.path.display(),
                    offset_us
                        .map(|o| format!("{o:.1} µs"))
                        .unwrap_or_else(|| "not found".to_owned())
                ));
                view.combine_error = None;
                // The previous session's document restores its TOF ranges
                // and manual bins in the reopened viewer — but only a
                // document with either is accepted by --selections.
                view.combine_job = Some(ViewerJob::launch(
                    run.path.clone(),
                    offset_us,
                    view.combine_spec
                        .as_ref()
                        .filter(|s| s.has_bins || s.ranges.iter().any(|r| r.tof_us.is_some()))
                        .map(|s| s.raw.as_str()),
                ));
            }
            let offset_note = view
                .preprocessed
                .as_ref()
                .and_then(|r| r.detector_offset_us)
                .map(|o| format!(" — detector offset {o:.1} µs from its NeXus"))
                .unwrap_or_default();
            ui.label(
                RichText::new(format!("first projection: {}{offset_note}", run.name))
                    .weak()
                    .size(11.0),
            );
        }
    }

    if let Some(e) = &view.combine_error {
        ui.colored_label(Color32::LIGHT_RED, e);
    }
    if let Some(spec) = &view.combine_spec {
        ui.add_space(4.0);
        let enabled: Vec<&tof::CombineRange> =
            spec.ranges.iter().filter(|r| r.enabled).collect();
        ui.label(
            RichText::new(format!("{} combine range(s) selected", enabled.len())).strong(),
        );
        for (i, range) in enabled.iter().enumerate() {
            ui.label(RichText::new(format!("#{}: {}", i + 1, combine_range_text(range))).size(12.0));
        }
        if spec.has_bins {
            ui.label(
                RichText::new("a manual binning table is included in the selections")
                    .weak()
                    .size(12.0),
            );
        }
    }
}

/// One selection on every axis the viewer could express it in, e.g.
/// `file index 120 – 240 | TOF 2000.0 – 4000.0 µs | λ 3.960 – 7.920 Å`.
fn combine_range_text(range: &tof::CombineRange) -> String {
    let mut parts = Vec::new();
    if let Some((a, b)) = range.file_index {
        parts.push(format!("file index {:.0} – {:.0}", a, b));
    }
    if let Some((a, b)) = range.tof_us {
        parts.push(format!("TOF {a:.1} – {b:.1} µs"));
    }
    if let Some((a, b)) = range.lambda_angstrom {
        parts.push(format!("λ {a:.3} – {b:.3} Å"));
    }
    if let Some((a, b)) = range.energy_ev {
        parts.push(format!("E {a:.4} – {b:.4} eV"));
    }
    if parts.is_empty() {
        "no usable axis values".to_owned()
    } else {
        parts.join("  |  ")
    }
}

fn log_combine_spec(spec: &CombineSpec) {
    let enabled: Vec<&tof::CombineRange> = spec.ranges.iter().filter(|r| r.enabled).collect();
    logger::log(format!(
        "TOF combine selections received: {} enabled range(s) (of {}) from {}",
        enabled.len(),
        spec.ranges.len(),
        spec.folder
    ));
    for (i, range) in enabled.iter().enumerate() {
        logger::log(format!("combine range #{}: {}", i + 1, combine_range_text(range)));
    }
    if spec.has_bins {
        logger::log("combine selections include a manual binning table");
    }
}

/// The sample and OB runs surviving the proton-charge band, each removable
/// (and restorable) from the next step with its checkbox.
fn runs_selection_ui(ui: &mut egui::Ui, view: &mut TofView) {
    let Some(result) = &view.preprocessed else {
        return;
    };
    let Some(range) = view.pc_range else {
        ui.label(RichText::new("no proton charge selection").weak());
        return;
    };
    let removed_sample = &mut view.removed_sample;
    let removed_ob = &mut view.removed_ob;
    ui.columns(2, |cols| {
        run_list_column(&mut cols[0], "sample", &result.sample, range, removed_sample);
        run_list_column(&mut cols[1], "ob", &result.ob, range, removed_ob);
    });
}

fn run_list_column(
    ui: &mut egui::Ui,
    label: &'static str,
    runs: &[RunInfo],
    range: (f64, f64),
    removed: &mut HashSet<String>,
) {
    let surviving: Vec<&RunInfo> = runs.iter().filter(|r| pc_in_range(r, range)).collect();
    let kept = surviving
        .iter()
        .filter(|r| !removed.contains(&r.name))
        .count();
    let heading = format!("{label} — {kept} run(s)");
    if kept == 0 {
        ui.colored_label(Color32::from_rgb(240, 180, 60), heading);
    } else {
        ui.label(RichText::new(heading).strong());
    }
    egui::ScrollArea::vertical()
        .id_salt((label, "next_step_runs"))
        .max_height(220.0)
        .auto_shrink([false, true])
        .show(ui, |ui| {
            for run in &surviving {
                ui.horizontal(|ui| {
                    let mut keep = !removed.contains(&run.name);
                    if ui
                        .checkbox(&mut keep, "")
                        .on_hover_text("uncheck to remove this run from the next step")
                        .changed()
                    {
                        if keep {
                            logger::log(format!("restored {label} run: {}", run.name));
                            removed.remove(&run.name);
                        } else {
                            logger::log(format!("manually removed {label} run: {}", run.name));
                            removed.insert(run.name.clone());
                        }
                    }
                    let pc = run
                        .proton_charge_c
                        .map(|pc| format!("{pc:.3} C"))
                        .unwrap_or_else(|| "?".to_owned());
                    let text = format!("{} — {pc} — {} images", run.name, run.n_images);
                    let text = RichText::new(text).size(12.0);
                    if keep {
                        ui.label(text);
                    } else {
                        ui.label(text.weak().strikethrough());
                    }
                });
            }
        });
    let removed_here = surviving
        .iter()
        .filter(|r| removed.contains(&r.name))
        .count();
    if removed_here > 0
        && ui
            .button(format!("↩ Restore the {removed_here} removed {label} run(s)"))
            .clicked()
    {
        logger::log(format!("restored all {removed_here} removed {label} runs"));
        for run in &surviving {
            removed.remove(&run.name);
        }
    }
}

/// Selection band defaulted to the median of the SAMPLE proton charges ±10%,
/// and the slider scale enclosing both the band and every measured value
/// (sample and OB, with a small margin).
fn default_pc_selection(result: &PreprocessResult) -> Option<((f64, f64), (f64, f64))> {
    let mut sample_pcs: Vec<f64> = result
        .sample
        .iter()
        .filter(|r| !r.rejected_empty)
        .filter_map(|r| r.proton_charge_c)
        .collect();
    if sample_pcs.is_empty() {
        return None;
    }
    sample_pcs.sort_by(f64::total_cmp);
    let median = if sample_pcs.len() % 2 == 0 {
        (sample_pcs[sample_pcs.len() / 2 - 1] + sample_pcs[sample_pcs.len() / 2]) / 2.0
    } else {
        sample_pcs[sample_pcs.len() / 2]
    };
    let range = (median * 0.9, median * 1.1);
    let all_pcs = result
        .sample
        .iter()
        .chain(&result.ob)
        .filter(|r| !r.rejected_empty)
        .filter_map(|r| r.proton_charge_c);
    let (mut low, mut high) = range;
    for pc in all_pcs {
        low = low.min(pc);
        high = high.max(pc);
    }
    let pad = ((high - low) * 0.05).max(high.abs() * 1e-3).max(1e-9);
    Some(((range.0, range.1), (low - pad, high + pad)))
}

fn pc_in_range(run: &RunInfo, range: (f64, f64)) -> bool {
    !run.rejected_empty
        && run
            .proton_charge_c
            .is_some_and(|pc| pc >= range.0 && pc <= range.1)
}

/// One line per data type: how many runs survive, plus rejections and runs
/// whose proton charge could not be read.
fn preprocess_summary_ui(ui: &mut egui::Ui, result: &PreprocessResult) {
    for (label, runs) in [("sample", &result.sample), ("ob", &result.ob)] {
        let empty = runs.iter().filter(|r| r.rejected_empty).count();
        let kept = runs.len() - empty;
        let missing_pc = runs
            .iter()
            .filter(|r| !r.rejected_empty && r.proton_charge_c.is_none())
            .count();
        let mut line = format!("{label}: {kept} runs");
        if empty > 0 {
            line.push_str(&format!(" — {empty} empty folder(s) rejected"));
        }
        if missing_pc > 0 {
            line.push_str(&format!(" — {missing_pc} without proton charge"));
        }
        if empty > 0 || missing_pc > 0 {
            ui.colored_label(Color32::from_rgb(240, 180, 60), line);
        } else {
            ui.label(line);
        }
    }
}

/// Sample and OB proton charges (C) against run number on one plot, with the
/// vertical range slider that selects which runs continue to the next step.
fn proton_charge_section(ui: &mut egui::Ui, view: &mut TofView) {
    const PLOT_HEIGHT: f32 = 240.0;
    let Some(result) = &view.preprocessed else {
        return;
    };
    let Some(range) = &mut view.pc_range else {
        ui.label(RichText::new("no proton charge values to plot").weak());
        return;
    };

    fn points(runs: &[RunInfo], keep: impl Fn(&RunInfo) -> bool) -> Vec<[f64; 2]> {
        runs.iter()
            .filter(|r| !r.rejected_empty && keep(r))
            .filter_map(|r| Some([r.run_number? as f64, r.proton_charge_c?]))
            .collect()
    }
    let selection = *range;
    let sample_in = points(&result.sample, |r| pc_in_range(r, selection));
    let sample_out = points(&result.sample, |r| !pc_in_range(r, selection));
    let ob_in = points(&result.ob, |r| pc_in_range(r, selection));
    let ob_out = points(&result.ob, |r| !pc_in_range(r, selection));

    ui.add_space(4.0);
    ui.label(RichText::new("Proton charge per run (C)").strong());
    let mut released = false;
    ui.horizontal(|ui| {
        released = vertical_range_slider(
            ui,
            range,
            view.pc_bounds,
            PLOT_HEIGHT,
            &mut view.pc_drag_upper,
        );
        let band = Color32::from_rgb(120, 200, 120);
        egui_plot::Plot::new("proton_charge_plot")
            .height(PLOT_HEIGHT)
            .x_axis_label("run number")
            .y_axis_label("proton charge (C)")
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.hline(egui_plot::HLine::new("selection", selection.0).color(band).width(1.0));
                plot_ui.hline(egui_plot::HLine::new("selection", selection.1).color(band).width(1.0));
                plot_ui.points(
                    egui_plot::Points::new("sample", sample_in)
                        .radius(3.5)
                        .color(Color32::from_rgb(100, 170, 255)),
                );
                plot_ui.points(
                    egui_plot::Points::new("ob", ob_in)
                        .radius(3.5)
                        .color(Color32::from_rgb(255, 160, 70)),
                );
                plot_ui.points(
                    egui_plot::Points::new("excluded", sample_out)
                        .radius(3.0)
                        .color(Color32::from_gray(110)),
                );
                plot_ui.points(
                    egui_plot::Points::new("excluded", ob_out)
                        .radius(3.0)
                        .color(Color32::from_gray(110)),
                );
            });
    });

    let selection = *view.pc_range.as_ref().unwrap();
    let kept = |runs: &[RunInfo]| runs.iter().filter(|r| pc_in_range(r, selection)).count();
    let (s_kept, s_all) = (kept(&result.sample), result.sample.len());
    let (o_kept, o_all) = (kept(&result.ob), result.ob.len());
    let line = format!(
        "selection: {:.3} – {:.3} C — keeping {s_kept}/{s_all} sample and {o_kept}/{o_all} ob runs",
        selection.0, selection.1
    );
    if released {
        logger::log(format!("proton charge selection changed: {line}"));
    }
    if s_kept == 0 || o_kept == 0 {
        ui.colored_label(
            Color32::from_rgb(240, 180, 60),
            format!("{line} — the next step needs at least one of each!"),
        );
    } else {
        ui.label(line);
    }
}

/// A custom-painted vertical two-handle range slider; returns `true` when a
/// drag ends (so changes can be logged once). `active` remembers which handle
/// the drag moves across frames.
fn vertical_range_slider(
    ui: &mut egui::Ui,
    selection: &mut (f64, f64),
    bounds: (f64, f64),
    height: f32,
    active: &mut Option<bool>,
) -> bool {
    use egui::{Pos2, Sense, Stroke, vec2};
    const WIDTH: f32 = 34.0;
    const HANDLE_R: f32 = 7.0;
    let (rect, response) = ui.allocate_exact_size(vec2(WIDTH, height), Sense::click_and_drag());
    let span = (bounds.1 - bounds.0).max(f64::EPSILON);
    let inner_top = rect.top() + HANDLE_R;
    let inner_height = (rect.height() - 2.0 * HANDLE_R).max(1.0);
    let to_y = |v: f64| inner_top + (((bounds.1 - v) / span) as f32) * inner_height;
    let to_v = |y: f32| bounds.1 - f64::from((y - inner_top) / inner_height).clamp(0.0, 1.0) * span;

    if let Some(pos) = response.interact_pointer_pos() {
        if response.drag_started() || (response.clicked() && active.is_none()) {
            // Grab whichever handle is closer to the press.
            let upper = (pos.y - to_y(selection.1)).abs() < (pos.y - to_y(selection.0)).abs();
            *active = Some(upper);
        }
        if let Some(upper) = *active {
            let v = to_v(pos.y);
            if upper {
                selection.1 = v.max(selection.0);
            } else {
                selection.0 = v.min(selection.1);
            }
        }
    }
    let released = active.is_some() && response.drag_stopped();
    if released {
        *active = None;
    }

    let painter = ui.painter();
    let cx = rect.center().x;
    // Track, selected band, handles.
    painter.line_segment(
        [Pos2::new(cx, inner_top), Pos2::new(cx, inner_top + inner_height)],
        Stroke::new(4.0, Color32::from_gray(70)),
    );
    painter.line_segment(
        [
            Pos2::new(cx, to_y(selection.1)),
            Pos2::new(cx, to_y(selection.0)),
        ],
        Stroke::new(6.0, Color32::from_rgb(120, 200, 120)),
    );
    for v in [selection.0, selection.1] {
        painter.circle_filled(Pos2::new(cx, to_y(v)), HANDLE_R, Color32::from_gray(230));
    }
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
    }
    released
}

fn log_preprocess_result(result: &PreprocessResult) {
    for (label, runs) in [("sample", &result.sample), ("ob", &result.ob)] {
        let rejected: Vec<&str> = runs
            .iter()
            .filter(|r| r.rejected_empty)
            .map(|r| r.name.as_str())
            .collect();
        let missing_pc: Vec<&str> = runs
            .iter()
            .filter(|r| !r.rejected_empty && r.proton_charge_c.is_none())
            .map(|r| r.name.as_str())
            .collect();
        let with_pc = runs
            .iter()
            .filter(|r| r.proton_charge_c.is_some())
            .count();
        logger::log(format!(
            "preprocessing {label}: {} runs, {} with proton charge",
            runs.len() - rejected.len(),
            with_pc
        ));
        if label == "sample" {
            match result.detector_offset_us {
                Some(offset) => logger::log(format!("detector offset: {offset:.1} µs")),
                None => logger::error("detector offset not found in the first sample NeXus"),
            }
        }
        if !rejected.is_empty() {
            logger::log(format!(
                "rejected {} empty {label} runs: {:?}",
                rejected.len(),
                rejected
            ));
        }
        if !missing_pc.is_empty() {
            logger::error(format!(
                "proton charge not found for {} {label} runs: {:?}",
                missing_pc.len(),
                missing_pc
            ));
        }
    }
}

impl eframe::App for CtApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if self.logo.is_none() {
            self.logo = load_logo(&ctx);
        }
        self.log_panel(ui, &ctx);
        if matches!(self.screen, Screen::Setup) {
            self.poll_scan(&ctx);
            let mut next = false;
            egui::Panel::bottom("admin").show(ui, |ui| {
                self.admin_panel(ui);
            });
            egui::CentralPanel::default().show(ui, |ui| {
                next = self.setup_ui(ui);
            });
            if next && let (Some(mode), Some(ipts)) = (self.mode, self.selected.clone()) {
                logger::log(format!(
                    "next -> {} workflow: {} / {}",
                    mode.label(),
                    self.instrument.name(),
                    ipts.name
                ));
                let session = Session {
                    instrument: self.instrument,
                    ipts,
                    mode,
                };
                let view = match mode {
                    Mode::WhiteBeam => WorkflowView::WhiteBeam(WhiteBeamView::new(&session)),
                    Mode::Tof => WorkflowView::Tof(TofView::new(&session)),
                };
                self.screen = Screen::Workflow { session, view };
            }
        } else {
            let mut back = false;
            egui::CentralPanel::default().show(ui, |ui| {
                if let Screen::Workflow { session, view } = &mut self.screen {
                    back = workflow_ui(ui, session, view, self.logo.as_ref(), &mut self.log_view_open);
                }
            });
            if back {
                logger::log("returned to setup screen");
                self.screen = Screen::Setup;
            }
        }
    }
}
