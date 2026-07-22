//! The egui/eframe application shell.
//!
//! The UI is a small state machine: the setup screen (instrument → IPTS →
//! acquisition mode) gates everything else, and the [`Session`] it produces
//! decides which workflow screen the rest of the application shows.

use crate::combine::{
    self, CombineOutput, CombineScan, ImageSelection, LoadJob, LoadedStack, RunToCombine,
    SaveJob, SaveMeta, StackSaveJob,
};
use crate::clean::{self, CleanJob, CleanSettings, CleanStats, LogConvertJob, LogStats};
use crate::config;
use crate::crop::{CropJob, CropRect};
use crate::instrument::Instrument;
use crate::ipts::{self, IptsEntry, IptsScan};
use crate::logger;
use crate::normalize::{NormJob, NormSettings, RoiJob, VisualizeJob};
use crate::rotate::{self, RotateJob};
use crate::stripes::{self, StripeAlgo, StripeApplyJob, StripeTestJob};
use crate::tilt::{CorJob, TiltApplyJob, TiltCalcJob, TiltResult};
pub use crate::session::{Mode, Session};
use crate::tof::{
    self, CombineSpec, Detector, FolderScan, ImageFolder, PreprocessResult, PreprocessScan,
    RunInfo, ViewerJob,
};
use crate::white_beam::{self, AngleSource, ImageIntensity, IntensityScan, MetaAnglesScan, WbDetector};

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
    /// Pre-processing of a stack of projections loaded from a previously
    /// saved HDF5 file.
    Stack(StackView),
    /// Evaluating the reconstruction of a pre-processed stack.
    Recon(ReconView),
}

/// The SVMBIR optimizer of the sibling repo.
const SVMBIR_OPTIMIZER_BIN: &str =
    "/SNS/VENUS/shared/software/git/rust_svmbir_optimizer/target/release/svmbir_optimizer";

/// The reconstruction algorithms of the pipeline (the Python
/// `ReconstructionAlgorithm` list); each gets its own standalone evaluation
/// application, launched from the reconstruction screen.
struct ReconAlgorithm {
    /// Configuration key: the saved parameters live in the checkpoint's
    /// metadata as `<key>_config`.
    key: &'static str,
    label: &'static str,
    description: &'static str,
    /// The standalone evaluator binary, when it exists already.
    binary: Option<&'static str>,
}

const RECON_ALGORITHMS: [ReconAlgorithm; 6] = [
    ReconAlgorithm {
        key: "svmbir",
        label: "SVMBIR",
        description: "sparse-view model-based iterative — high quality",
        binary: Some(SVMBIR_OPTIMIZER_BIN),
    },
    ReconAlgorithm {
        key: "mbirjax",
        label: "MBIRJAX",
        description: "JAX-based model-based iterative — GPU accelerated",
        binary: Some(
            "/SNS/VENUS/shared/software/git/rust_mbirjax_optimizer/target/release/mbirjax_optimizer",
        ),
    },
    ReconAlgorithm {
        key: "astra_fbp",
        label: "ASTRA FBP",
        description: "filtered back projection — fast, GPU support",
        binary: None,
    },
    ReconAlgorithm {
        key: "tomopy_fbp",
        label: "TomoPy FBP",
        description: "filtered back projection — versatile, well tested",
        binary: None,
    },
    ReconAlgorithm {
        key: "algotom_fbp",
        label: "AlgoTom FBP",
        description: "filtered back projection — optimized for large data",
        binary: None,
    },
    ReconAlgorithm {
        key: "algotom_gridrec",
        label: "AlgoTom GridRec",
        description: "very fast — good for quick previews",
        binary: None,
    },
];

struct ReconView {
    stack: std::sync::Arc<LoadedStack>,
    /// Center of rotation in px; `None` = the horizontal center.
    cor: Option<f64>,
    /// An algorithm evaluator session in flight, with its label.
    optimizer_job: Option<(String, std::sync::mpsc::Receiver<Result<(), String>>)>,
    /// Reloading the file after the optimizer closes, to pick up the saved
    /// parameters.
    reload_job: Option<LoadJob>,
    opt_error: Option<String>,
}

impl ReconView {
    fn new(stack: LoadedStack, cor: Option<f64>) -> Self {
        Self {
            stack: std::sync::Arc::new(stack),
            cor,
            optimizer_job: None,
            reload_job: None,
            opt_error: None,
        }
    }
}

/// The reconstruction evaluation screen (the algorithms come next); returns
/// `true` to go back to the setup screen.
fn recon_ui(
    ui: &mut egui::Ui,
    view: &mut ReconView,
    logo: Option<&egui::TextureHandle>,
    log_open: &mut bool,
) -> bool {
    let mut back = false;
    ui.horizontal(|ui| {
        if ui.button("↩ Back").clicked() {
            back = true;
        }
        ui.label(
            RichText::new(format!(
                "Evaluate the reconstruction — {}",
                view.stack
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default()
            ))
            .size(15.0)
            .strong(),
        );
        top_right_bar(ui, logo, log_open, 28.0);
    });
    ui.add_space(10.0);

    let stack = &view.stack;
    let angles: Vec<f64> = stack.sample.iter().filter_map(|p| p.angle_deg).collect();
    let dims = stack
        .sample
        .first()
        .map(|p| format!("{}x{}", p.height, p.width))
        .unwrap_or_default();
    let default_center = stack
        .sample
        .first()
        .map(|p| (p.width as f64 - 1.0) / 2.0)
        .unwrap_or(0.0);
    ui.label(
        RichText::new(format!(
            "{} projections ({dims}), angles {} — center of rotation: {}",
            stack.sample.len(),
            match (angles.first(), angles.last()) {
                (Some(a), Some(b)) => format!("{a:.3}° to {b:.3}°"),
                _ => "unknown".to_owned(),
            },
            match view.cor {
                Some(cor) => format!("{cor:.2} px (from the pre-processing)"),
                None => format!("{default_center:.1} px (horizontal center)"),
            }
        ))
        .strong(),
    );
    ui.add_space(6.0);
    egui::CollapsingHeader::new(RichText::new("Provenance").strong())
        .default_open(false)
        .show(ui, |ui| {
            for (name, value) in &stack.metadata {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{name}:")).strong().size(12.0));
                    ui.label(RichText::new(value).size(12.0));
                });
            }
        });
    ui.add_space(10.0);
    // The algorithms: one standalone evaluator per method, launched from
    // here; the saved parameters live in the checkpoint's metadata.
    let ctx = ui.ctx().clone();
    if let Some((label, rx)) = &view.optimizer_job {
        let label = label.clone();
        match rx.try_recv() {
            Ok(Ok(())) => {
                logger::log(format!("{label} evaluator closed — reloading the checkpoint"));
                view.optimizer_job = None;
                if view.stack.path.is_file() {
                    view.reload_job = Some(LoadJob::start(view.stack.path.clone()));
                }
            }
            Ok(Err(e)) => {
                logger::error(format!("{label} evaluator failed: {e}"));
                view.opt_error = Some(e);
                view.optimizer_job = None;
            }
            Err(_) => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(format!(
                        "{label} evaluator is open — tune, evaluate, then save the \
                         parameters there"
                    ));
                });
                ctx.request_repaint_after(Duration::from_millis(300));
            }
        }
    }
    if let Some(job) = &mut view.reload_job {
        match job.poll() {
            Some(Ok(stack)) => {
                view.cor = stack.center_of_rotation.or(view.cor);
                view.stack = std::sync::Arc::new(stack);
                view.reload_job = None;
            }
            Some(Err(e)) => {
                logger::error(format!("reloading the checkpoint failed: {e}"));
                view.opt_error = Some(e);
                view.reload_job = None;
            }
            None => ctx.request_repaint_after(Duration::from_millis(300)),
        }
    }

    ui.label(RichText::new("Reconstruction algorithms").strong());
    let on_disk = view.stack.path.is_file();
    if !on_disk {
        ui.colored_label(
            Color32::from_rgb(240, 180, 60),
            "the stack is not saved to a file yet — save the checkpoint first to \
             evaluate the algorithms",
        );
    }
    let busy = view.optimizer_job.is_some() || view.reload_job.is_some();
    let mut launch: Option<(&'static str, String)> = None;
    for algo in &RECON_ALGORITHMS {
        ui.horizontal(|ui| {
            let available = algo.binary.is_some();
            let response = ui
                .add_enabled(
                    available && on_disk && !busy,
                    egui::Button::new(format!(
                        "🧮 Evaluate the {} reconstruction",
                        algo.label
                    )),
                )
                .on_hover_text(algo.description)
                .on_disabled_hover_text(if available {
                    "save the checkpoint first"
                } else {
                    "the evaluator for this algorithm is not built yet"
                });
            if response.clicked()
                && let Some(binary) = algo.binary
            {
                launch = Some((binary, algo.label.to_owned()));
            }
            let config_key = format!("{}_config", algo.key);
            match view
                .stack
                .metadata
                .iter()
                .find(|(name, _)| *name == config_key)
            {
                Some((_, json)) => {
                    ui.label(
                        RichText::new(format!("saved parameters: {json}"))
                            .color(Color32::from_rgb(120, 200, 120))
                            .size(11.0),
                    );
                }
                None if available => {
                    ui.label(RichText::new("no saved parameters yet").weak().size(11.0));
                }
                None => {
                    ui.label(RichText::new("coming soon").weak().size(11.0));
                }
            }
        });
    }
    if let Some((binary, label)) = launch {
        logger::log(format!(
            "opening the {label} evaluator on {}",
            view.stack.path.display()
        ));
        view.opt_error = None;
        let path = view.stack.path.clone();
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let result = std::process::Command::new(binary)
                .arg(&path)
                .arg("--called-from-app")
                .output();
            let _ = tx.send(match result {
                Err(e) => Err(format!("cannot launch {binary}: {e}")),
                Ok(out) if !out.status.success() => Err(format!(
                    "evaluator failed ({}): {}",
                    out.status,
                    String::from_utf8_lossy(&out.stderr).trim()
                )),
                Ok(_) => Ok(()),
            });
        });
        view.optimizer_job = Some((label, rx));
    }
    if let Some(e) = &view.opt_error {
        ui.colored_label(Color32::LIGHT_RED, e);
    }
    back
}

/// `true` when a loaded file is a pre-processing checkpoint and can go
/// straight to the reconstruction evaluation.
fn stack_is_preprocessed(stack: &LoadedStack) -> bool {
    let stage = stack
        .metadata
        .iter()
        .find(|(name, _)| name == "processing_stage")
        .map(|(_, value)| value.as_str());
    match stage {
        Some(stage) => stage == "preprocessed",
        // Older files without the flag: normalized means pre-processed.
        None => stack.metadata.iter().any(|(name, _)| name == "normalization"),
    }
}

/// The accordion sections of the pre-processing screen — one open at a time.
#[derive(Clone, Copy, PartialEq, Eq)]
enum StackSection {
    Provenance,
    Crop,
    Clean,
    Normalize,
    Stripes,
    Rotate,
    Tilt,
    Cor,
    Sinogram,
    Log,
}

/// Shown on a section's header: whether the step was run.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SectionStatus {
    /// Informational section, no run state.
    NoStatus,
    /// The step was run (green check).
    Done,
    /// Optional step, not run.
    NotRun,
    /// Mandatory step, still to be run.
    Required,
}

struct StackView {
    /// Which accordion section is open (one at a time).
    open_section: Option<StackSection>,
    /// The stack as loaded — crops are always drawn on this one, so the
    /// region can be widened again later.
    original: std::sync::Arc<LoadedStack>,
    /// The current stack (cropped when a crop is applied) — what the
    /// pre-processing steps work on.
    stack: std::sync::Arc<LoadedStack>,
    crop: Option<CropRect>,
    crop_job: Option<CropJob>,
    crop_error: Option<String>,

    // Remove outliers.
    clean_settings: CleanSettings,
    clean_job: Option<CleanJob>,
    clean_stats: Option<CleanStats>,
    /// The stack before cleaning (i.e. the crop output), kept so the
    /// cleaning can be re-run with different settings.
    uncleaned: Option<std::sync::Arc<LoadedStack>>,
    /// Show the OB histogram instead of the sample one.
    hist_show_ob: bool,
    /// Edge-trimmed summed images for the histogram plots, keyed by
    /// (stack pointer, is_ob). A small FIFO so before/after × sample/ob fit.
    sum_cache: Vec<((usize, bool), Vec<f32>)>,
    sum_jobs: Vec<((usize, bool), std::sync::mpsc::Receiver<Vec<f32>>)>,
    /// Histograms of the summed images: key (stack, is_ob, bins, range
    /// fingerprint — 0 for the data's own min/max).
    hist_cache: Vec<((usize, bool, usize, u64), (f64, f64, Vec<u64>))>,

    // Normalization (mandatory).
    norm_settings: NormSettings,
    roi_job: Option<RoiJob>,
    norm_job: Option<NormJob>,
    normalized: bool,
    norm_summary: Option<String>,
    norm_error: Option<String>,
    visualize_job: Option<VisualizeJob>,

    // Rotation (after normalization; the rotation axis must be vertical).
    /// The normalized, un-rotated baseline every rotation starts from.
    unrotated: Option<std::sync::Arc<LoadedStack>>,
    /// Selected rotation, in quarter turns clockwise (0..=3).
    rotation_quarters: usize,
    /// Rotation currently applied to the stack, in quarter turns.
    rotation_applied: usize,
    rotate_job: Option<RotateJob>,
    /// Frame shown by the rotation preview.
    rot_frame: usize,
    /// Preview texture, keyed by (baseline, frame, quarters).
    rot_tex: Option<((usize, usize, usize), egui::TextureHandle)>,

    // Sinogram view (after normalization).
    /// Detector row (slice) whose sinogram is shown.
    sino_row: usize,
    /// Sinogram texture, keyed by (stack, row).
    sino_tex: Option<((usize, usize), egui::TextureHandle)>,

    // Tilt correction (after normalization).
    /// Row range (y_top, y_bottom) the tilt fit samples.
    tilt_range: Option<(usize, usize)>,
    /// Frame shown by the tilt preview.
    tilt_frame: usize,
    /// Show the preview with the estimated correction applied.
    tilt_preview_corrected: bool,
    /// Preview texture, keyed by (stack, frame, corrected, result bits).
    tilt_tex: Option<((usize, usize, bool, u64), egui::TextureHandle)>,
    tilt_calc: Option<TiltCalcJob>,
    tilt_result: Option<TiltResult>,
    tilt_apply: Option<TiltApplyJob>,
    tilt_applied: Option<TiltResult>,
    tilt_error: Option<String>,

    // Center of rotation (optional; the horizontal center otherwise).
    /// Slice (row) the estimation compares the 0°/180° projections on.
    cor_slice: Option<usize>,
    cor_job: Option<CorJob>,
    /// Calculated center of rotation; `None` = use the horizontal center.
    cor_result: Option<f64>,
    cor_error: Option<String>,
    /// Frame shown by the preview (`None` = the 0° projection).
    cor_frame: Option<usize>,
    /// Show the 0° + mirrored-180° overlay instead of a single image.
    cor_overlay: bool,
    /// Preview texture, keyed by (stack, frame or MAX for overlay, cor bits).
    cor_tex: Option<((usize, usize, u64), egui::TextureHandle)>,

    // Remove stripes (after normalization; tomopy via the pixi python).
    stripe_algos: Vec<StripeAlgo>,
    /// Row band (y0, y1) the test run works on.
    stripe_range: Option<(usize, usize)>,
    stripe_test_job: Option<StripeTestJob>,
    /// Test result: (n, band_h, w, y0, before, after) volumes.
    stripe_test: Option<(usize, usize, usize, usize, Vec<f32>, Vec<f32>)>,
    /// Row (absolute) whose before/after sinograms are shown.
    stripe_test_row: usize,
    stripe_test_tex: Option<((usize, usize), egui::TextureHandle, egui::TextureHandle)>,
    stripe_apply_job: Option<StripeApplyJob>,
    /// The stack before the stripe removal, kept so it can be re-run.
    unstriped: Option<std::sync::Arc<LoadedStack>>,
    stripes_applied: Option<String>,
    stripe_error: Option<String>,

    // Log conversion (mandatory before saving): transmission -> -log.
    log_job: Option<LogConvertJob>,
    log_converted: bool,
    log_stats: Option<LogStats>,
    log_visualize_job: Option<VisualizeJob>,
    /// The transmission stack, kept so the conversion can be undone.
    unlogged: Option<std::sync::Arc<LoadedStack>>,

    // Saving the pre-processing checkpoint.
    stack_save_job: Option<StackSaveJob>,
    stack_save_status: Option<Result<String, String>>,
    /// Set by the "evaluate the reconstruction" button; the main loop picks
    /// it up and switches screens.
    goto_recon: bool,
}

impl StackView {
    fn new(stack: LoadedStack) -> Self {
        let stack = std::sync::Arc::new(stack);
        let mut view = Self {
            open_section: Some(StackSection::Normalize),
            original: std::sync::Arc::clone(&stack),
            stack,
            crop: None,
            crop_job: None,
            crop_error: None,
            clean_settings: CleanSettings::default(),
            clean_job: None,
            clean_stats: None,
            uncleaned: None,
            hist_show_ob: false,
            sum_cache: Vec::new(),
            sum_jobs: Vec::new(),
            hist_cache: Vec::new(),
            norm_settings: NormSettings::default(),
            roi_job: None,
            norm_job: None,
            normalized: false,
            norm_summary: None,
            norm_error: None,
            visualize_job: None,
            unrotated: None,
            rotation_quarters: 0,
            rotation_applied: 0,
            rotate_job: None,
            rot_frame: 0,
            rot_tex: None,
            sino_row: 0,
            sino_tex: None,
            tilt_range: None,
            tilt_frame: 0,
            tilt_preview_corrected: false,
            tilt_tex: None,
            tilt_calc: None,
            tilt_result: None,
            tilt_apply: None,
            tilt_applied: None,
            tilt_error: None,
            cor_slice: None,
            cor_job: None,
            cor_result: None,
            cor_error: None,
            cor_frame: None,
            cor_overlay: false,
            cor_tex: None,
            stripe_algos: stripes::default_algorithms(),
            stripe_range: None,
            stripe_test_job: None,
            stripe_test: None,
            stripe_test_row: 0,
            stripe_test_tex: None,
            stripe_apply_job: None,
            unstriped: None,
            stripes_applied: None,
            stripe_error: None,
            log_job: None,
            log_converted: false,
            log_stats: None,
            log_visualize_job: None,
            unlogged: None,
            stack_save_job: None,
            stack_save_status: None,
            goto_recon: false,
        };

        // A pre-processing checkpoint restores its state: a stack saved
        // after normalization reopens ready for the next steps.
        let meta = |name: &str| -> Option<String> {
            view.stack
                .metadata
                .iter()
                .find(|(k, _)| k == name)
                .map(|(_, v)| v.clone())
        };
        if let Some(desc) = meta("normalization") {
            view.normalized = true;
            view.unrotated = Some(std::sync::Arc::clone(&view.stack));
            view.norm_summary = Some(format!("restored from the loaded file ({desc})"));
            view.open_section = Some(StackSection::Rotate);
        }
        if let Some(rotation) = meta("rotation")
            && let Some(deg) = rotation
                .split('°')
                .next()
                .and_then(|d| d.trim().parse::<usize>().ok())
        {
            view.rotation_quarters = (deg / 90) % 4;
            view.rotation_applied = view.rotation_quarters;
        }
        if let Some(tilt) = meta("tilt_correction") {
            // "tilt {:.4} deg, axis shift {} px (…)"
            let mut words = tilt.split_whitespace();
            let deg = words.nth(1).and_then(|v| v.parse::<f64>().ok());
            let shift = tilt
                .split("shift")
                .nth(1)
                .and_then(|rest| rest.split_whitespace().next())
                .and_then(|v| v.parse::<i64>().ok());
            if let (Some(tilt_deg), Some(shift_px)) = (deg, shift) {
                view.tilt_applied = Some(TiltResult {
                    tilt_deg,
                    shift_px,
                    slope: 2.0 * tilt_deg.to_radians().tan(),
                    intercept: 0.0,
                    rows_used: 0,
                });
            }
        }
        if let Some(desc) = meta("remove_stripes") {
            view.stripes_applied = Some(desc);
        }
        if meta("log_conversion").is_some() {
            view.log_converted = true;
        }
        // The numeric dataset is authoritative; older checkpoints may only
        // have a metadata string.
        if let Some(cor) = view.stack.center_of_rotation {
            view.cor_result = Some(cor);
        } else if let Some(cor) = meta("center_of_rotation")
            && let Ok(value) = cor.trim().parse::<f64>()
        {
            view.cor_result = Some(value);
        }
        view
    }

    fn clear_log(&mut self) {
        self.log_job = None;
        self.log_converted = false;
        self.log_stats = None;
        self.log_visualize_job = None;
        self.unlogged = None;
    }

    fn clear_stripes(&mut self) {
        self.stripe_range = None;
        self.stripe_test_job = None;
        self.stripe_test = None;
        self.stripe_test_tex = None;
        self.stripe_apply_job = None;
        self.unstriped = None;
        self.stripes_applied = None;
        self.stripe_error = None;
    }

    fn clear_cor(&mut self) {
        self.cor_slice = None;
        self.cor_job = None;
        self.cor_result = None;
        self.cor_error = None;
        self.cor_frame = None;
        self.cor_overlay = false;
        self.cor_tex = None;
    }

    fn clear_tilt(&mut self) {
        self.tilt_range = None;
        self.tilt_frame = 0;
        self.tilt_preview_corrected = false;
        self.tilt_tex = None;
        self.tilt_calc = None;
        self.tilt_result = None;
        self.tilt_apply = None;
        self.tilt_applied = None;
        self.tilt_error = None;
    }

    /// What the next cleaning run starts from: the pre-cleaning stack when a
    /// cleaning was already applied, the current stack otherwise.
    fn clean_input(&self) -> std::sync::Arc<LoadedStack> {
        self.uncleaned
            .clone()
            .unwrap_or_else(|| std::sync::Arc::clone(&self.stack))
    }
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

    // Data to use.
    use_percentage: bool,
    /// 1–100, meaningful when `use_percentage` (same 50% default as the
    /// Python pipeline).
    percentage: u8,
    meta_job: Option<MetaAnglesScan>,
    /// Metadata angles cached per selection: (file count, first file) key.
    meta_angles: Option<(usize, Option<PathBuf>, Vec<Result<f64, String>>)>,

    // Exclude images.
    exclude_text: String,
    excluded_runs: HashSet<u32>,
    exclude_error: Option<String>,
    intensity_job: Option<IntensityScan>,
    intensities: Option<IntensityCache>,
    /// Images with an intensity below this are excluded.
    threshold: f64,
    threshold_bounds: (f64, f64),
    threshold_dragging: bool,
    /// Show the intensity plot with a log10 y axis.
    intensity_log: bool,

    // Reading + stacking the final selection, and saving it to HDF5.
    process: Option<CombineScan>,
    processed: Option<std::sync::Arc<CombineOutput>>,
    save_job: Option<SaveJob>,
    save_status: Option<Result<String, String>>,
    /// Set by the "continue to pre-processing" button; the main loop picks
    /// it up and switches to the pre-processing screen.
    goto_preprocess: Option<LoadedStack>,
}

/// Integrated intensities cached per selection: the scan covers the used
/// files first, then the superseded old revisions; `values` stays aligned
/// with that input (`None` = unreadable image).
struct IntensityCache {
    total: usize,
    first: Option<PathBuf>,
    values: Vec<Option<ImageIntensity>>,
    kept_len: usize,
    failed: usize,
}

/// The projection angles of the current sample selection, per the chosen
/// retrieval method.
enum AngleData {
    /// Usable angles (in [0, 360)) and how many files yielded none.
    Ready(Vec<f64>, usize),
    /// Metadata read in progress: message to show next to the spinner.
    Pending(String),
    /// Not available yet: what the user still has to do.
    Invalid(String),
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
            use_percentage: false,
            percentage: 50,
            meta_job: None,
            meta_angles: None,
            exclude_text: String::new(),
            excluded_runs: HashSet::new(),
            exclude_error: None,
            intensity_job: None,
            intensities: None,
            threshold: 0.0,
            threshold_bounds: (0.0, 1.0),
            threshold_dragging: false,
            intensity_log: false,
            process: None,
            processed: None,
            save_job: None,
            save_status: None,
            goto_preprocess: None,
        }
    }

    /// The final sample selection: used files (highest revisions) that
    /// survive the manual run exclusions and the intensity threshold, with
    /// their angles, down-selected to the coverage percentage when that mode
    /// is on. `Err` explains what is still missing.
    fn final_selection(&self) -> Result<Vec<RunToCombine>, String> {
        let files: Vec<PathBuf> = self
            .sample
            .selected
            .iter()
            .flat_map(|(_, files, _)| files.iter().cloned())
            .collect();
        if files.is_empty() {
            return Err("select the sample folder(s) first".to_owned());
        }
        let angles = self
            .per_file_angles(&files)
            .ok_or("set up the projection angles first (and let the metadata read finish)")?;

        // Exclusions: manual run numbers, then the intensity threshold when
        // intensities were computed for this exact selection.
        let thresholded: Option<&IntensityCache> = self
            .intensities
            .as_ref()
            .filter(|c| c.kept_len == files.len() && c.first.as_ref() == files.first());
        let mut candidates: Vec<usize> = Vec::new();
        for (index, file) in files.iter().enumerate() {
            let run = tof::run_number(&file.file_name().unwrap_or_default().to_string_lossy());
            if run.is_some_and(|r| self.excluded_runs.contains(&r)) {
                continue;
            }
            if let Some(cache) = thresholded
                && let Some(Some(v)) = cache.values.get(index)
                && v.intensity < self.threshold
            {
                continue;
            }
            candidates.push(index);
        }
        if candidates.is_empty() {
            return Err("every image is excluded — nothing to save".to_owned());
        }

        // Coverage down-selection on what survived the exclusions.
        let survivors: Vec<usize> = if self.use_percentage {
            let candidate_angles: Vec<f64> = candidates
                .iter()
                .map(|&i| angles[i].unwrap_or(f64::MAX))
                .collect();
            let n = ((self.percentage as f64 / 100.0) * candidates.len() as f64).round()
                as usize;
            let n = n.max(5.min(candidates.len()));
            white_beam::select_coverage(&candidate_angles, n)
                .iter()
                .map(|&pos| candidates[pos])
                .collect()
        } else {
            candidates
        };

        Ok(survivors
            .iter()
            .map(|&i| {
                let file = &files[i];
                RunToCombine {
                    name: file
                        .file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    run_number: tof::run_number(
                        &file.file_name().unwrap_or_default().to_string_lossy(),
                    ),
                    images: vec![file.clone()],
                    angle_deg: angles[i],
                }
            })
            .collect())
    }

    /// The angle of each sample file (aligned with the flattened selection),
    /// per the configured retrieval method; `None` when the method is not
    /// ready yet (metadata not read, ASCII mismatch, fields not set up).
    fn per_file_angles(&self, files: &[PathBuf]) -> Option<Vec<Option<f64>>> {
        match self.angle_source {
            AngleSource::NamingConvention => {
                let picked: Vec<usize> = self
                    .nc_checked
                    .iter()
                    .enumerate()
                    .filter(|(_, c)| **c)
                    .map(|(i, _)| i)
                    .collect();
                let [i, j] = picked.as_slice() else { return None };
                Some(
                    files
                        .iter()
                        .map(|f| white_beam::angle_from_fields(f, *i, *j))
                        .collect(),
                )
            }
            AngleSource::AsciiFile => {
                let angles = self.ascii_angles.as_ref()?;
                (angles.len() == files.len())
                    .then(|| angles.iter().map(|a| Some(*a)).collect())
            }
            AngleSource::Metadata => {
                let (n, first, results) = self.meta_angles.as_ref()?;
                ((*n, first.as_ref()) == (files.len(), files.first()))
                    .then(|| results.iter().map(|r| r.as_ref().ok().copied()).collect())
            }
        }
    }

    /// First selected sample image — the naming-convention example and the
    /// metadata test subject.
    fn first_sample_image(&self) -> Option<&PathBuf> {
        self.sample
            .selected
            .iter()
            .flat_map(|(_, files, _)| files.iter())
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
    /// Selected folders (sorted by name): `(folder, used TIFFs, superseded
    /// TIFFs)` — superseded are older revisions of retaken projections,
    /// excluded automatically but kept visible.
    selected: Vec<(PathBuf, Vec<PathBuf>, Vec<PathBuf>)>,
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
        self.selected.iter().any(|(d, ..)| d == dir)
    }

    fn total_files(&self) -> usize {
        self.selected.iter().map(|(_, files, _)| files.len()).sum()
    }

    /// Add (recording its TIFF files) or remove one folder.
    fn toggle(&mut self, dir: PathBuf) {
        if let Some(i) = self.selected.iter().position(|(d, ..)| d == &dir) {
            logger::log(format!("{} folder removed: {}", self.kind, dir.display()));
            self.selected.remove(i);
            return;
        }
        match white_beam::tiff_files_in(&dir) {
            Ok((files, superseded)) => {
                logger::log(format!(
                    "{} folder selected: {} ({} tiff images{})",
                    self.kind,
                    dir.display(),
                    files.len(),
                    if superseded.is_empty() {
                        String::new()
                    } else {
                        format!(
                            ", {} older revision(s) excluded automatically",
                            superseded.len()
                        )
                    }
                ));
                self.selected.push((dir, files, superseded));
                self.selected.sort_by(|(a, ..), (b, ..)| a.cmp(b));
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
            for (dir, files, superseded) in &self.selected {
                let name = dir
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| dir.display().to_string());
                let revisions = if superseded.is_empty() {
                    String::new()
                } else {
                    format!(" (+{} old revisions)", superseded.len())
                };
                let text =
                    RichText::new(format!("{name} — {} tiff{revisions}", files.len())).size(12.0);
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
    /// Set by the "continue to pre-processing" button; the main loop picks
    /// it up and switches to the pre-processing screen.
    goto_preprocess: Option<LoadedStack>,
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
            goto_preprocess: None,
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

    /// Loading a previously saved HDF5 from the setup screen; jumps to the
    /// pre-processing screen when it finishes.
    load_job: Option<LoadJob>,
    load_error: Option<String>,

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
            load_job: None,
            load_error: None,
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
            ui.add_space(16.0);
            self.load_stack_row(ui);
        });
        self.next_button(ui)
    }

    /// "— or —" load a previously saved HDF5 and jump straight to the
    /// pre-processing screen.
    fn load_stack_row(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("— or —").weak().size(12.0));
        ui.add_space(4.0);
        if let Some(_job) = &self.load_job {
            ui.horizontal(|ui| {
                ui.add_space((ui.available_width() / 2.0 - 120.0).max(0.0));
                ui.spinner();
                ui.label("loading the HDF5 stack…");
            });
            return;
        }
        if ui
            .button("📂 Load a previously saved HDF5…")
            .clicked()
        {
            let mut dialog = rfd::FileDialog::new()
                .set_title("Select a previously saved projections HDF5")
                .add_filter("HDF5", &["h5", "hdf5"]);
            if let Some(entry) = &self.selected {
                let shared = entry.path.join("shared");
                if shared.is_dir() {
                    dialog = dialog.set_directory(shared);
                }
            }
            if let Some(path) = dialog.pick_file() {
                logger::log(format!("loading saved stack: {}", path.display()));
                self.load_error = None;
                self.load_job = Some(LoadJob::start(path));
            }
        }
        if let Some(e) = &self.load_error {
            ui.colored_label(Color32::LIGHT_RED, e);
        }
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

/// Pre-processing of a loaded stack of projections; returns `true` to go
/// back to the setup screen.
fn stack_ui(
    ui: &mut egui::Ui,
    view: &mut StackView,
    logo: Option<&egui::TextureHandle>,
    log_open: &mut bool,
) -> bool {
    let mut back = false;
    let title_name = view
        .stack
        .path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    ui.horizontal(|ui| {
        if ui.button("↩ Back").clicked() {
            back = true;
        }
        ui.label(
            RichText::new(format!("Pre-processing — {title_name}"))
                .size(15.0)
                .strong(),
        );
        let busy =
            view.crop_job.is_some() || view.clean_job.is_some() || view.norm_job.is_some();
        let touched = view.crop.is_some() || view.clean_stats.is_some() || view.normalized;
        if ui
            .add_enabled(!busy && touched, egui::Button::new("🔄 Reset to the loaded data"))
            .on_hover_text(
                "discard the crop and outlier removal and start over from the data set \
                 this screen was opened with",
            )
            .clicked()
        {
            logger::log("pre-processing reset: back to the loaded data set");
            view.stack = std::sync::Arc::clone(&view.original);
            view.crop = None;
            view.crop_error = None;
            view.uncleaned = None;
            view.clean_stats = None;
            view.sum_cache.clear();
            view.sum_jobs.clear();
            view.hist_cache.clear();
            view.normalized = false;
            view.norm_summary = None;
            view.norm_error = None;
            view.norm_settings.roi = None;
            view.unrotated = None;
            view.rotation_quarters = 0;
            view.rotation_applied = 0;
            view.rot_tex = None;
            view.clear_tilt();
            view.clear_stripes();
            view.clear_log();
            view.clear_cor();
        }
        top_right_bar(ui, logo, log_open, 28.0);
    });
    ui.add_space(10.0);

    let stack = &view.stack;
    let angles: Vec<f64> = stack.sample.iter().filter_map(|p| p.angle_deg).collect();
    let dims = stack
        .sample
        .first()
        .map(|p| format!("{}x{}", p.height, p.width))
        .unwrap_or_default();
    ui.label(
        RichText::new(format!(
            "{} projections ({dims}), angles {} — {} ob image(s)",
            stack.sample.len(),
            match (angles.first(), angles.last()) {
                (Some(a), Some(b)) => format!("{a:.3}° to {b:.3}°"),
                _ => "unknown".to_owned(),
            },
            stack.ob.len()
        ))
        .strong(),
    );
    ui.add_space(6.0);
    // Accordion: exactly one section open at a time; clicking an open
    // section's header collapses it.
    let mut clicked: Option<StackSection> = None;
    let mut section =
        |ui: &mut egui::Ui,
         view: &mut StackView,
         which: StackSection,
         title: &str,
         status: SectionStatus,
         body: &mut dyn FnMut(&mut egui::Ui, &mut StackView)| {
            let header = match status {
                SectionStatus::NoStatus => RichText::new(title).strong(),
                SectionStatus::Done => RichText::new(format!("✔ {title}"))
                    .strong()
                    .color(Color32::from_rgb(120, 200, 120)),
                SectionStatus::NotRun => RichText::new(format!("{title} — not run"))
                    .strong()
                    .color(Color32::from_gray(150)),
                SectionStatus::Required => RichText::new(format!("{title} — required"))
                    .strong()
                    .color(Color32::from_rgb(240, 180, 60)),
            };
            let open = view.open_section == Some(which);
            let response = egui::CollapsingHeader::new(header)
                .open(Some(open))
                .show(ui, |ui| body(ui, view));
            if response.header_response.clicked() {
                clicked = Some(which);
            }
            ui.add_space(4.0);
        };
    let run_status = |ran: bool| {
        if ran {
            SectionStatus::Done
        } else {
            SectionStatus::NotRun
        }
    };
    section(
        ui,
        view,
        StackSection::Provenance,
        "Provenance",
        SectionStatus::NoStatus,
        &mut |ui, view| {
            for (name, value) in &view.stack.metadata {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{name}:")).strong().size(12.0));
                    ui.label(RichText::new(value).size(12.0));
                });
            }
        },
    );
    section(
        ui,
        view,
        StackSection::Crop,
        "Crop",
        run_status(view.crop.is_some()),
        &mut |ui, view| {
            crop_section_ui(ui, view);
        },
    );
    section(
        ui,
        view,
        StackSection::Clean,
        "Remove outliers",
        run_status(view.clean_stats.is_some()),
        &mut |ui, view| {
            clean_section_ui(ui, view);
        },
    );
    section(
        ui,
        view,
        StackSection::Normalize,
        "Normalization (mandatory)",
        if view.normalized {
            SectionStatus::Done
        } else {
            SectionStatus::Required
        },
        &mut |ui, view| {
            normalization_section_ui(ui, view);
        },
    );
    if view.normalized {
        section(
            ui,
            view,
            StackSection::Stripes,
            "Remove stripes",
            run_status(view.stripes_applied.is_some()),
            &mut |ui, view| {
                stripes_section_ui(ui, view);
            },
        );
        section(
            ui,
            view,
            StackSection::Rotate,
            "Rotate the data",
            run_status(view.rotation_applied != 0),
            &mut |ui, view| {
                rotation_section_ui(ui, view);
            },
        );
        section(
            ui,
            view,
            StackSection::Tilt,
            "Tilt correction",
            run_status(view.tilt_applied.is_some()),
            &mut |ui, view| {
                tilt_section_ui(ui, view);
            },
        );
        section(
            ui,
            view,
            StackSection::Cor,
            "Center of rotation",
            run_status(view.cor_result.is_some()),
            &mut |ui, view| {
                cor_section_ui(ui, view);
            },
        );
        section(
            ui,
            view,
            StackSection::Sinogram,
            "Sinogram",
            SectionStatus::NoStatus,
            &mut |ui, view| {
                sinogram_section_ui(ui, view);
            },
        );
        section(
            ui,
            view,
            StackSection::Log,
            "Log conversion (mandatory)",
            if view.log_converted {
                SectionStatus::Done
            } else {
                SectionStatus::Required
            },
            &mut |ui, view| {
                log_section_ui(ui, view);
            },
        );
    }
    if let Some(which) = clicked {
        view.open_section = if view.open_section == Some(which) {
            None
        } else {
            Some(which)
        };
    }
    // Checkpoint: save the pre-processed stack so a later session can load
    // it and start directly at the reconstruction step.
    if view.normalized {
        ui.separator();
        if let Some(job) = &mut view.stack_save_job {
            match job.poll() {
                Some(Ok(msg)) => {
                    logger::log(format!("pre-processing checkpoint saved: {msg}"));
                    view.stack_save_status = Some(Ok(msg));
                    view.stack_save_job = None;
                }
                Some(Err(e)) => {
                    logger::error(format!("saving the checkpoint failed: {e}"));
                    view.stack_save_status = Some(Err(e));
                    view.stack_save_job = None;
                }
                None => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("writing the checkpoint HDF5…");
                    });
                    ui.ctx().request_repaint_after(Duration::from_millis(300));
                }
            }
        }
        let savable = view.log_converted;
        if !savable {
            ui.label(
                RichText::new(
                    "the log conversion (last section) is mandatory before saving or \
                     evaluating the reconstruction",
                )
                .color(Color32::from_rgb(240, 180, 60)),
            );
        }
        ui.horizontal(|ui| {
            if ui
                .add_enabled(
                    savable && view.stack_save_job.is_none(),
                    egui::Button::new("💾 Save the pre-processed stack…"),
                )
                .on_hover_text(
                    "writes an HDF5 checkpoint (with the whole provenance, including the \
                     center of rotation) that can be loaded later from the setup screen to \
                     start directly at the reconstruction step",
                )
                .clicked()
            {
                let default_name = view
                    .stack
                    .path
                    .file_stem()
                    .map(|n| {
                        format!(
                            "{}_preprocessed.h5",
                            n.to_string_lossy().trim_end_matches("_preprocessed")
                        )
                    })
                    .unwrap_or_else(|| "ct_preprocessed.h5".to_owned());
                let mut dialog = rfd::FileDialog::new()
                    .set_title("Save the pre-processed stack")
                    .add_filter("HDF5", &["h5", "hdf5"])
                    .set_file_name(default_name);
                if let Some(dir) = view.stack.path.parent().filter(|p| p.is_dir()) {
                    dialog = dialog.set_directory(dir);
                }
                if let Some(path) = dialog.save_file() {
                    logger::log(format!(
                        "saving the pre-processing checkpoint to {} (cor: {:?})",
                        path.display(),
                        view.cor_result
                    ));
                    view.stack_save_status = None;
                    view.stack_save_job = Some(StackSaveJob::start(
                        path,
                        std::sync::Arc::clone(&view.stack),
                        view.cor_result,
                        Vec::new(),
                    ));
                }
            }
            if ui
                .add_enabled(savable, egui::Button::new("🚀 Evaluate the reconstruction"))
                .on_hover_text("continue with this stack right away — saving is optional")
                .clicked()
            {
                view.goto_recon = true;
            }
            ui.label(
                RichText::new("loading a saved checkpoint later starts directly at the \
                               reconstruction")
                    .weak()
                    .size(11.0),
            );
        });
        match &view.stack_save_status {
            Some(Ok(msg)) => {
                ui.colored_label(Color32::from_rgb(120, 200, 120), format!("saved: {msg}"));
            }
            Some(Err(e)) => {
                ui.colored_label(Color32::LIGHT_RED, format!("save failed: {e}"));
            }
            None => {}
        }
    }
    ui.add_space(24.0);
    back
}

/// "Remove outliers": the three cleaning methods of the Python
/// ImagesCleaner (in-house histogram, tomopy remove_outlier, scipy median
/// filter), applied to the sample and open-beam stacks.
fn clean_section_ui(ui: &mut egui::Ui, view: &mut StackView) {
    let ctx = ui.ctx().clone();
    // Fold a finished cleaning run into the view.
    if let Some(job) = &mut view.clean_job {
        if let Some((cleaned, stats)) = job.poll() {
            logger::log(format!(
                "outliers removed ({}): {} in-house, {} tomopy replacements{}",
                view.clean_settings.describe(),
                stats.in_house_replaced,
                stats.tomopy_replaced,
                if stats.scipy_applied {
                    ", scipy filter applied"
                } else {
                    ""
                }
            ));
            if view.uncleaned.is_none() {
                view.uncleaned = Some(std::sync::Arc::clone(&view.stack));
            }
            view.stack = std::sync::Arc::new(cleaned);
            view.clean_stats = Some(stats);
            view.clean_job = None;
        } else {
            let done = job.done();
            let frac = (done as f32 / job.total.max(1) as f32).min(1.0);
            ui.add(egui::ProgressBar::new(frac).text(format!("{done}/{} images", job.total)));
            ctx.request_repaint_after(Duration::from_millis(300));
            return;
        }
    }

    let settings = &mut view.clean_settings;
    ui.checkbox(&mut settings.in_house, "In-house (histogram)");
    if settings.in_house {
        ui.indent("in_house_settings", |ui| {
            ui.horizontal(|ui| {
                ui.add(
                    egui::Slider::new(&mut settings.nbr_bins, 10..=1000).text("bins"),
                );
                ui.label("exclude bins:");
                ui.add(egui::DragValue::new(&mut settings.exclude_left).range(0..=50));
                ui.label("left,");
                ui.add(egui::DragValue::new(&mut settings.exclude_right).range(0..=50));
                ui.label("right");
                ui.label("radius:");
                ui.add(egui::DragValue::new(&mut settings.correct_radius).range(1..=5))
                    .on_hover_text("replacement median window is (2r+1) x (2r+1)");
            });
        });
    }
    ui.checkbox(&mut settings.tomopy, "Tomopy (remove_outlier)");
    if settings.tomopy {
        ui.indent("tomopy_settings", |ui| {
            ui.add(
                egui::Slider::new(&mut settings.tomopy_diff, 1.0..=100.0).text("diff value"),
            )
            .on_hover_text(
                "pixels brighter than the 3x3 median of their image by more than this \
                 are replaced by that median",
            );
        });
    }
    ui.checkbox(&mut settings.scipy, "Scipy (median_filter)")
        .on_hover_text("every pixel is replaced by the 3x3 median of its image");

    // Histogram of the edge-trimmed summed image, sample or ob, before and
    // (once cleaned) after — the excluded bins of the in-house method in red.
    let ob_available = !view.stack.ob.is_empty();
    ui.horizontal(|ui| {
        ui.label(RichText::new("Histogram:").strong());
        if ui.selectable_label(!view.hist_show_ob, "sample").clicked() {
            view.hist_show_ob = false;
        }
        if ui
            .add_enabled(
                ob_available,
                egui::Button::selectable(view.hist_show_ob, "ob"),
            )
            .clicked()
        {
            view.hist_show_ob = true;
        }
        if !ob_available {
            ui.label(RichText::new("(no ob in this stack)").weak().size(11.0));
        }
    });
    let use_ob = view.hist_show_ob && ob_available;
    let before = view.clean_input();
    let cleaned = (view.clean_stats.is_some() && view.uncleaned.is_some())
        .then(|| std::sync::Arc::clone(&view.stack));
    match cleaned {
        None => {
            stack_histogram_ui(ui, &ctx, view, &before, use_ob, "before cleaning", true, None);
        }
        Some(after) => {
            ui.columns(2, |cols| {
                stack_histogram_ui(
                    &mut cols[0],
                    &ctx,
                    view,
                    &before,
                    use_ob,
                    "before cleaning",
                    true,
                    None,
                );
                // Same bin edges as the before histogram, so the two compare
                // bin for bin (only the red-marked tails should empty out).
                let bins = view.clean_settings.nbr_bins;
                let before_key = (std::sync::Arc::as_ptr(&before) as usize, use_ob, bins, 0u64);
                let before_range = view
                    .hist_cache
                    .iter()
                    .find(|(k, _)| *k == before_key)
                    .map(|(_, (min, max, _))| (*min, *max));
                stack_histogram_ui(
                    &mut cols[1],
                    &ctx,
                    view,
                    &after,
                    use_ob,
                    "after cleaning",
                    false,
                    before_range,
                );
            });
        }
    }

    let busy = view.clean_job.is_some();
    let ready = view.clean_settings.any_enabled() && !busy;
    if ui
        .add_enabled(ready, egui::Button::new("▶ Remove the outliers"))
        .clicked()
    {
        let input = view.clean_input();
        logger::log(format!(
            "removing outliers ({}) on {} sample + {} ob images…",
            view.clean_settings.describe(),
            input.sample.len(),
            input.ob.len()
        ));
        if view.uncleaned.is_none() {
            view.uncleaned = Some(std::sync::Arc::clone(&input));
        }
        view.clean_stats = None;
        view.clean_job = Some(CleanJob::start(input, view.clean_settings.clone()));
    }
    if !view.clean_settings.any_enabled() {
        ui.label(RichText::new("select at least one method").weak());
    }
    if let Some(stats) = &view.clean_stats {
        let mut parts = Vec::new();
        if view.clean_settings.in_house {
            parts.push(format!("{} pixels by the in-house histogram", stats.in_house_replaced));
        }
        if view.clean_settings.tomopy {
            parts.push(format!("{} pixels by tomopy", stats.tomopy_replaced));
        }
        if stats.scipy_applied {
            parts.push("every pixel median-filtered by scipy".to_owned());
        }
        ui.label(
            RichText::new(format!("cleaned (sample + ob): {}", parts.join(", "))).strong(),
        );
        ui.label(
            RichText::new("re-running applies the current settings to the pre-cleaning stack")
                .weak()
                .size(11.0),
        );
    }
}

/// Histogram of the edge-trimmed summed image (sample or ob) of one stack,
/// computed once per (stack, data type) on a background thread. The excluded
/// bins of the in-house method are drawn in red.
#[allow(clippy::too_many_arguments)]
fn stack_histogram_ui(
    ui: &mut egui::Ui,
    ctx: &egui::Context,
    view: &mut StackView,
    stack: &std::sync::Arc<LoadedStack>,
    use_ob: bool,
    title: &str,
    mark_exclusions: bool,
    range: Option<(f64, f64)>,
) {
    const EDGE_TRIM: usize = 10;
    ui.label(RichText::new(title).strong().size(13.0));
    let projections = if use_ob { &stack.ob } else { &stack.sample };
    if projections.is_empty() {
        ui.label(RichText::new("no data").weak());
        return;
    }
    let key = (std::sync::Arc::as_ptr(stack) as usize, use_ob);

    // Summed image: cached, else polled from its job, else started.
    let cached = view.sum_cache.iter().position(|(k, _)| *k == key);
    let index = match cached {
        Some(index) => index,
        None => {
            if let Some(pos) = view.sum_jobs.iter().position(|(k, _)| *k == key) {
                match view.sum_jobs[pos].1.try_recv() {
                    Ok(values) => {
                        view.sum_jobs.remove(pos);
                        if view.sum_cache.len() >= 4 {
                            view.sum_cache.remove(0);
                        }
                        view.sum_cache.push((key, values));
                        view.sum_cache.len() - 1
                    }
                    Err(_) => {
                        ui.horizontal(|ui| {
                            ui.spinner();
                            ui.label("summing the stack for the histogram…");
                        });
                        ctx.request_repaint_after(Duration::from_millis(300));
                        return;
                    }
                }
            } else {
                let (tx, rx) = std::sync::mpsc::channel();
                let stack = std::sync::Arc::clone(stack);
                std::thread::spawn(move || {
                    let projections = if use_ob { &stack.ob } else { &stack.sample };
                    let Some(first) = projections.first() else {
                        let _ = tx.send(Vec::new());
                        return;
                    };
                    let (w, h) = (first.width, first.height);
                    let mut sum = vec![0.0f64; w * h];
                    for p in projections {
                        for (acc, v) in sum.iter_mut().zip(&p.mean) {
                            *acc += f64::from(*v);
                        }
                    }
                    let trim = EDGE_TRIM.min(w.saturating_sub(1) / 2).min(h.saturating_sub(1) / 2);
                    let mut values =
                        Vec::with_capacity((h - 2 * trim).saturating_mul(w - 2 * trim));
                    for y in trim..h - trim {
                        for x in trim..w - trim {
                            values.push(sum[y * w + x] as f32);
                        }
                    }
                    let _ = tx.send(values);
                });
                view.sum_jobs.push((key, rx));
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("summing the stack for the histogram…");
                });
                ctx.request_repaint_after(Duration::from_millis(300));
                return;
            }
        }
    };
    if view.sum_cache[index].1.is_empty() {
        ui.label(RichText::new("no data").weak());
        return;
    }

    let bins = view.clean_settings.nbr_bins;
    // A fixed range (the before histogram's edges) is part of the key so a
    // full-range entry cached earlier cannot shadow it.
    let range_fingerprint = range
        .map(|(lo, hi)| lo.to_bits() ^ hi.to_bits().rotate_left(1))
        .unwrap_or(0);
    let hist_key = (key.0, key.1, bins, range_fingerprint);
    if !view.hist_cache.iter().any(|(k, _)| *k == hist_key) {
        let values = &view.sum_cache[index].1;
        let (min, max, counts) = match range {
            Some((lo, hi)) => clean::histogram_range(values, bins, lo, hi),
            None => clean::histogram(values, bins),
        };
        if view.hist_cache.len() >= 8 {
            view.hist_cache.remove(0);
        }
        view.hist_cache.push((hist_key, (min, max, counts)));
    }
    let Some((_, (min, max, counts))) = view.hist_cache.iter().find(|(k, _)| *k == hist_key)
    else {
        return;
    };
    let bin_width = (max - min) / counts.len() as f64;
    let mark_exclusions = mark_exclusions && view.clean_settings.in_house;
    let left = view.clean_settings.exclude_left;
    let right = view.clean_settings.exclude_right;
    let bars: Vec<egui_plot::Bar> = counts
        .iter()
        .enumerate()
        .map(|(k, count)| {
            let excluded =
                mark_exclusions && (k < left || k >= counts.len().saturating_sub(right));
            egui_plot::Bar::new(min + (k as f64 + 0.5) * bin_width, (*count as f64 + 1.0).log10())
                .width(bin_width)
                .fill(if excluded {
                    Color32::from_rgb(230, 100, 100)
                } else {
                    Color32::from_rgb(100, 170, 255)
                })
        })
        .collect();
    egui_plot::Plot::new(format!("clean_histogram_{title}_{use_ob}"))
        .height(180.0)
        .x_axis_label("summed image value")
        .y_axis_label("log10(count + 1)")
        .show(ui, |plot_ui| {
            plot_ui.bar_chart(egui_plot::BarChart::new("histogram", bars));
        });
}

/// Normalization — the mandatory step: NeuNorm division by the (mean) open
/// beam, with an optional shared ROI powering the beam-fluctuation
/// correction and/or the sample-ROI median anchor.
fn normalization_section_ui(ui: &mut egui::Ui, view: &mut StackView) {
    let ctx = ui.ctx().clone();
    // Fold finished background work into the view.
    if let Some(job) = &mut view.roi_job {
        match job.poll() {
            Some(Ok(Some(rect))) => {
                logger::log(format!(
                    "normalization ROI selected: x={}, y={}, {}x{}",
                    rect.x, rect.y, rect.width, rect.height
                ));
                view.norm_settings.roi = Some(rect);
                view.norm_error = None;
                view.roi_job = None;
            }
            Some(Ok(None)) => {
                logger::log("ROI selector closed without saving a selection");
                view.roi_job = None;
            }
            Some(Err(e)) => {
                logger::error(format!("ROI selection failed: {e}"));
                view.norm_error = Some(e);
                view.roi_job = None;
            }
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(
                        "ROI selector is open — draw a region AWAY from the sample and \
                         press its save/return button",
                    );
                });
                ctx.request_repaint_after(Duration::from_millis(300));
                return;
            }
        }
    }
    if let Some(job) = &mut view.norm_job {
        match job.poll() {
            Some(Ok((normalized, summary))) => {
                logger::log(format!("normalization done: {summary}"));
                view.stack = std::sync::Arc::new(normalized);
                view.normalized = true;
                view.norm_summary = Some(summary);
                view.norm_error = None;
                view.norm_job = None;
                // The rotation step starts from this normalized baseline,
                // and the accordion moves on to it.
                view.unrotated = Some(std::sync::Arc::clone(&view.stack));
                view.rotation_quarters = 0;
                view.rotation_applied = 0;
                view.rot_tex = None;
                view.clear_tilt();
                view.clear_stripes();
                view.clear_log();
                view.clear_cor();
                view.open_section = Some(StackSection::Rotate);
            }
            Some(Err(e)) => {
                logger::error(format!("normalization failed: {e}"));
                view.norm_error = Some(e);
                view.norm_job = None;
            }
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("NeuNorm is running…");
                });
                ctx.request_repaint_after(Duration::from_millis(300));
                return;
            }
        }
    }

    if view.stack.ob.is_empty() && !view.normalized {
        ui.colored_label(
            Color32::LIGHT_RED,
            "this stack has no open beam — normalization cannot run",
        );
        return;
    }
    if view.normalized {
        if let Some(summary) = &view.norm_summary {
            ui.label(
                RichText::new(format!("normalized ✔ — {summary}"))
                    .color(Color32::from_rgb(120, 200, 120)),
            );
        }
        // Visualize the normalized data in the sibling TIFF viewer.
        if let Some(job) = &mut view.visualize_job {
            match job.poll() {
                Some(Ok(())) => {
                    logger::log("normalized-data viewer closed");
                    view.visualize_job = None;
                }
                Some(Err(e)) => {
                    logger::error(format!("visualizing the normalized data failed: {e}"));
                    view.norm_error = Some(e);
                    view.visualize_job = None;
                }
                None => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("TIFF viewer is open on the normalized images");
                    });
                    ctx.request_repaint_after(Duration::from_millis(300));
                }
            }
        }
        if view.visualize_job.is_none()
            && ui
                .button("👁 Visualize the normalized data")
                .on_hover_text(
                    "opens the stack in rust_tiff_viewer, on the single-image view",
                )
                .clicked()
        {
            logger::log("opening the TIFF viewer on the normalized stack (single-image view)");
            view.norm_error = None;
            view.visualize_job = Some(VisualizeJob::start(std::sync::Arc::clone(&view.stack)));
        }
        ui.label(
            RichText::new("use the Reset button in the header to normalize differently")
                .weak()
                .size(11.0),
        );
        if let Some(e) = &view.norm_error {
            ui.colored_label(Color32::LIGHT_RED, e);
        }
        return;
    }

    ui.checkbox(
        &mut view.norm_settings.beam_fluctuation,
        "Use beam fluctuation correction (ROI)",
    )
    .on_hover_text(
        "each sample and OB image is divided by the mean of its own ROI before the \
         division — corrects the beam intensity varying between projections",
    );

    if view.norm_settings.needs_roi() {
        ui.horizontal(|ui| {
            let label = if view.norm_settings.roi.is_some() {
                "🎯 Reselect the ROI…"
            } else {
                "🎯 Select the ROI…"
            };
            if ui.button(label).clicked() {
                logger::log("opening the ROI selector on the integrated sample image");
                view.norm_error = None;
                view.roi_job = Some(RoiJob::start(std::sync::Arc::clone(&view.stack)));
            }
            match &view.norm_settings.roi {
                Some(roi) => {
                    ui.label(
                        RichText::new(format!(
                            "ROI: x={}, y={}, {}x{}",
                            roi.x, roi.y, roi.width, roi.height
                        ))
                        .strong(),
                    );
                }
                None => {
                    ui.label(
                        RichText::new("select a region away from the sample (clear at every angle)")
                            .weak(),
                    );
                }
            }
        });
    }

    let missing_roi = view.norm_settings.needs_roi() && view.norm_settings.roi.is_none();
    let ready = !missing_roi && view.norm_job.is_none() && view.roi_job.is_none();
    if ui
        .add_enabled(ready, egui::Button::new("▶ Normalize (NeuNorm)"))
        .clicked()
    {
        logger::log(format!(
            "normalizing {} projections with {} ob image(s): {}",
            view.stack.sample.len(),
            view.stack.ob.len(),
            view.norm_settings.describe()
        ));
        view.norm_error = None;
        view.norm_job = Some(NormJob::start(
            std::sync::Arc::clone(&view.stack),
            view.norm_settings.clone(),
        ));
    }
    if missing_roi {
        ui.label(RichText::new("the checked corrections need a ROI first").weak());
    }
    if let Some(e) = &view.norm_error {
        ui.colored_label(Color32::LIGHT_RED, e);
    }
}

/// Rotate the stack in 90° steps so the scan's rotation axis is vertical —
/// with a live preview (any projection, via a frame slider) of the selected
/// rotation before applying it to the whole stack.
fn rotation_section_ui(ui: &mut egui::Ui, view: &mut StackView) {
    let ctx = ui.ctx().clone();
    if let Some(job) = &mut view.rotate_job {
        if let Some(rotated) = job.poll() {
            logger::log(format!(
                "rotation applied to sample and open beams: {}°",
                view.rotation_quarters * 90
            ));
            view.stack = std::sync::Arc::new(rotated);
            view.rotation_applied = view.rotation_quarters;
            view.rotate_job = None;
        } else {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("rotating the stack…");
            });
            ctx.request_repaint_after(Duration::from_millis(300));
            return;
        }
    }

    let Some(baseline) = view.unrotated.clone() else {
        ui.label(RichText::new("normalize first").weak());
        return;
    };
    let n = baseline.sample.len();
    if n == 0 {
        return;
    }

    ui.horizontal(|ui| {
        ui.label(RichText::new("Rotation:").strong());
        for quarters in 0..=3 {
            ui.radio_value(
                &mut view.rotation_quarters,
                quarters,
                format!("{}°", quarters * 90),
            )
            .on_hover_text("clockwise — the rotation axis must end up vertical");
        }
        ui.label(
            RichText::new(if view.rotation_applied == view.rotation_quarters {
                format!("applied: {}°", view.rotation_applied * 90)
            } else {
                format!(
                    "applied: {}° — previewing {}°",
                    view.rotation_applied * 90,
                    view.rotation_quarters * 90
                )
            })
            .weak(),
        );
    });

    // Frame slider through ALL projections, so the effect of the rotation is
    // visible on any of them.
    view.rot_frame = view.rot_frame.min(n - 1);
    ui.horizontal(|ui| {
        ui.add(egui::Slider::new(&mut view.rot_frame, 0..=n - 1).text("image"));
        let p = &baseline.sample[view.rot_frame];
        ui.label(
            RichText::new(match p.angle_deg {
                Some(a) => format!("{} — {a:.3}°", p.name),
                None => p.name.clone(),
            })
            .weak()
            .size(11.0),
        );
    });

    // Preview: the selected frame, downsampled then rotated.
    let key = (
        std::sync::Arc::as_ptr(&baseline) as usize,
        view.rot_frame,
        view.rotation_quarters,
    );
    if view.rot_tex.as_ref().map(|(k, _)| *k) != Some(key) {
        let p = &baseline.sample[view.rot_frame];
        let stride = (p.width.max(p.height) / 512).max(1);
        let (sw, sh) = (p.width.div_ceil(stride), p.height.div_ceil(stride));
        let mut small = Vec::with_capacity(sw * sh);
        for y in (0..p.height).step_by(stride) {
            for x in (0..p.width).step_by(stride) {
                small.push(p.mean[y * p.width + x]);
            }
        }
        let (rw, rh, rotated) = rotate::rotate_quarter(&small, sw, sh, view.rotation_quarters);
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for v in &rotated {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        let span = (hi - lo).max(1e-6);
        let pixels: Vec<Color32> = rotated
            .iter()
            .map(|v| {
                let g = (((v - lo) / span) * 255.0) as u8;
                Color32::from_gray(g)
            })
            .collect();
        let image = egui::ColorImage {
            size: [rw, rh],
            source_size: egui::vec2(rw as f32, rh as f32),
            pixels,
        };
        let tex = ctx.load_texture("rotation_preview", image, egui::TextureOptions::LINEAR);
        view.rot_tex = Some((key, tex));
    }
    if let Some((_, tex)) = &view.rot_tex {
        let size = tex.size_vec2();
        let scale = (420.0 / size.x.max(size.y)).min(2.0);
        ui.add(egui::Image::from_texture(tex).fit_to_exact_size(size * scale));
    }

    let ready = view.rotation_quarters != view.rotation_applied && view.rotate_job.is_none();
    if ui
        .add_enabled(ready, egui::Button::new("▶ Apply the rotation to the stack"))
        .clicked()
    {
        logger::log(format!(
            "rotating the stack (sample + ob) by {}° clockwise…",
            view.rotation_quarters * 90
        ));
        view.rotate_job = Some(RotateJob::start(baseline, view.rotation_quarters));
    }
    if view.rotation_applied == view.rotation_quarters {
        ui.label(
            RichText::new("the preview matches what the stack currently is")
                .weak()
                .size(11.0),
        );
    }
}

/// Tilt correction (neutompy find_COR port): estimate the rotation-axis
/// tilt and offset from the 0° and 180° projections, then rotate + roll
/// every projection to straighten it.
fn tilt_section_ui(ui: &mut egui::Ui, view: &mut StackView) {
    let ctx = ui.ctx().clone();
    // Fold finished background work into the view.
    if let Some(job) = &mut view.tilt_calc {
        match job.poll() {
            Some(Ok(result)) => {
                logger::log(format!(
                    "tilt estimated: {:.4} deg, axis shift {} px ({} rows fitted)",
                    result.tilt_deg, result.shift_px, result.rows_used
                ));
                view.tilt_result = Some(result);
                view.tilt_preview_corrected = true;
                view.tilt_error = None;
                view.tilt_calc = None;
            }
            Some(Err(e)) => {
                logger::error(format!("tilt estimation failed: {e}"));
                view.tilt_error = Some(e);
                view.tilt_calc = None;
            }
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("estimating the tilt…");
                });
                ctx.request_repaint_after(Duration::from_millis(300));
                return;
            }
        }
    }
    if let Some(job) = &mut view.tilt_apply {
        if let Some(corrected) = job.poll() {
            let applied = view.tilt_result.take();
            logger::log(format!(
                "tilt correction applied: {:.4} deg, {} px roll",
                applied.map(|r| r.tilt_deg).unwrap_or(0.0),
                applied.map(|r| r.shift_px).unwrap_or(0)
            ));
            view.stack = std::sync::Arc::new(corrected);
            view.tilt_applied = applied;
            view.tilt_apply = None;
            view.clear_log();
        } else {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("applying the tilt correction to the stack…");
            });
            ctx.request_repaint_after(Duration::from_millis(300));
            return;
        }
    }

    let stack = std::sync::Arc::clone(&view.stack);
    let Some(first) = stack.sample.first() else {
        return;
    };
    let h = first.height;
    // The projections closest to 0° and 180°.
    let nearest = |target: f64| -> Option<usize> {
        stack
            .sample
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.angle_deg.map(|a| (i, (a - target).abs())))
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
    };
    let (Some(i0), Some(i180)) = (nearest(0.0), nearest(180.0)) else {
        ui.colored_label(
            Color32::LIGHT_RED,
            "the projections carry no angles — the tilt needs the 0° and 180° images",
        );
        return;
    };
    ui.label(
        RichText::new(format!(
            "0°: {} — 180°: {}",
            stack.sample[i0].name, stack.sample[i180].name
        ))
        .weak()
        .size(11.0),
    );

    let (mut y_top, mut y_bottom) = view
        .tilt_range
        .unwrap_or((h / 10, h.saturating_sub(1 + h / 10)));
    ui.horizontal(|ui| {
        ui.label("slice range:");
        ui.add(egui::DragValue::new(&mut y_top).range(0..=h.saturating_sub(2)));
        ui.label("to");
        ui.add(egui::DragValue::new(&mut y_bottom).range(1..=h - 1));
        ui.label(
            RichText::new("(rows where the sample is visible)")
                .weak()
                .size(11.0),
        );
    });
    y_bottom = y_bottom.clamp(y_top + 1, h - 1);
    view.tilt_range = Some((y_top, y_bottom));

    // Preview: any projection (frame slider), with the slice range shaded
    // and — once estimated — either the axis overlay on the raw image or the
    // image with the correction applied.
    let w = first.width;
    let n = stack.sample.len();
    view.tilt_frame = view.tilt_frame.min(n - 1);
    ui.horizontal(|ui| {
        ui.add(egui::Slider::new(&mut view.tilt_frame, 0..=n - 1).text("image"));
        let p = &stack.sample[view.tilt_frame];
        ui.label(
            RichText::new(match p.angle_deg {
                Some(a) => format!("{} — {a:.3}°", p.name),
                None => p.name.clone(),
            })
            .weak()
            .size(11.0),
        );
        ui.add_enabled(
            view.tilt_result.is_some(),
            egui::Checkbox::new(
                &mut view.tilt_preview_corrected,
                "with the correction applied",
            ),
        );
    });
    let corrected = view.tilt_preview_corrected && view.tilt_result.is_some();
    let result_bits = view
        .tilt_result
        .map(|r| r.tilt_deg.to_bits() ^ (r.shift_px as u64).rotate_left(17))
        .unwrap_or(0);
    let tex_key = (
        std::sync::Arc::as_ptr(&stack) as usize,
        view.tilt_frame,
        corrected,
        if corrected { result_bits } else { 0 },
    );
    if view.tilt_tex.as_ref().map(|(k, _)| *k) != Some(tex_key) {
        let p = &stack.sample[view.tilt_frame];
        let stride = (p.width.max(p.height) / 512).max(1);
        let (sw, sh) = (p.width.div_ceil(stride), p.height.div_ceil(stride));
        let mut small = Vec::with_capacity(sw * sh);
        for y in (0..p.height).step_by(stride) {
            for x in (0..p.width).step_by(stride) {
                small.push(p.mean[y * p.width + x]);
            }
        }
        if corrected && let Some(result) = &view.tilt_result {
            // Same correction, scaled to the downsampled preview.
            let shift_small = (result.shift_px as f64 / stride as f64).round() as i64;
            small = crate::tilt::rotate_roll(&small, sw, sh, result.tilt_deg, shift_small);
        }
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for v in &small {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        let span = (hi - lo).max(1e-6);
        let pixels: Vec<Color32> = small
            .iter()
            .map(|v| Color32::from_gray((((v - lo) / span) * 255.0) as u8))
            .collect();
        let image = egui::ColorImage {
            size: [sw, sh],
            source_size: egui::vec2(sw as f32, sh as f32),
            pixels,
        };
        let tex = ctx.load_texture("tilt_preview", image, egui::TextureOptions::LINEAR);
        view.tilt_tex = Some((tex_key, tex));
    }
    if let Some((_, tex)) = &view.tilt_tex {
        let size = tex.size_vec2();
        let scale = (440.0 / size.x.max(size.y)).min(2.0);
        let response = ui.add(egui::Image::from_texture(tex).fit_to_exact_size(size * scale));
        let rect = response.rect;
        let painter = ui.painter_at(rect);
        let y_of = |row: f64| rect.top() + (row / h as f64) as f32 * rect.height();
        let x_of = |col: f64| rect.left() + (col / w as f64) as f32 * rect.width();
        // Slice range: shaded band and its two edge lines.
        let band = egui::Rect::from_min_max(
            egui::pos2(rect.left(), y_of(y_top as f64)),
            egui::pos2(rect.right(), y_of(y_bottom as f64)),
        );
        painter.rect_filled(band, 0.0, Color32::from_rgba_unmultiplied(100, 170, 255, 28));
        for row in [y_top, y_bottom] {
            painter.line_segment(
                [
                    egui::pos2(rect.left(), y_of(row as f64)),
                    egui::pos2(rect.right(), y_of(row as f64)),
                ],
                egui::Stroke::new(1.5, Color32::from_rgb(100, 170, 255)),
            );
        }
        // Vertical reference through the detector center.
        let center_x = x_of((w as f64 - 1.0) / 2.0);
        painter.line_segment(
            [
                egui::pos2(center_x, rect.top()),
                egui::pos2(center_x, rect.bottom()),
            ],
            egui::Stroke::new(1.0, Color32::from_gray(120)),
        );
        // On the raw image: the estimated rotation axis (what the correction
        // will straighten). On the corrected image it coincides with the
        // vertical reference, so it is not drawn.
        if !corrected
            && let Some(result) = &view.tilt_result
        {
            painter.line_segment(
                [
                    egui::pos2(x_of(result.axis_column(0.0, w)), y_of(0.0)),
                    egui::pos2(x_of(result.axis_column(h as f64 - 1.0, w)), y_of(h as f64 - 1.0)),
                ],
                egui::Stroke::new(2.0, Color32::from_rgb(255, 160, 70)),
            );
        }
    }
    ui.label(
        RichText::new(if corrected {
            "correction applied in the preview — the rotation axis should now match the \
             gray vertical reference"
        } else if view.tilt_result.is_some() {
            "blue: slice range — gray: vertical reference — orange: estimated rotation axis \
             (the correction makes it match the gray line)"
        } else {
            "blue: slice range used by the fit — gray: vertical reference; calculate to see \
             the estimated axis"
        })
        .weak()
        .size(11.0),
    );

    let busy = view.tilt_calc.is_some() || view.tilt_apply.is_some();
    ui.horizontal(|ui| {
        if ui
            .add_enabled(!busy, egui::Button::new("🧮 Calculate the tilt"))
            .clicked()
        {
            logger::log(format!(
                "estimating the tilt from {} (0°) and {} (180°), rows {y_top}..{y_bottom}",
                stack.sample[i0].name, stack.sample[i180].name
            ));
            view.tilt_error = None;
            view.tilt_calc = Some(TiltCalcJob::start(
                std::sync::Arc::clone(&stack),
                i0,
                i180,
                y_top,
                y_bottom,
            ));
        }
        if let Some(result) = &view.tilt_result {
            ui.label(
                RichText::new(format!(
                    "tilt: {:.4}° — axis shift: {} px ({} rows fitted)",
                    result.tilt_deg, result.shift_px, result.rows_used
                ))
                .strong(),
            );
        }
    });

    if view.tilt_result.is_some()
        && ui
            .add_enabled(!busy, egui::Button::new("▶ Apply the correction to the stack"))
            .on_hover_text("rotates every projection by the tilt and rolls it by the axis shift")
            .clicked()
        && let Some(result) = view.tilt_result
    {
        logger::log(format!(
            "applying the tilt correction ({:.4} deg, {} px) to {} projections…",
            result.tilt_deg,
            result.shift_px,
            stack.sample.len()
        ));
        view.tilt_apply = Some(TiltApplyJob::start(stack, result));
    }
    if let Some(applied) = &view.tilt_applied {
        ui.label(
            RichText::new(format!(
                "correction applied ✔ ({:.4}°, {} px) — recalculate to verify: it should now \
                 find ≈ 0°",
                applied.tilt_deg, applied.shift_px
            ))
            .color(Color32::from_rgb(120, 200, 120)),
        );
    }
    if let Some(e) = &view.tilt_error {
        ui.colored_label(Color32::LIGHT_RED, e);
    }
}

/// Log conversion (mandatory before saving): the notebook's
/// `log_conversion_and_cleaning` — transmission to attenuation via -log,
/// with tomopy-style outlier removal and negatives set to 0 on the way.
fn log_section_ui(ui: &mut egui::Ui, view: &mut StackView) {
    let ctx = ui.ctx().clone();
    if let Some(job) = &mut view.log_job {
        if let Some((converted, stats)) = job.poll() {
            logger::log(format!(
                "log conversion done: {} outliers replaced, {} negatives zeroed",
                stats.outliers_replaced, stats.negatives_zeroed
            ));
            if view.unlogged.is_none() {
                view.unlogged = Some(std::sync::Arc::clone(&view.stack));
            }
            view.stack = std::sync::Arc::new(converted);
            view.log_converted = true;
            view.log_stats = Some(stats);
            view.log_job = None;
        } else {
            let done = job.done();
            let frac = (done as f32 / job.total.max(1) as f32).min(1.0);
            ui.add(egui::ProgressBar::new(frac).text(format!("{done}/{} images", job.total)));
            ctx.request_repaint_after(Duration::from_millis(300));
            return;
        }
    }

    if view.log_converted {
        let stats = view.log_stats;
        ui.label(
            RichText::new(format!(
                "converted to attenuation ✔ — -log(T){}",
                stats
                    .map(|s| format!(
                        " ({} outliers replaced, {} negatives zeroed)",
                        s.outliers_replaced, s.negatives_zeroed
                    ))
                    .unwrap_or_default()
            ))
            .color(Color32::from_rgb(120, 200, 120)),
        );
        // Visualize the attenuation data in the sibling TIFF viewer.
        if let Some(job) = &mut view.log_visualize_job {
            match job.poll() {
                Some(Ok(())) => {
                    logger::log("attenuation-data viewer closed");
                    view.log_visualize_job = None;
                }
                Some(Err(e)) => {
                    logger::error(format!("visualizing the attenuation data failed: {e}"));
                    view.log_visualize_job = None;
                }
                None => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("TIFF viewer is open on the attenuation images");
                    });
                    ctx.request_repaint_after(Duration::from_millis(300));
                }
            }
        }
        ui.horizontal(|ui| {
            if view.log_visualize_job.is_none()
                && ui
                    .button("👁 Visualize the attenuation data")
                    .on_hover_text("opens the stack in rust_tiff_viewer, on the single-image view")
                    .clicked()
            {
                logger::log("opening the TIFF viewer on the attenuation stack (single-image view)");
                view.log_visualize_job =
                    Some(VisualizeJob::start(std::sync::Arc::clone(&view.stack)));
            }
            if view.unlogged.is_some() && ui.button("↩ Back to transmission").clicked() {
                logger::log("log conversion undone: back to the transmission stack");
                view.stack = view.unlogged.take().expect("checked above");
                view.log_converted = false;
                view.log_stats = None;
            }
        });
        return;
    }

    ui.label(
        "converts the transmission data to attenuation (-log), the form the \
         reconstruction algorithms need, and cleans it on the way (tomopy outlier \
         removal, diff 0.2, and negatives set to 0)",
    );
    ui.label(
        RichText::new("this step is mandatory before saving the checkpoint — run every \
                       other correction first, this is the last one")
            .weak()
            .size(11.0),
    );
    if ui
        .add_enabled(
            view.log_job.is_none(),
            egui::Button::new("▶ Convert to attenuation (-log)"),
        )
        .clicked()
    {
        logger::log(format!(
            "converting {} projections to attenuation (-log)…",
            view.stack.sample.len()
        ));
        view.log_job = Some(LogConvertJob::start(std::sync::Arc::clone(&view.stack)));
    }
}

/// "Remove stripes": the tomopy stripe-removal algorithms of the Python
/// notebook — pick and order the algorithms, test them on a band of rows
/// (before/after sinograms), then apply to the whole stack.
fn stripes_section_ui(ui: &mut egui::Ui, view: &mut StackView) {
    let ctx = ui.ctx().clone();
    // Fold finished background work into the view.
    if let Some(job) = &mut view.stripe_test_job {
        match job.poll() {
            Some(Ok((before, after, n, band_h, w))) => {
                let y0 = view.stripe_range.map(|(y0, _)| y0).unwrap_or(0);
                logger::log(format!(
                    "stripe removal test done on rows {y0}..{} ({})",
                    y0 + band_h - 1,
                    stripes::describe(&view.stripe_algos)
                ));
                view.stripe_test = Some((n, band_h, w, y0, before, after));
                view.stripe_test_tex = None;
                view.stripe_error = None;
                view.stripe_test_job = None;
            }
            Some(Err(e)) => {
                logger::error(format!("stripe removal test failed: {e}"));
                view.stripe_error = Some(e);
                view.stripe_test_job = None;
            }
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("tomopy is running on the test band…");
                });
                ctx.request_repaint_after(Duration::from_millis(300));
                return;
            }
        }
    }
    if let Some(job) = &mut view.stripe_apply_job {
        match job.poll() {
            Some(Ok(cleaned)) => {
                let desc = stripes::describe(&view.stripe_algos);
                logger::log(format!("stripes removed from the whole stack: {desc}"));
                if view.unstriped.is_none() {
                    view.unstriped = Some(std::sync::Arc::clone(&view.stack));
                }
                view.stack = std::sync::Arc::new(cleaned);
                view.stripes_applied = Some(desc);
                view.stripe_error = None;
                view.stripe_apply_job = None;
                view.clear_log();
            }
            Some(Err(e)) => {
                logger::error(format!("stripe removal failed: {e}"));
                view.stripe_error = Some(e);
                view.stripe_apply_job = None;
            }
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("tomopy is running on the whole stack…");
                });
                ctx.request_repaint_after(Duration::from_millis(300));
                return;
            }
        }
    }

    let Some(first) = view.stack.sample.first() else {
        return;
    };
    let h = first.height;

    // Algorithm list (applied in this order), with their parameters.
    ui.label(
        RichText::new("tomopy algorithms — the checked ones are applied in this order")
            .weak()
            .size(12.0),
    );
    for algo in &mut view.stripe_algos {
        ui.horizontal(|ui| {
            ui.checkbox(&mut algo.enabled, algo.name)
                .on_hover_text(algo.help);
        });
        if algo.enabled {
            ui.indent(algo.name, |ui| {
                ui.horizontal_wrapped(|ui| {
                    for param in &mut algo.params {
                        ui.label(format!("{}:", param.name));
                        match &mut param.value {
                            stripes::ParamValue::Int(v) => {
                                let widget = egui::DragValue::new(v).range(0..=999);
                                let response = ui.add(widget);
                                if param.zero_means_auto {
                                    response.on_hover_text("0 = auto");
                                }
                            }
                            stripes::ParamValue::Float(v) => {
                                ui.add(egui::DragValue::new(v).speed(0.1).range(0.0..=999.0));
                            }
                            stripes::ParamValue::Bool(v) => {
                                ui.checkbox(v, "");
                            }
                        }
                    }
                });
            });
        }
    }
    let any_enabled = view.stripe_algos.iter().any(|a| a.enabled);
    if !any_enabled {
        ui.label(RichText::new("select at least one algorithm").weak());
    }

    // Test band + buttons.
    let (mut y0, mut y1) = view
        .stripe_range
        .unwrap_or((h / 3, (2 * h / 3).min(h - 1)));
    ui.horizontal(|ui| {
        ui.label("test on rows:");
        ui.add(egui::DragValue::new(&mut y0).range(0..=h.saturating_sub(2)));
        ui.label("to");
        ui.add(egui::DragValue::new(&mut y1).range(1..=h - 1));
        let busy = view.stripe_test_job.is_some() || view.stripe_apply_job.is_some();
        if ui
            .add_enabled(any_enabled && !busy, egui::Button::new("🧪 Test on this band"))
            .clicked()
        {
            logger::log(format!(
                "testing stripe removal on rows {y0}..{y1}: {}",
                stripes::describe(&view.stripe_algos)
            ));
            view.stripe_error = None;
            view.stripe_test = None;
            view.stripe_test_tex = None;
            view.stripe_test_job = Some(StripeTestJob::start(
                std::sync::Arc::clone(&view.stack),
                y0.min(h - 2),
                y1.clamp(y0 + 1, h - 1),
                view.stripe_algos.clone(),
            ));
        }
        if ui
            .add_enabled(
                any_enabled && !busy,
                egui::Button::new("▶ Apply to the whole stack"),
            )
            .clicked()
        {
            let input = view
                .unstriped
                .clone()
                .unwrap_or_else(|| std::sync::Arc::clone(&view.stack));
            logger::log(format!(
                "removing stripes from the whole stack: {}",
                stripes::describe(&view.stripe_algos)
            ));
            view.stripe_error = None;
            view.stripe_apply_job = Some(StripeApplyJob::start(input, view.stripe_algos.clone()));
        }
    });
    y1 = y1.clamp(y0 + 1, h - 1);
    view.stripe_range = Some((y0, y1));

    // Test result: before/after sinograms of a row inside the band.
    if let Some((n, band_h, w, band_y0, before, after)) = &view.stripe_test {
        let (n, band_h, w, band_y0) = (*n, *band_h, *w, *band_y0);
        view.stripe_test_row = view.stripe_test_row.clamp(band_y0, band_y0 + band_h - 1);
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut view.stripe_test_row, band_y0..=band_y0 + band_h - 1)
                    .text("sinogram row"),
            );
        });
        let key = (view.stripe_test_row, view.stripe_algos.len());
        if view.stripe_test_tex.as_ref().map(|(k, ..)| *k) != Some(key) {
            let row = view.stripe_test_row - band_y0;
            let stride = (w / 1024).max(1);
            let sino_w = w.div_ceil(stride);
            let build = |volume: &[f32]| -> Vec<f32> {
                let mut values = Vec::with_capacity(n * sino_w);
                for i in 0..n {
                    let line = &volume[(i * band_h + row) * w..(i * band_h + row) * w + w];
                    for x in (0..w).step_by(stride) {
                        values.push(line[x]);
                    }
                }
                values
            };
            let sino_before = build(before);
            let sino_after = build(after);
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for v in sino_before.iter().chain(&sino_after) {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
            let span = (hi - lo).max(1e-6);
            let to_tex = |values: &[f32], name: &str| {
                let pixels: Vec<Color32> = values
                    .iter()
                    .map(|v| Color32::from_gray((((v - lo) / span) * 255.0) as u8))
                    .collect();
                ctx.load_texture(
                    name.to_owned(),
                    egui::ColorImage {
                        size: [sino_w, n],
                        source_size: egui::vec2(sino_w as f32, n as f32),
                        pixels,
                    },
                    egui::TextureOptions::NEAREST,
                )
            };
            view.stripe_test_tex = Some((
                key,
                to_tex(&sino_before, "stripe_before"),
                to_tex(&sino_after, "stripe_after"),
            ));
        }
        if let Some((_, before_tex, after_tex)) = &view.stripe_test_tex {
            ui.columns(2, |cols| {
                for (col, tex, title) in [
                    (0usize, before_tex, "before"),
                    (1, after_tex, "after"),
                ] {
                    let ui = &mut cols[col];
                    ui.label(RichText::new(title).strong().size(13.0));
                    let size = tex.size_vec2();
                    let width = (ui.available_width() - 12.0).clamp(200.0, 500.0);
                    let height = (size.y * 3.0).clamp(240.0, 420.0);
                    ui.add(
                        egui::Image::from_texture(tex)
                            .fit_to_exact_size(egui::vec2(width, height)),
                    );
                }
            });
            ui.label(
                RichText::new(
                    "sinograms of the selected row — vertical lines are the stripes the \
                     algorithms should remove",
                )
                .weak()
                .size(11.0),
            );
        }
    }

    if let Some(desc) = &view.stripes_applied {
        ui.label(
            RichText::new(format!("stripes removed ✔ — {desc}"))
                .color(Color32::from_rgb(120, 200, 120)),
        );
        ui.label(
            RichText::new("re-applying runs the current selection on the pre-removal stack")
                .weak()
                .size(11.0),
        );
    }
    if let Some(e) = &view.stripe_error {
        ui.colored_label(Color32::LIGHT_RED, e);
    }
}

/// Center of rotation — optional: without running it, the horizontal center
/// of the detector is used; running it compares the 0° and 180° projections
/// on a band around the chosen slice (the reconstruction will use the
/// resulting value).
fn cor_section_ui(ui: &mut egui::Ui, view: &mut StackView) {
    let ctx = ui.ctx().clone();
    if let Some(job) = &mut view.cor_job {
        match job.poll() {
            Some(Ok(cor)) => {
                logger::log(format!("center of rotation calculated: {cor:.2} px"));
                view.cor_result = Some(cor);
                view.cor_error = None;
                view.cor_job = None;
            }
            Some(Err(e)) => {
                logger::error(format!("center of rotation failed: {e}"));
                view.cor_error = Some(e);
                view.cor_job = None;
            }
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("comparing the 0° and 180° projections…");
                });
                ctx.request_repaint_after(Duration::from_millis(300));
                return;
            }
        }
    }

    let stack = std::sync::Arc::clone(&view.stack);
    let Some(first) = stack.sample.first() else {
        return;
    };
    let (w, h) = (first.width, first.height);
    let default_center = (w as f64 - 1.0) / 2.0;
    let cor = view.cor_result.unwrap_or(default_center);
    ui.label(match view.cor_result {
        Some(value) => RichText::new(format!(
            "center of rotation: {value:.2} px (calculated — {:+.2} px from the center)",
            value - default_center
        ))
        .strong()
        .color(Color32::from_rgb(120, 200, 120)),
        None => RichText::new(format!(
            "center of rotation: {default_center:.1} px (horizontal center — default until calculated)"
        ))
        .strong(),
    });

    // The projections closest to 0° and 180°.
    let nearest = |target: f64| -> Option<usize> {
        stack
            .sample
            .iter()
            .enumerate()
            .filter_map(|(i, p)| p.angle_deg.map(|a| (i, (a - target).abs())))
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
    };
    let indices = (nearest(0.0), nearest(180.0));
    let mut slice_row = view.cor_slice.unwrap_or(h / 2);
    if let (Some(i0), Some(i180)) = indices {
        ui.label(
            RichText::new(
                "the slice selected below is the one used to estimate the center of rotation",
            )
            .weak()
            .size(12.0),
        );
        ui.horizontal(|ui| {
            ui.add(
                egui::Slider::new(&mut slice_row, 0..=h - 1)
                    .text("slice used for the estimation"),
            );
            let busy = view.cor_job.is_some();
            if ui
                .add_enabled(!busy, egui::Button::new("🧮 Calculate the center of rotation"))
                .clicked()
            {
                logger::log(format!(
                    "calculating the center of rotation from {} (0°) and {} (180°), slice {slice_row}",
                    stack.sample[i0].name, stack.sample[i180].name
                ));
                view.cor_error = None;
                view.cor_job = Some(CorJob::start(
                    std::sync::Arc::clone(&stack),
                    i0,
                    i180,
                    slice_row,
                ));
            }
            if view.cor_result.is_some() && ui.button("use the horizontal center instead").clicked()
            {
                logger::log("center of rotation reset to the horizontal center");
                view.cor_result = None;
            }
        });
        view.cor_slice = Some(slice_row.min(h - 1));
    } else {
        ui.colored_label(
            Color32::LIGHT_RED,
            "the projections carry no angles — only the horizontal center can be used",
        );
    }

    // Preview: a single projection (frame slider, starting on the 0° one),
    // or the 0° + mirrored-180° overlay that verifies the center directly.
    let n = stack.sample.len();
    let overlay_possible = indices.0.is_some() && indices.1.is_some();
    ui.horizontal(|ui| {
        if ui.selectable_label(!view.cor_overlay, "single image").clicked() {
            view.cor_overlay = false;
        }
        if ui
            .add_enabled(
                overlay_possible,
                egui::Button::selectable(view.cor_overlay, "0° + 180° overlay"),
            )
            .on_hover_text(
                "the 180° projection is mirrored about the center of rotation and blended \
                 with the 0° one: features coincide when the center is correct",
            )
            .clicked()
        {
            view.cor_overlay = true;
        }
    });
    let overlay = view.cor_overlay && overlay_possible;
    let mut preview_index = view.cor_frame.unwrap_or_else(|| indices.0.unwrap_or(0)).min(n - 1);
    if !overlay {
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut preview_index, 0..=n - 1).text("image"));
            let p = &stack.sample[preview_index];
            ui.label(
                RichText::new(match p.angle_deg {
                    Some(a) => format!("{} — {a:.3}°", p.name),
                    None => p.name.clone(),
                })
                .weak()
                .size(11.0),
            );
        });
        view.cor_frame = Some(preview_index);
    }
    let tex_key = (
        std::sync::Arc::as_ptr(&stack) as usize,
        if overlay { usize::MAX } else { preview_index },
        if overlay { cor.to_bits() } else { 0 },
    );
    if view.cor_tex.as_ref().map(|(k, _)| *k) != Some(tex_key) {
        let downsample = |p: &crate::combine::Projection, stride: usize| -> Vec<f32> {
            let mut small =
                Vec::with_capacity(p.width.div_ceil(stride) * p.height.div_ceil(stride));
            for y in (0..p.height).step_by(stride) {
                for x in (0..p.width).step_by(stride) {
                    small.push(p.mean[y * p.width + x]);
                }
            }
            small
        };
        let stride = (w.max(h) / 512).max(1);
        let (sw, sh) = (w.div_ceil(stride), h.div_ceil(stride));
        let image = if overlay {
            let (i0, i180) = (indices.0.expect("overlay"), indices.1.expect("overlay"));
            let s0 = downsample(&stack.sample[i0], stride);
            let s180 = downsample(&stack.sample[i180], stride);
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for v in s0.iter().chain(&s180) {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
            let span = (hi - lo).max(1e-6);
            let cor_small = cor / stride as f64;
            let mut pixels = Vec::with_capacity(sw * sh);
            for y in 0..sh {
                for x in 0..sw {
                    let v0 = (s0[y * sw + x] - lo) / span;
                    // 180° mirrored about the center of rotation, with
                    // linear interpolation.
                    let sx = 2.0 * cor_small - x as f64;
                    let x0 = sx.floor();
                    let f = (sx - x0) as f32;
                    let clamp = |v: f64| (v.max(0.0) as usize).min(sw - 1);
                    let a = s180[y * sw + clamp(x0)];
                    let b = s180[y * sw + clamp(x0 + 1.0)];
                    let v180 = ((a * (1.0 - f) + b * f) - lo) / span;
                    pixels.push(Color32::from_rgb(
                        (v0.clamp(0.0, 1.0) * 255.0) as u8,
                        (v180.clamp(0.0, 1.0) * 255.0) as u8,
                        (v180.clamp(0.0, 1.0) * 255.0) as u8,
                    ));
                }
            }
            egui::ColorImage {
                size: [sw, sh],
                source_size: egui::vec2(sw as f32, sh as f32),
                pixels,
            }
        } else {
            let small = downsample(&stack.sample[preview_index], stride);
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for v in &small {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
            let span = (hi - lo).max(1e-6);
            let pixels: Vec<Color32> = small
                .iter()
                .map(|v| Color32::from_gray((((v - lo) / span) * 255.0) as u8))
                .collect();
            egui::ColorImage {
                size: [sw, sh],
                source_size: egui::vec2(sw as f32, sh as f32),
                pixels,
            }
        };
        let tex = ctx.load_texture("cor_preview", image, egui::TextureOptions::LINEAR);
        view.cor_tex = Some((tex_key, tex));
    }
    if let Some((_, tex)) = &view.cor_tex {
        let size = tex.size_vec2();
        let scale = (440.0 / size.x.max(size.y)).min(2.0);
        let response = ui.add(egui::Image::from_texture(tex).fit_to_exact_size(size * scale));
        let rect = response.rect;
        let painter = ui.painter_at(rect);
        // The center of rotation as a vertical line.
        let x = rect.left() + (cor / w as f64) as f32 * rect.width();
        let color = if view.cor_result.is_some() {
            Color32::from_rgb(120, 200, 120)
        } else {
            Color32::from_rgb(100, 170, 255)
        };
        painter.line_segment(
            [egui::pos2(x, rect.top()), egui::pos2(x, rect.bottom())],
            egui::Stroke::new(2.0, color),
        );
        // The slice used by the calculation, as a dashed horizontal guide.
        if indices.0.is_some() {
            let y = rect.top() + (slice_row as f64 / h as f64) as f32 * rect.height();
            let dash = 8.0;
            let mut x0 = rect.left();
            while x0 < rect.right() {
                painter.line_segment(
                    [
                        egui::pos2(x0, y),
                        egui::pos2((x0 + dash * 0.6).min(rect.right()), y),
                    ],
                    egui::Stroke::new(1.0, Color32::from_gray(150)),
                );
                x0 += dash;
            }
        }
    }
    ui.label(
        RichText::new(if overlay {
            "red: 0° — cyan: 180° mirrored about the center of rotation — features turn \
             gray when the center is correct, red/cyan ghosting means it is off"
        } else if view.cor_result.is_some() {
            "green: calculated center of rotation — dashes: the slice used to estimate it"
        } else {
            "blue: horizontal center (used until a calculation is run) — dashes: the slice \
             the estimation would use"
        })
        .weak()
        .size(11.0),
    );
    if let Some(e) = &view.cor_error {
        ui.colored_label(Color32::LIGHT_RED, e);
    }
}

/// The sinogram of one detector row of the current (normalized, possibly
/// rotated) stack: one line per projection, in stack order (increasing
/// angle), against the detector column.
fn sinogram_section_ui(ui: &mut egui::Ui, view: &mut StackView) {
    let stack = std::sync::Arc::clone(&view.stack);
    let Some(first) = stack.sample.first() else {
        return;
    };
    let (w, h, n) = (first.width, first.height, stack.sample.len());
    if stack.sample.iter().any(|p| (p.width, p.height) != (w, h)) {
        ui.colored_label(Color32::LIGHT_RED, "projections have inconsistent sizes");
        return;
    }

    view.sino_row = view.sino_row.min(h - 1);
    ui.horizontal(|ui| {
        ui.add(egui::Slider::new(&mut view.sino_row, 0..=h - 1).text("slice (row)"));
        ui.label(
            RichText::new(format!("{n} projections × {w} columns"))
                .weak()
                .size(11.0),
        );
    });

    let key = (std::sync::Arc::as_ptr(&stack) as usize, view.sino_row);
    if view.sino_tex.as_ref().map(|(k, _)| *k) != Some(key) {
        // One sinogram line per projection at the chosen row, columns
        // stride-sampled so wide CCD frames stay a reasonable texture.
        let stride = (w / 1024).max(1);
        let sino_w = w.div_ceil(stride);
        let mut values = Vec::with_capacity(n * sino_w);
        for p in &stack.sample {
            let row = &p.mean[view.sino_row * w..(view.sino_row + 1) * w];
            for x in (0..w).step_by(stride) {
                values.push(row[x]);
            }
        }
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for v in &values {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        let span = (hi - lo).max(1e-6);
        let pixels: Vec<Color32> = values
            .iter()
            .map(|v| Color32::from_gray((((v - lo) / span) * 255.0) as u8))
            .collect();
        let image = egui::ColorImage {
            size: [sino_w, n],
            source_size: egui::vec2(sino_w as f32, n as f32),
            pixels,
        };
        // Nearest-neighbor so individual projection rows stay crisp when the
        // sinogram is stretched vertically (few projections).
        let tex = ui
            .ctx()
            .load_texture("sinogram", image, egui::TextureOptions::NEAREST);
        view.sino_tex = Some((key, tex));
    }
    if let Some((_, tex)) = &view.sino_tex {
        let size = tex.size_vec2();
        let width = (ui.available_width() - 16.0).clamp(400.0, 950.0).min(size.x * 4.0);
        // Stretch small projection counts so every row stays readable.
        let height = (size.y * 4.0).clamp(350.0, 620.0);
        ui.add(egui::Image::from_texture(tex).fit_to_exact_size(egui::vec2(width, height)));
        ui.label(
            RichText::new(
                "columns: detector x at the selected row — rows: projections, top to \
                 bottom in increasing angle",
            )
            .weak()
            .size(11.0),
        );
    }
}

/// Cropping the stack with the rust_crop_tiff tool: the sample 3-D array is
/// handed over, and the returned region is applied to the sample AND the
/// open beams.
fn crop_section_ui(ui: &mut egui::Ui, view: &mut StackView) {
    let ctx = ui.ctx().clone();
    if let Some(job) = &mut view.crop_job {
        match job.poll() {
            Some(Ok(Some((rect, cropped)))) => {
                logger::log(format!(
                    "crop applied to sample and open beams: x={}, y={}, {}x{} (was {}x{})",
                    rect.x,
                    rect.y,
                    rect.width,
                    rect.height,
                    view.original.sample.first().map(|p| p.width).unwrap_or(0),
                    view.original.sample.first().map(|p| p.height).unwrap_or(0),
                ));
                view.stack = std::sync::Arc::new(cropped);
                view.crop = Some(rect);
                view.crop_error = None;
                view.crop_job = None;
                // Cleaning and normalization applied to the previous crop
                // are void — and the ROI coordinates no longer match.
                view.uncleaned = None;
                view.clean_stats = None;
                view.sum_cache.clear();
                view.sum_jobs.clear();
                view.hist_cache.clear();
                view.normalized = false;
                view.norm_summary = None;
                view.norm_settings.roi = None;
                view.unrotated = None;
                view.rotation_quarters = 0;
                view.rotation_applied = 0;
                view.rot_tex = None;
                view.clear_tilt();
                view.clear_stripes();
                view.clear_log();
                view.clear_cor();
            }
            Some(Ok(None)) => {
                logger::log("crop tool closed without saving a region");
                view.crop_job = None;
            }
            Some(Err(e)) => {
                logger::error(format!("crop failed: {e}"));
                view.crop_error = Some(e);
                view.crop_job = None;
            }
            None => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label(
                        "crop tool is open — draw the region and press \
                         '↩ Return to main application'",
                    );
                });
                ctx.request_repaint_after(Duration::from_millis(300));
                return;
            }
        }
    }

    match &view.crop {
        Some(rect) => {
            let original = view.original.sample.first();
            ui.label(
                RichText::new(format!(
                    "crop: x={}, y={}, {}x{} (original {}x{}) — applied to sample and open beams",
                    rect.x,
                    rect.y,
                    rect.width,
                    rect.height,
                    original.map(|p| p.width).unwrap_or(0),
                    original.map(|p| p.height).unwrap_or(0),
                ))
                .strong(),
            );
        }
        None => {
            ui.label(RichText::new("no crop applied — the full images are used").weak());
        }
    }
    let label = if view.crop.is_some() {
        "✂ Adjust the crop region…"
    } else {
        "✂ Select the crop region…"
    };
    if ui.button(label).clicked() {
        logger::log(format!(
            "opening the crop tool on the sample stack ({} projections)",
            view.original.sample.len()
        ));
        view.crop_error = None;
        view.crop_job = Some(CropJob::start(
            std::sync::Arc::clone(&view.original),
            view.crop,
        ));
    }
    ui.label(
        RichText::new(
            "(an evenly sub-sampled ~10% of the projections is handed to the crop tool; \
             the region comes back and is applied to the full sample stack and the open beams)",
        )
        .weak()
        .size(11.0),
    );
    if let Some(e) = &view.crop_error {
        ui.colored_label(Color32::LIGHT_RED, e);
    }
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
    ui.add_space(4.0);
    egui::CollapsingHeader::new(RichText::new("Data to use").strong())
        .default_open(true)
        .show(ui, |ui| {
            data_to_use_ui(ui, view);
        });
    ui.add_space(4.0);
    egui::CollapsingHeader::new(RichText::new("Exclude images").strong())
        .default_open(true)
        .show(ui, |ui| {
            exclude_images_ui(ui, view);
        });
    ui.add_space(4.0);
    egui::CollapsingHeader::new(RichText::new("Save to HDF5").strong())
        .default_open(true)
        .show(ui, |ui| {
            wb_save_ui(ui, session, view);
        });
}

/// Read the final selection (one image per projection, exclusions applied,
/// increasing angle) and save it in the same HDF5 layout as the TOF side.
fn wb_save_ui(ui: &mut egui::Ui, session: &Session, view: &mut WhiteBeamView) {
    let ctx = ui.ctx().clone();
    // Fold finished background work into the view.
    if let Some(scan) = &mut view.process {
        if let Some(output) = scan.poll() {
            let angles: Vec<f64> = output.sample.iter().filter_map(|p| p.angle_deg).collect();
            logger::log(format!(
                "white beam stack read: {} projections (angles {}), {} ob images",
                output.sample.len(),
                match (angles.first(), angles.last()) {
                    (Some(a), Some(b)) => format!("{a:.3} deg -> {b:.3} deg, increasing"),
                    _ => "unknown".to_owned(),
                },
                output.ob.len()
            ));
            for e in &output.skipped {
                logger::error(format!("white beam stack: {e}"));
            }
            view.processed = Some(std::sync::Arc::new(output));
            view.process = None;
        } else {
            let done = scan.progress();
            let frac = (done as f32 / scan.total_images.max(1) as f32).min(1.0);
            ui.add(egui::ProgressBar::new(frac).text(format!(
                "{done}/{} images",
                scan.total_images
            )));
            ctx.request_repaint_after(Duration::from_millis(300));
        }
    }
    if let Some(job) = &mut view.save_job {
        match job.poll() {
            Some(Ok(msg)) => {
                logger::log(format!("saved white beam data: {msg}"));
                view.save_status = Some(Ok(msg));
                view.save_job = None;
            }
            Some(Err(e)) => {
                logger::error(format!("saving white beam data failed: {e}"));
                view.save_status = Some(Err(e));
                view.save_job = None;
            }
            None => ctx.request_repaint_after(Duration::from_millis(300)),
        }
    }

    let selection = view.final_selection();
    match &selection {
        Err(msg) => {
            ui.label(RichText::new(msg.as_str()).weak());
        }
        Ok(runs) => {
            let total: usize = view.sample.total_files();
            ui.label(format!(
                "{} of {total} projections will be saved (exclusions and coverage applied), plus {} ob images",
                runs.len(),
                view.ob.total_files()
            ));
        }
    }

    let busy = view.process.is_some();
    if ui
        .add_enabled(
            selection.is_ok() && !busy,
            egui::Button::new("▶ Read & stack the projections"),
        )
        .clicked()
        && let Ok(sample_runs) = selection
    {
        let ob_runs: Vec<RunToCombine> = view
            .ob
            .selected
            .iter()
            .flat_map(|(_, files, _)| files.iter())
            .map(|file| RunToCombine {
                name: file
                    .file_stem()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default(),
                run_number: tof::run_number(
                    &file.file_name().unwrap_or_default().to_string_lossy(),
                ),
                images: vec![file.clone()],
                angle_deg: None,
            })
            .collect();
        logger::log(format!(
            "reading white beam stack: {} projections, {} ob images",
            sample_runs.len(),
            ob_runs.len()
        ));
        view.processed = None;
        view.save_status = None;
        view.process = Some(CombineScan::start(
            sample_runs,
            ob_runs,
            ImageSelection::All,
            false,
        ));
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
                "stacked: {} projections ({dims}), angles {} — {} ob images",
                output.sample.len(),
                match (angles.first(), angles.last()) {
                    (Some(a), Some(b)) => format!("{a:.3}° to {b:.3}°"),
                    _ => "unknown".to_owned(),
                },
                output.ob.len()
            ))
            .strong(),
        );
        for e in &output.skipped {
            ui.colored_label(Color32::from_rgb(240, 180, 60), format!("skipped: {e}"));
        }
        let folder_list = |pick: &MultiFolderPick| {
            pick.selected
                .iter()
                .map(|(dir, ..)| dir.display().to_string())
                .collect::<Vec<_>>()
                .join("; ")
        };
        let mut mode = format!(
            "white beam, one image per projection; angles from {}",
            view.angle_source.label()
        );
        if view.use_percentage {
            mode.push_str(&format!("; {}% coverage selection", view.percentage));
        }
        if !view.excluded_runs.is_empty() {
            let mut sorted: Vec<u32> = view.excluded_runs.iter().copied().collect();
            sorted.sort();
            mode.push_str(&format!("; excluded runs {sorted:?}"));
        }
        if view.intensities.is_some() {
            mode.push_str(&format!("; intensity threshold {:.4e}", view.threshold));
        }
        let meta = SaveMeta {
            instrument: session.instrument.name().to_owned(),
            ipts: session.ipts.name.clone(),
            detector: view.detector.label().to_owned(),
            sample_folder: folder_list(&view.sample),
            ob_folder: folder_list(&view.ob),
            combine_mode: mode,
            selections_json: None,
            detector_offset_us: None,
        };
        let saving = view.save_job.is_some();
        let mut jump = false;
        ui.horizontal(|ui| {
            if ui
                .add_enabled(!saving, egui::Button::new("💾 Save to HDF5…"))
                .clicked()
            {
                let default_name = view
                    .sample
                    .selected
                    .first()
                    .and_then(|(dir, ..)| dir.file_name())
                    .map(|n| format!("{}_white_beam.h5", n.to_string_lossy()))
                    .unwrap_or_else(|| "ct_white_beam.h5".to_owned());
                let mut dialog = rfd::FileDialog::new()
                    .set_title("Save the white beam projections")
                    .add_filter("HDF5", &["h5", "hdf5"])
                    .set_file_name(default_name);
                let shared = session.ipts.path.join("shared");
                if shared.is_dir() {
                    dialog = dialog.set_directory(shared);
                }
                if let Some(path) = dialog.save_file() {
                    logger::log(format!("saving white beam data to {}", path.display()));
                    view.save_status = None;
                    view.save_job = Some(SaveJob::start(
                        path,
                        std::sync::Arc::clone(output),
                        meta.clone(),
                    ));
                }
            }
            jump = ui
                .button("🚀 Continue to pre-processing")
                .on_hover_text("work on the stacked projections right away — saving is optional")
                .clicked();
        });
        if jump {
            logger::log("continuing to pre-processing with the white beam stack");
            let pseudo = session
                .ipts
                .path
                .join("shared")
                .join("white_beam_combined (not saved).h5");
            view.goto_preprocess = Some(combine::stack_from_output(output, &meta, pseudo));
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

/// Manual run-number exclusion plus an intensity threshold: the integrated
/// intensity of every image against its run number, with a draggable
/// horizontal threshold under which images are dropped.
fn exclude_images_ui(ui: &mut egui::Ui, view: &mut WhiteBeamView) {
    let ctx = ui.ctx().clone();
    let files: Vec<PathBuf> = view
        .sample
        .selected
        .iter()
        .flat_map(|(_, files, _)| files.iter().cloned())
        .collect();
    if files.is_empty() {
        ui.label(RichText::new("select the sample folder(s) first").weak());
        return;
    }

    // Manual exclusion by run number.
    ui.horizontal(|ui| {
        ui.label("Run numbers to exclude:");
        let edit = ui.add(
            egui::TextEdit::singleline(&mut view.exclude_text)
                .hint_text("e.g. 1,2,5-10 — or click dots on the plot below")
                .desired_width(560.0),
        );
        if edit.changed() {
            match white_beam::parse_run_list(&view.exclude_text) {
                Ok(runs) => {
                    view.excluded_runs = runs;
                    view.exclude_error = None;
                }
                Err(e) => view.exclude_error = Some(e),
            }
        }
        if edit.lost_focus() && !view.excluded_runs.is_empty() {
            let mut sorted: Vec<u32> = view.excluded_runs.iter().copied().collect();
            sorted.sort();
            logger::log(format!("manually excluded runs: {sorted:?}"));
        }
    });
    if let Some(e) = &view.exclude_error {
        ui.colored_label(Color32::LIGHT_RED, e);
    }
    ui.add_space(6.0);

    // Integrated intensities (heavy: every image is read once). The scan
    // covers the used files plus the superseded old revisions so retakes
    // stay visible on the plot.
    let superseded: Vec<PathBuf> = view
        .sample
        .selected
        .iter()
        .flat_map(|(_, _, old)| old.iter().cloned())
        .collect();
    let key = (files.len() + superseded.len(), files.first().cloned());
    if let Some(cache) = &view.intensities
        && (cache.total, cache.first.as_ref()) != (key.0, key.1.as_ref())
    {
        view.intensities = None;
    }
    if let Some(job) = &mut view.intensity_job {
        if let Some(results) = job.poll() {
            let mut values = Vec::with_capacity(results.len());
            let mut failed = 0;
            for result in results {
                match result {
                    Ok(v) => values.push(Some(v)),
                    Err(e) => {
                        logger::error(format!("integrated intensity: {e}"));
                        values.push(None);
                        failed += 1;
                    }
                }
            }
            let (min, max) = values
                .iter()
                .flatten()
                .fold((f64::MAX, f64::MIN), |(lo, hi), v| {
                    (lo.min(v.intensity), hi.max(v.intensity))
                });
            let span = (max - min).max(max.abs() * 1e-3).max(1e-9);
            // Default threshold below every value: nothing excluded yet.
            view.threshold = min - span * 0.05;
            view.threshold_bounds = (min - span * 0.1, max + span * 0.1);
            logger::log(format!(
                "integrated intensities: {} images (range {:.4e} to {:.4e}), {} failed",
                values.len() - failed,
                min,
                max,
                failed
            ));
            view.intensities = Some(IntensityCache {
                total: key.0,
                first: key.1.clone(),
                values,
                kept_len: files.len(),
                failed,
            });
            view.intensity_job = None;
        } else {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(format!(
                    "computing integrated intensities… {}/{}",
                    job.done(),
                    job.total
                ));
            });
            ctx.request_repaint_after(Duration::from_millis(300));
        }
    } else if view.intensities.is_none() {
        ui.horizontal(|ui| {
            if ui.button("📊 Compute integrated intensities").clicked() {
                logger::log(format!(
                    "computing integrated intensities of {} images ({} old revisions included)…",
                    files.len() + superseded.len(),
                    superseded.len()
                ));
                let mut scan_files = files.clone();
                scan_files.extend(superseded.iter().cloned());
                view.intensity_job = Some(IntensityScan::start(scan_files));
            }
            ui.label(
                RichText::new("(reads every image once — can take a while)")
                    .weak()
                    .size(11.0),
            );
        });
    }

    let manual_excluded_count = files
        .iter()
        .filter(|f| {
            tof::run_number(&f.file_name().unwrap_or_default().to_string_lossy())
                .is_some_and(|r| view.excluded_runs.contains(&r))
        })
        .count();

    let Some(cache) = &view.intensities else {
        if manual_excluded_count > 0 {
            ui.label(
                RichText::new(format!(
                    "{manual_excluded_count} of {} images excluded manually",
                    files.len()
                ))
                .strong(),
            );
        }
        return;
    };
    if cache.failed > 0 {
        ui.colored_label(
            Color32::from_rgb(240, 180, 60),
            format!("{} image(s) could not be read", cache.failed),
        );
    }

    // X axis: file index after sorting the used files by angle (falls back to
    // plain file order until the angle retrieval is set up). Old revisions
    // sit at the same file index as the retake that replaced them.
    let angles = view.per_file_angles(&files);
    let kept_len = cache.kept_len.min(files.len());
    let mut order: Vec<usize> = (0..kept_len).collect();
    if let Some(angles) = &angles {
        order.sort_by(|&a, &b| match (angles.get(a).copied().flatten(), angles.get(b).copied().flatten()) {
            (Some(x), Some(y)) => x.total_cmp(&y),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.cmp(&b),
        });
    }
    let mut rank = vec![0usize; order.len()];
    for (position, &index) in order.iter().enumerate() {
        rank[index] = position;
    }
    // Revision base of each used file → its (rank, index), so a superseded
    // file lands on its replacement's x position and angle.
    let base_of: HashMap<String, usize> = files
        .iter()
        .take(kept_len)
        .enumerate()
        .map(|(i, f)| (white_beam::revision_base(f), i))
        .collect();

    ui.checkbox(&mut view.intensity_log, "log y scale");
    let log_scale = view.intensity_log;
    let y_of = |v: f64| if log_scale { v.max(1e-30).log10() } else { v };

    /// One dot with everything the tooltip / click handling needs.
    #[derive(Clone)]
    struct Dot {
        x: f64,
        y: f64,
        angle: Option<f64>,
        run: Option<u32>,
        intensity: f64,
        old_revision: bool,
    }

    let threshold = view.threshold;
    let is_manual = |v: &ImageIntensity| v.run_number.is_some_and(|r| view.excluded_runs.contains(&r));
    let mut kept: Vec<[f64; 2]> = Vec::new();
    let mut dropped: Vec<[f64; 2]> = Vec::new();
    let mut manual: Vec<[f64; 2]> = Vec::new();
    let mut old_revisions: Vec<[f64; 2]> = Vec::new();
    let mut dots: Vec<Dot> = Vec::new();
    for (index, v) in cache.values.iter().enumerate() {
        let Some(v) = v else { continue };
        let old_revision = index >= kept_len;
        // Old revisions borrow the position/angle of the retake that
        // replaced them.
        let kept_index = if old_revision {
            base_of.get(&white_beam::revision_base(&v.path)).copied()
        } else {
            Some(index)
        };
        let x = kept_index
            .and_then(|k| rank.get(k))
            .map(|&r| r as f64)
            .unwrap_or(rank.len() as f64);
        let angle = kept_index
            .and_then(|k| angles.as_ref().and_then(|a| a.get(k).copied().flatten()));
        let point = [x, y_of(v.intensity)];
        dots.push(Dot {
            x: point[0],
            y: point[1],
            angle,
            run: v.run_number,
            intensity: v.intensity,
            old_revision,
        });
        if old_revision {
            old_revisions.push(point);
        } else if is_manual(v) {
            manual.push(point);
        } else if v.intensity < threshold {
            dropped.push(point);
        } else {
            kept.push(point);
        }
    }

    const PLOT_HEIGHT: f32 = 340.0;
    let mut released = false;
    let mut hovered_dot: Option<Dot> = None;
    let mut plot_clicked = false;
    ui.horizontal(|ui| {
        // In log mode the slider works in log10 space so its position always
        // matches the plotted threshold line.
        if log_scale {
            let bounds = (
                view.threshold_bounds.0.max(1e-30).log10(),
                view.threshold_bounds.1.max(1e-30).log10(),
            );
            let mut value = view.threshold.max(1e-30).log10();
            released = vertical_threshold_slider(
                ui,
                &mut value,
                bounds,
                PLOT_HEIGHT,
                &mut view.threshold_dragging,
            );
            view.threshold = 10f64.powf(value);
        } else {
            released = vertical_threshold_slider(
                ui,
                &mut view.threshold,
                view.threshold_bounds,
                PLOT_HEIGHT,
                &mut view.threshold_dragging,
            );
        }
        let x_label = if angles.is_some() {
            "file index (sorted by angle)"
        } else {
            "file index (set up the projection angles to sort)"
        };
        // Separate plot ids per scale: egui_plot remembers zoom/bounds per
        // id, and the two scales live in completely different value ranges.
        let plot_id = if log_scale {
            "wb_intensity_plot_log"
        } else {
            "wb_intensity_plot_lin"
        };
        let plot_response = egui_plot::Plot::new(plot_id)
            .height(PLOT_HEIGHT)
            .x_axis_label(x_label)
            .y_axis_label(if log_scale {
                "integrated intensity (log scale)"
            } else {
                "integrated intensity"
            })
            // Readable ticks: scientific notation instead of a wall of
            // zeros; in log mode the tick shows the real intensity the
            // log10 position corresponds to.
            .y_axis_formatter(move |mark, _range| {
                let value = if log_scale {
                    10f64.powf(mark.value)
                } else {
                    mark.value
                };
                if value == 0.0 {
                    "0".to_owned()
                } else {
                    format!("{value:.1e}")
                }
            })
            .y_axis_min_width(52.0)
            .legend(egui_plot::Legend::default())
            .show(ui, |plot_ui| {
                plot_ui.hline(
                    egui_plot::HLine::new("threshold", y_of(view.threshold))
                        .color(Color32::from_rgb(230, 100, 100))
                        .width(1.5),
                );
                plot_ui.points(
                    egui_plot::Points::new("kept", kept.clone())
                        .radius(3.0)
                        .color(Color32::from_rgb(100, 170, 255)),
                );
                plot_ui.points(
                    egui_plot::Points::new("below threshold", dropped.clone())
                        .radius(3.0)
                        .color(Color32::from_gray(110)),
                );
                plot_ui.points(
                    egui_plot::Points::new("manual", manual.clone())
                        .radius(3.5)
                        .color(Color32::from_rgb(255, 160, 70)),
                );
                plot_ui.points(
                    egui_plot::Points::new("old revision", old_revisions.clone())
                        .radius(3.0)
                        .color(Color32::from_rgb(200, 120, 255)),
                );
                // Closest dot within 14 px of the pointer, for the tooltip
                // and click-to-exclude.
                if let Some(pointer) = plot_ui.pointer_coordinate() {
                    let transform = plot_ui.transform();
                    let pointer_pos =
                        transform.position_from_point(&pointer);
                    let mut best_d2 = 14.0f32 * 14.0;
                    for dot in &dots {
                        let screen = transform
                            .position_from_point(&egui_plot::PlotPoint::new(dot.x, dot.y));
                        let d2 = screen.distance_sq(pointer_pos);
                        if d2 < best_d2 {
                            best_d2 = d2;
                            hovered_dot = Some(dot.clone());
                        }
                    }
                }
            });
        if let Some(dot) = &hovered_dot {
            plot_clicked = plot_response.response.clicked();
            plot_response.response.on_hover_ui_at_pointer(|ui| {
                ui.label(RichText::new(format!("file index: {}", dot.x as usize)).strong());
                ui.label(match dot.angle {
                    Some(a) => format!("angle: {a:.3}°"),
                    None => "angle: n/a".to_owned(),
                });
                ui.label(match dot.run {
                    Some(r) => format!("run: {r}"),
                    None => "run: n/a".to_owned(),
                });
                ui.label(format!("intensity: {:.4e}", dot.intensity));
                if dot.old_revision {
                    ui.label(
                        RichText::new("old revision — excluded automatically")
                            .color(Color32::from_rgb(200, 120, 255)),
                    );
                }
            });
        }
    });
    // A click on a dot appends its run to the manual exclusion list (old
    // revisions are already out, and their run number is shared with the
    // retake that replaced them — excluding it would drop the retake too).
    if plot_clicked
        && let Some(dot) = hovered_dot.as_ref().filter(|d| !d.old_revision)
        && let Some(run) = dot.run
        && !view.excluded_runs.contains(&run)
    {
        let trimmed = view.exclude_text.trim().trim_end_matches(',').trim();
        view.exclude_text = if trimmed.is_empty() {
            run.to_string()
        } else {
            format!("{trimmed}, {run}")
        };
        view.excluded_runs.insert(run);
        view.exclude_error = None;
        logger::log(format!("run {run} added to the exclusion list (clicked on the plot)"));
    }

    let n_kept = kept.len();
    let line = format!(
        "keeping {n_kept} of {} images — {} below threshold, {} excluded manually, {} old revisions",
        kept.len() + dropped.len() + manual.len(),
        dropped.len(),
        manual.len(),
        old_revisions.len()
    );
    if released {
        logger::log(format!(
            "intensity threshold set to {:.4e}: {line}",
            view.threshold
        ));
    }
    if n_kept == 0 {
        ui.colored_label(Color32::from_rgb(240, 180, 60), format!("{line} — nothing left!"));
    } else {
        ui.label(RichText::new(line).strong());
    }
}

/// A custom-painted vertical single-handle threshold slider; the zone below
/// the handle (excluded) is tinted red. Returns `true` when a drag ends.
fn vertical_threshold_slider(
    ui: &mut egui::Ui,
    value: &mut f64,
    bounds: (f64, f64),
    height: f32,
    dragging: &mut bool,
) -> bool {
    use egui::{Pos2, Rect, Sense, Stroke, vec2};
    const WIDTH: f32 = 34.0;
    const HANDLE_R: f32 = 7.0;
    let (rect, response) = ui.allocate_exact_size(vec2(WIDTH, height), Sense::click_and_drag());
    let span = (bounds.1 - bounds.0).max(f64::EPSILON);
    let inner_top = rect.top() + HANDLE_R;
    let inner_height = (rect.height() - 2.0 * HANDLE_R).max(1.0);
    let to_y = |v: f64| inner_top + (((bounds.1 - v) / span) as f32) * inner_height;
    let to_v = |y: f32| bounds.1 - f64::from((y - inner_top) / inner_height).clamp(0.0, 1.0) * span;

    if let Some(pos) = response.interact_pointer_pos() {
        if response.drag_started() || response.clicked() {
            *dragging = true;
        }
        if *dragging {
            *value = to_v(pos.y);
        }
    }
    let released = *dragging && response.drag_stopped();
    if released {
        *dragging = false;
    }

    let painter = ui.painter();
    let cx = rect.center().x;
    painter.line_segment(
        [Pos2::new(cx, inner_top), Pos2::new(cx, inner_top + inner_height)],
        Stroke::new(4.0, Color32::from_gray(70)),
    );
    // Excluded zone: below the handle.
    let y = to_y(*value);
    painter.rect_filled(
        Rect::from_min_max(
            Pos2::new(cx - 3.0, y),
            Pos2::new(cx + 3.0, inner_top + inner_height),
        ),
        2.0,
        Color32::from_rgba_unmultiplied(230, 100, 100, 120),
    );
    painter.circle_filled(Pos2::new(cx, y), HANDLE_R, Color32::from_gray(230));
    if response.hovered() {
        ui.ctx().set_cursor_icon(egui::CursorIcon::ResizeVertical);
    }
    released
}

/// The projection angles of every sample image per the chosen method —
/// computed on the fly for the naming convention and the ASCII list, read on
/// a background thread (and cached per selection) for the metadata.
fn collect_angles(view: &mut WhiteBeamView, ctx: &egui::Context) -> AngleData {
    let files: Vec<PathBuf> = view
        .sample
        .selected
        .iter()
        .flat_map(|(_, files, _)| files.iter().cloned())
        .collect();
    if files.is_empty() {
        return AngleData::Invalid("select the sample folder(s) first".to_owned());
    }
    match view.angle_source {
        AngleSource::NamingConvention => {
            let picked: Vec<usize> = view
                .nc_checked
                .iter()
                .enumerate()
                .filter(|(_, c)| **c)
                .map(|(i, _)| i)
                .collect();
            let [i, j] = picked.as_slice() else {
                return AngleData::Invalid(
                    "set up the naming convention above (2 fields) first".to_owned(),
                );
            };
            let mut angles = Vec::with_capacity(files.len());
            let mut failed = 0;
            for file in &files {
                match white_beam::angle_from_fields(file, *i, *j) {
                    Some(a) => angles.push(a.rem_euclid(360.0)),
                    None => failed += 1,
                }
            }
            if angles.is_empty() {
                AngleData::Invalid("no file name yields an angle with those fields".to_owned())
            } else {
                AngleData::Ready(angles, failed)
            }
        }
        AngleSource::AsciiFile => match &view.ascii_angles {
            None => AngleData::Invalid("import the list of angles above first".to_owned()),
            Some(angles) if angles.len() == files.len() => {
                AngleData::Ready(angles.clone(), 0)
            }
            Some(angles) => AngleData::Invalid(format!(
                "{} angles for {} images — fix the ASCII list first",
                angles.len(),
                files.len()
            )),
        },
        AngleSource::Metadata => {
            let key = (files.len(), files.first().cloned());
            if let Some((n, first, results)) = &view.meta_angles
                && (*n, first.clone()) == key
            {
                let angles: Vec<f64> = results
                    .iter()
                    .filter_map(|r| r.as_ref().ok())
                    .map(|a| a.rem_euclid(360.0))
                    .collect();
                let failed = results.len() - angles.len();
                return if angles.is_empty() {
                    AngleData::Invalid("no image carries a metadata angle".to_owned())
                } else {
                    AngleData::Ready(angles, failed)
                };
            }
            match &mut view.meta_job {
                Some(job) => {
                    if let Some(results) = job.poll() {
                        for e in results.iter().filter_map(|r| r.as_ref().err()).take(3) {
                            logger::error(format!("metadata angle: {e}"));
                        }
                        logger::log(format!(
                            "metadata angles read: {}/{} images",
                            results.iter().filter(|r| r.is_ok()).count(),
                            results.len()
                        ));
                        view.meta_angles = Some((key.0, key.1.clone(), results));
                        view.meta_job = None;
                        ctx.request_repaint();
                        AngleData::Pending("finalizing…".to_owned())
                    } else {
                        ctx.request_repaint_after(Duration::from_millis(200));
                        AngleData::Pending(format!(
                            "reading metadata angles… {}/{}",
                            job.done(),
                            job.total
                        ))
                    }
                }
                None => {
                    logger::log(format!(
                        "reading metadata angles of {} sample images…",
                        files.len()
                    ));
                    view.meta_job = Some(MetaAnglesScan::start(files));
                    ctx.request_repaint_after(Duration::from_millis(200));
                    AngleData::Pending("reading metadata angles…".to_owned())
                }
            }
        }
    }
}

/// Use everything, or down-select to a percentage with the coverage-first
/// sampling (0° and 180° always kept), previewed on a polar plot.
fn data_to_use_ui(ui: &mut egui::Ui, view: &mut WhiteBeamView) {
    let ctx = ui.ctx().clone();
    let mut use_percentage = view.use_percentage;
    ui.radio_value(&mut use_percentage, false, "use all projections");
    ui.radio_value(&mut use_percentage, true, "use percentage of the projections");
    if use_percentage != view.use_percentage {
        logger::log(if use_percentage {
            format!("data to use: {}% of the projections", view.percentage)
        } else {
            "data to use: all projections".to_owned()
        });
        view.use_percentage = use_percentage;
    }
    if view.use_percentage {
        let response = ui.add(
            egui::Slider::new(&mut view.percentage, 1..=100)
                .suffix("%")
                .text("of the projections"),
        );
        if response.drag_stopped() || (response.changed() && !response.dragged()) {
            logger::log(format!("data to use: {}% of the projections", view.percentage));
        }
    }
    ui.add_space(6.0);

    match collect_angles(view, &ctx) {
        AngleData::Invalid(msg) => {
            ui.label(RichText::new(msg).weak());
        }
        AngleData::Pending(msg) => {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(msg);
            });
        }
        AngleData::Ready(angles, failed) => {
            if failed > 0 {
                ui.colored_label(
                    Color32::from_rgb(240, 180, 60),
                    format!("{failed} image(s) without an angle value are ignored"),
                );
            }
            if !view.use_percentage {
                ui.label(
                    RichText::new(format!("using all {} projections", angles.len())).strong(),
                );
                return;
            }
            let n = ((view.percentage as f64 / 100.0) * angles.len() as f64).round() as usize;
            let n = n.max(5.min(angles.len()));
            let used: Vec<f64> = white_beam::select_coverage(&angles, n)
                .iter()
                .map(|&i| angles[i])
                .collect();
            ui.label(
                RichText::new(format!(
                    "using {} of {} projections",
                    used.len(),
                    angles.len()
                ))
                .strong(),
            );
            ui.label(
                RichText::new("coverage-first selection — 0° and 180° are always kept")
                    .weak()
                    .size(12.0),
            );
            polar_plot(ui, &angles, &used);
        }
    }
}

/// All angles (small, blue) and the ones that will be used (larger, green)
/// at a fixed radius on a polar view — the coverage at a glance.
fn polar_plot(ui: &mut egui::Ui, all: &[f64], used: &[f64]) {
    use egui::{Align2, FontId, Pos2, Sense, Stroke, vec2};
    const SIZE: f32 = 250.0;
    let (rect, _) = ui.allocate_exact_size(vec2(SIZE, SIZE), Sense::hover());
    if !ui.is_rect_visible(rect) {
        return;
    }
    let painter = ui.painter();
    let center = rect.center();
    let radius = SIZE / 2.0 - 18.0;
    let pos_at = |deg: f64, r: f32| {
        let rad = deg.to_radians();
        Pos2::new(
            center.x + r * rad.cos() as f32,
            center.y - r * rad.sin() as f32,
        )
    };
    painter.circle_stroke(center, radius, Stroke::new(1.0, Color32::from_gray(90)));
    for deg in [0.0, 90.0, 180.0, 270.0] {
        painter.line_segment(
            [center, pos_at(deg, radius)],
            Stroke::new(0.5, Color32::from_gray(60)),
        );
        painter.text(
            pos_at(deg, radius + 11.0),
            Align2::CENTER_CENTER,
            format!("{deg:.0}°"),
            FontId::proportional(10.0),
            Color32::from_gray(160),
        );
    }
    for &a in all {
        painter.circle_filled(pos_at(a, radius * 0.8), 2.5, Color32::from_rgb(100, 170, 255));
    }
    for &a in used {
        painter.circle_filled(pos_at(a, radius * 0.8), 4.0, Color32::from_rgb(120, 200, 120));
    }
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
            if view.first_sample_image().is_none() {
                ui.label(
                    RichText::new("select at least one sample folder to set up the convention")
                        .weak(),
                );
                return;
            }
            // The current example must still belong to the selection;
            // otherwise fall back to the first sample image and rebuild.
            let example_valid = view.nc_example.as_ref().is_some_and(|e| {
                view.sample
                    .selected
                    .iter()
                    .any(|(_, files, _)| files.contains(e))
            });
            if !example_valid {
                let example = view.first_sample_image().cloned().expect("checked above");
                view.nc_fields = white_beam::name_fields(&example);
                view.nc_checked = vec![false; view.nc_fields.len()];
                if let Some((i, j)) = white_beam::default_angle_fields(view.nc_fields.len()) {
                    view.nc_checked[i] = true;
                    view.nc_checked[j] = true;
                }
                view.nc_example = Some(example);
            }
            let example = view.nc_example.clone().expect("set above");
            ui.horizontal(|ui| {
                if ui
                    .small_button("🎲")
                    .on_hover_text(
                        "try another randomly picked file name, keeping the checked \
                         fields, to make sure they are the right ones",
                    )
                    .clicked()
                {
                    let files: Vec<&PathBuf> = view
                        .sample
                        .selected
                        .iter()
                        .flat_map(|(_, files, _)| files.iter())
                        .collect();
                    if files.len() > 1 {
                        let nanos = std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.subsec_nanos() as usize)
                            .unwrap_or(0);
                        let mut index = nanos % files.len();
                        if files[index] == &example {
                            index = (index + 1) % files.len();
                        }
                        let new_example = files[index].clone();
                        let fields = white_beam::name_fields(&new_example);
                        // Same field count: the checked fields carry over so
                        // they can be validated; otherwise back to defaults.
                        if fields.len() != view.nc_fields.len() {
                            view.nc_checked = vec![false; fields.len()];
                            if let Some((i, j)) = white_beam::default_angle_fields(fields.len()) {
                                view.nc_checked[i] = true;
                                view.nc_checked[j] = true;
                            }
                        }
                        view.nc_fields = fields;
                        view.nc_example = Some(new_example);
                    }
                }
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
            let example = view.nc_example.clone().expect("set above");
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
        egui::CollapsingHeader::new(
            RichText::new("Combine the TOF images of each run (save to HDF5)").strong(),
        )
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
                "the TOF images ({}) inside each run are averaged into ONE projection: \
                 {} sample runs (one per angle) and {} ob runs — runs are not combined \
                 with each other",
                sel.describe(),
                kept_sample.len(),
                kept_ob.len(),
            ));
        }
    }

    if ui
        .checkbox(
            &mut view.merge_same_angle,
            "Combine runs having the same angle (the exception: this one merges runs)",
        )
        .on_hover_text(
            "off: when several runs share a projection angle, only the one with the \
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
        .add_enabled(
            ready,
            egui::Button::new("▶ Combine the TOF images of each run (mean)"),
        )
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
                        angle_deg: None,
                    })
                })
                .collect()
        };
        let sample_runs = to_combine(&kept_sample);
        let ob_runs = to_combine(&kept_ob);
        logger::log(format!(
            "combining the TOF images ({}) of each run with the mean: {} sample runs, {} ob runs",
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
        let saving = view.save_job.is_some();
        let mut jump = false;
        ui.horizontal(|ui| {
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
                    logger::log(format!("saving combined data to {}", path.display()));
                    view.save_status = None;
                    view.save_job = Some(SaveJob::start(
                        path,
                        std::sync::Arc::clone(output),
                        meta.clone(),
                    ));
                }
            }
            jump = ui
                .button("🚀 Continue to pre-processing")
                .on_hover_text("work on the combined stack right away — saving is optional")
                .clicked();
        });
        if jump {
            logger::log("continuing to pre-processing with the combined TOF stack");
            let pseudo = session
                .ipts
                .path
                .join("shared")
                .join("tof_combined (not saved).h5");
            view.goto_preprocess = Some(combine::stack_from_output(output, &meta, pseudo));
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
            // A finished HDF5 load jumps straight to pre-processing.
            if let Some(job) = &mut self.load_job {
                match job.poll() {
                    Some(Ok(stack)) => {
                        let preprocessed = stack_is_preprocessed(&stack);
                        logger::log(format!(
                            "stack loaded: {} — {} projections, {} ob — jumping to {}",
                            stack.path.display(),
                            stack.sample.len(),
                            stack.ob.len(),
                            if preprocessed {
                                "the reconstruction evaluation (pre-processing checkpoint)"
                            } else {
                                "pre-processing"
                            }
                        ));
                        self.load_job = None;
                        self.screen = if preprocessed {
                            let cor = stack.center_of_rotation;
                            Screen::Recon(ReconView::new(stack, cor))
                        } else {
                            Screen::Stack(StackView::new(stack))
                        };
                    }
                    Some(Err(e)) => {
                        logger::error(format!("loading saved stack failed: {e}"));
                        self.load_error = Some(e);
                        self.load_job = None;
                    }
                    None => ctx.request_repaint_after(Duration::from_millis(300)),
                }
            }
            if !matches!(self.screen, Screen::Setup) {
                return;
            }
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
            egui::CentralPanel::default().show(ui, |ui| match &mut self.screen {
                Screen::Workflow { session, view } => {
                    back = workflow_ui(ui, session, view, self.logo.as_ref(), &mut self.log_view_open);
                }
                Screen::Stack(view) => {
                    back = stack_ui(ui, view, self.logo.as_ref(), &mut self.log_view_open);
                }
                Screen::Recon(view) => {
                    back = recon_ui(ui, view, self.logo.as_ref(), &mut self.log_view_open);
                }
                Screen::Setup => {}
            });
            // A workflow's "continue to pre-processing" button hands its
            // combined stack over without going through a file.
            let pending = match &mut self.screen {
                Screen::Workflow {
                    view: WorkflowView::Tof(tof_view),
                    ..
                } => tof_view.goto_preprocess.take(),
                Screen::Workflow {
                    view: WorkflowView::WhiteBeam(wb_view),
                    ..
                } => wb_view.goto_preprocess.take(),
                _ => None,
            };
            // Same for pre-processing → reconstruction evaluation.
            let recon = match &mut self.screen {
                Screen::Stack(view) if view.goto_recon => {
                    view.goto_recon = false;
                    let cor = view.cor_result.or(view.stack.center_of_rotation);
                    Some(ReconView {
                        stack: std::sync::Arc::clone(&view.stack),
                        cor,
                        optimizer_job: None,
                        reload_job: None,
                        opt_error: None,
                    })
                }
                _ => None,
            };
            if let Some(view) = recon {
                logger::log("continuing to the reconstruction evaluation");
                self.screen = Screen::Recon(view);
            } else if let Some(stack) = pending {
                self.screen = Screen::Stack(StackView::new(stack));
            } else if back {
                logger::log("returned to setup screen");
                self.screen = Screen::Setup;
            }
        }
    }
}
