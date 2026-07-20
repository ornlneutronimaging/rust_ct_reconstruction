//! The egui/eframe application shell.
//!
//! The UI is a small state machine: the setup screen (instrument → IPTS →
//! acquisition mode) gates everything else, and the [`Session`] it produces
//! decides which workflow screen the rest of the application shows.

use crate::config;
use crate::instrument::Instrument;
use crate::ipts::{self, IptsEntry, IptsScan};
use crate::logger;
pub use crate::session::{Mode, Session};

use egui::{Align, Color32, Layout, RichText};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};

/// How much of the log file the viewer shows, and how often it auto-refreshes.
const LOG_TAIL_BYTES: u64 = 64 * 1024;
const LOG_REFRESH_EVERY: Duration = Duration::from_secs(2);

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
    Workflow(Session),
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
        Self {
            screen: Screen::Setup,
            instrument: Instrument::Venus,
            scans: HashMap::new(),
            selected: None,
            mode: None,
            filter: String::new(),
            manual: String::new(),
            manual_error: None,
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
        }
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
                    if ui.button("⟳ Refresh").clicked() {
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
        top_right_bar(ui, self.logo.as_ref(), &mut self.log_view_open);
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
                                    if ui.selectable_label(is_selected, &entry.name).clicked() {
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
                let button = egui::Button::new(RichText::new(mode.label()).size(26.0).strong())
                    .min_size(egui::vec2(W, H))
                    .selected(self.mode == Some(mode));
                if ui.add(button).clicked() && self.mode != Some(mode) {
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
                    }
                    Err(e) => {
                        logger::error(format!("debug config IPTS rejected: {e}"));
                        self.selected = None;
                        self.debug_error = Some(e);
                    }
                }
                self.mode = Some(cfg.mode);
                logger::log(format!(
                    "debug mode ON — prefilled from {}: {} / {} / {}",
                    path.display(),
                    cfg.instrument.name(),
                    cfg.ipts,
                    cfg.mode.label()
                ));
                self.debug_info = Some(format!(
                    "{} → {} / {} / {}",
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

/// Top-right corner of the current row: the imaging team logo and the toggle
/// that opens/closes the log viewer side panel.
fn top_right_bar(ui: &mut egui::Ui, logo: Option<&egui::TextureHandle>, log_open: &mut bool) {
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
        if let Some(tex) = logo {
            ui.add(egui::Image::from_texture(tex).max_height(LOGO_MAX_HEIGHT));
        }
        if ui.selectable_label(*log_open, "📜 Log").clicked() {
            *log_open = !*log_open;
        }
    });
}

/// Placeholder for the per-mode workflow screens; returns `true` when the user
/// wants to go back to the setup screen.
fn workflow_ui(
    ui: &mut egui::Ui,
    session: &Session,
    logo: Option<&egui::TextureHandle>,
    log_open: &mut bool,
) -> bool {
    let mut back = false;
    ui.horizontal(|ui| {
        if ui.button("← Back to setup").clicked() {
            back = true;
        }
        top_right_bar(ui, logo, log_open);
    });
    ui.add_space(12.0);
    ui.vertical_centered(|ui| {
        ui.label(
            RichText::new(format!(
                "{} — {} — {}",
                session.instrument.name(),
                session.ipts.name,
                session.mode.label()
            ))
            .size(26.0)
            .strong(),
        );
        ui.add_space(8.0);
        ui.label(format!("Experiment folder: {}", session.ipts.path.display()));
        ui.add_space(32.0);
        ui.label(
            RichText::new(format!(
                "The {} workflow is not implemented yet.",
                session.mode.label()
            ))
            .weak()
            .size(16.0),
        );
    });
    back
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
                    "next → {} workflow: {} / {}",
                    mode.label(),
                    self.instrument.name(),
                    ipts.name
                ));
                self.screen = Screen::Workflow(Session {
                    instrument: self.instrument,
                    ipts,
                    mode,
                });
            }
        } else {
            let mut back = false;
            egui::CentralPanel::default().show(ui, |ui| {
                if let Screen::Workflow(session) = &self.screen {
                    back = workflow_ui(ui, session, self.logo.as_ref(), &mut self.log_view_open);
                }
            });
            if back {
                logger::log("returned to setup screen");
                self.screen = Screen::Setup;
            }
        }
    }
}
