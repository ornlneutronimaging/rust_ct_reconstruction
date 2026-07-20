//! The egui/eframe application shell.
//!
//! The UI is a small state machine: the setup screen (instrument → IPTS →
//! acquisition mode) gates everything else, and the [`Session`] it produces
//! decides which workflow screen the rest of the application shows.

use crate::config;
use crate::instrument::Instrument;
use crate::ipts::{self, IptsEntry, IptsScan};
pub use crate::session::{Mode, Session};

use egui::{Align, Color32, Layout, RichText};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Duration;

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
                Some(Ok(entries)) => *scan = Scan::Done(entries),
                Some(Err(e)) => *scan = Scan::Failed(e),
                None => ctx.request_repaint_after(Duration::from_millis(150)),
            }
        }
    }

    fn setup_ui(&mut self, ui: &mut egui::Ui) -> bool {
        show_logo(ui, self.logo.as_ref());
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
            self.selected = Some(entry);
            self.manual_error = None;
        }
        if manual_requested {
            match ipts::manual_entry(self.instrument, &self.manual) {
                Ok(entry) => {
                    self.selected = Some(entry);
                    self.manual_error = None;
                }
                Err(e) => self.manual_error = Some(e),
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
                if ui.add(button).clicked() {
                    self.mode = Some(mode);
                }
            }
        });
    }

    /// `Next ➡` in the bottom-right corner, enabled once the instrument, the
    /// experiment and the mode are all selected; returns `true` when clicked.
    fn next_button(&mut self, ui: &mut egui::Ui) -> bool {
        let mut go = false;
        let ready = self.selected.is_some() && self.mode.is_some();
        ui.with_layout(Layout::bottom_up(Align::Max), |ui| {
            ui.add_space(8.0);
            let button = egui::Button::new(RichText::new("Next  ➡").size(20.0).strong())
                .min_size(egui::vec2(160.0, 48.0));
            if ui.add_enabled(ready, button).clicked() {
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
                        self.selected = None;
                        self.debug_error = Some(e);
                    }
                }
                self.mode = Some(cfg.mode);
                self.debug_info = Some(format!(
                    "{} → {} / {} / {}",
                    path.display(),
                    cfg.instrument.name(),
                    cfg.ipts,
                    cfg.mode.label()
                ));
            }
            Err(e) => {
                self.debug_mode = false;
                self.debug_info = None;
                self.debug_error = Some(e);
            }
        }
    }

    fn disable_debug(&mut self) {
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
                            self.admin_unlocked = true;
                            self.admin_error = None;
                        } else {
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

/// Imaging team logo in the top-right corner of the current row.
fn show_logo(ui: &mut egui::Ui, logo: Option<&egui::TextureHandle>) {
    ui.with_layout(egui::Layout::right_to_left(egui::Align::Min), |ui| {
        if let Some(tex) = logo {
            ui.add(egui::Image::from_texture(tex).max_height(LOGO_MAX_HEIGHT));
        }
    });
}

/// Placeholder for the per-mode workflow screens; returns `true` when the user
/// wants to go back to the setup screen.
fn workflow_ui(ui: &mut egui::Ui, session: &Session, logo: Option<&egui::TextureHandle>) -> bool {
    let mut back = false;
    ui.horizontal(|ui| {
        if ui.button("← Back to setup").clicked() {
            back = true;
        }
        show_logo(ui, logo);
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
                    back = workflow_ui(ui, session, self.logo.as_ref());
                }
            });
            if back {
                self.screen = Screen::Setup;
            }
        }
    }
}
