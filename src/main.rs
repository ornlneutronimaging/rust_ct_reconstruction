//! CT Reconstruction — GUI for the VENUS/MARS neutron CT reconstruction
//! workflows (white beam and TOF). The application opens on a setup screen
//! where the user picks the instrument, an IPTS experiment they can read, and
//! the acquisition mode; that choice drives the rest of the UI.

use ct_reconstruction::app::CtApp;

const USAGE: &str = "\
ct_reconstruction — GUI for VENUS/MARS neutron CT reconstruction

USAGE:
  ct_reconstruction

OPTIONS:
  -h, --help    Show this help
";

fn main() -> eframe::Result<()> {
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s => {
                eprintln!("Error: unknown argument: {s}\n\n{USAGE}");
                std::process::exit(2);
            }
        }
    }

    if let Err(e) = ct_reconstruction::logger::init() {
        // The GUI stays usable without logging; the log viewer explains why.
        eprintln!("Warning: logging disabled: {e}");
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            // Tall enough for the whole setup screen (instrument, IPTS list,
            // mode buttons, load-HDF5 row, Next, admin bar) without scrolling.
            .with_inner_size([1280.0, 1000.0])
            .with_min_inner_size([900.0, 700.0])
            .with_title("CT Reconstruction"),
        ..Default::default()
    };

    eframe::run_native(
        "CT Reconstruction",
        native_options,
        Box::new(|cc| {
            // Always use the dark theme, regardless of the system/desktop theme.
            cc.egui_ctx.set_theme(egui::Theme::Dark);
            Ok(Box::new(CtApp::new()))
        }),
    )
}
