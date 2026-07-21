//! Write the debug config files (see `config.rs`) used by the admin debug
//! mode to prefill the setup screen. With no argument it writes both default
//! configs into `<repo>/config/`:
//!
//! - `config_venus_white_beam.h5` — VENUS / IPTS-36573 / White Beam
//! - `config_venus_tof.h5`        — VENUS / IPTS-37118 / TOF
//!
//! Usage: gen_config            (write both defaults)
//!        gen_config PATH.h5    (write the white-beam default to PATH)

use ct_reconstruction::config::{self, DebugConfig};
use ct_reconstruction::instrument::Instrument;
use ct_reconstruction::session::Mode;
use std::path::PathBuf;

fn write(path: &PathBuf, cfg: &DebugConfig) {
    match config::write(path, cfg) {
        Ok(()) => println!(
            "wrote {} ({} / {} / {})",
            path.display(),
            cfg.instrument.name(),
            cfg.ipts,
            cfg.mode.label()
        ),
        Err(e) => {
            eprintln!("Error: {e}");
            std::process::exit(1);
        }
    }
}

fn main() {
    let white_beam = DebugConfig {
        instrument: Instrument::Venus,
        ipts: "IPTS-36573".to_owned(),
        mode: Mode::WhiteBeam,
    };
    if let Some(path) = std::env::args().nth(1).map(PathBuf::from) {
        write(&path, &white_beam);
        return;
    }
    let dir = config::default_config_path()
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("config"));
    write(&dir.join("config_venus_white_beam.h5"), &white_beam);
    write(
        &dir.join("config_venus_tof.h5"),
        &DebugConfig {
            instrument: Instrument::Venus,
            ipts: "IPTS-37118".to_owned(),
            mode: Mode::Tof,
        },
    );
}
