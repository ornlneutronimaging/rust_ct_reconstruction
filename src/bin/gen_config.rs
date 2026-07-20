//! Write a debug config file (see `config.rs`), used by the admin debug mode
//! to prefill the setup screen. With no argument it writes Jean's default
//! config — VENUS / IPTS-36202 / TOF — to `<repo>/config/config_jean.h5`.
//!
//! Usage: gen_config [PATH.h5]

use ct_reconstruction::config::{self, DebugConfig};
use ct_reconstruction::instrument::Instrument;
use ct_reconstruction::session::Mode;
use std::path::PathBuf;

fn main() {
    let path = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(config::default_config_path);
    let cfg = DebugConfig {
        instrument: Instrument::Venus,
        ipts: "IPTS-36202".to_owned(),
        mode: Mode::Tof,
    };
    match config::write(&path, &cfg) {
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
