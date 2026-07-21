//! Debug/autopopulate configuration stored in HDF5 files.
//!
//! A config file holds three scalar string datasets at the root —
//! `instrument` (`VENUS`/`MARS`), `ipts` (e.g. `IPTS-36202`) and `mode`
//! (`White Beam`/`TOF`). When the admin debug mode is on, the setup screen is
//! prefilled from it. Files can be (re)generated with the `gen_config` binary
//! or inspected with `h5dump`.

use crate::instrument::Instrument;
use crate::session::Mode;
use hdf5_metno::types::VarLenUnicode;
use std::path::{Path, PathBuf};

/// The config debug mode loads at startup — the VENUS white-beam one while
/// that workflow is being developed; the admin Browse button switches to any
/// other (e.g. `config_venus_tof.h5`).
pub const DEFAULT_CONFIG_NAME: &str = "config_venus_white_beam.h5";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DebugConfig {
    pub instrument: Instrument,
    pub ipts: String,
    pub mode: Mode,
}

/// `<repo>/config/config_jean.h5`, resolving the repo root from the running
/// binary (`<repo>/target/release/...`), with the working directory as
/// fallback when the binary lives somewhere else.
pub fn default_config_path() -> PathBuf {
    let root = std::env::current_exe()
        .ok()
        .and_then(|exe| exe.ancestors().nth(3).map(Path::to_path_buf))
        .filter(|repo| repo.join("Cargo.toml").is_file())
        .unwrap_or_else(|| PathBuf::from("."));
    root.join("config").join(DEFAULT_CONFIG_NAME)
}

pub fn read(path: &Path) -> Result<DebugConfig, String> {
    let file = hdf5_metno::File::open(path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let get = |name: &str| -> Result<String, String> {
        file.dataset(name)
            .and_then(|ds| ds.read_scalar::<VarLenUnicode>())
            .map(|v| v.as_str().to_owned())
            .map_err(|e| format!("cannot read '{name}' from {}: {e}", path.display()))
    };
    let instrument_s = get("instrument")?;
    let instrument = Instrument::parse(&instrument_s)
        .ok_or_else(|| format!("unknown instrument '{instrument_s}' in {}", path.display()))?;
    let ipts = get("ipts")?;
    let mode_s = get("mode")?;
    let mode = Mode::parse(&mode_s)
        .ok_or_else(|| format!("unknown mode '{mode_s}' in {}", path.display()))?;
    Ok(DebugConfig { instrument, ipts, mode })
}

pub fn write(path: &Path, config: &DebugConfig) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let file = hdf5_metno::File::create(path)
        .map_err(|e| format!("cannot create {}: {e}", path.display()))?;
    let put = |name: &str, value: &str| -> Result<(), String> {
        let v: VarLenUnicode = value
            .parse()
            .map_err(|e| format!("invalid string for '{name}': {e}"))?;
        file.new_dataset::<VarLenUnicode>()
            .create(name)
            .and_then(|ds| ds.write_scalar(&v))
            .map_err(|e| format!("cannot write '{name}' to {}: {e}", path.display()))
    };
    put("instrument", config.instrument.name())?;
    put("ipts", &config.ipts)?;
    put("mode", config.mode.label())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let path = std::env::temp_dir().join(format!("ct_recon_config_test_{}.h5", std::process::id()));
        let config = DebugConfig {
            instrument: Instrument::Venus,
            ipts: "IPTS-36202".to_owned(),
            mode: Mode::Tof,
        };
        write(&path, &config).unwrap();
        let back = read(&path).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(back, config);
    }

    #[test]
    fn read_missing_file_fails() {
        assert!(read(Path::new("/nonexistent/nope.h5")).is_err());
    }
}
