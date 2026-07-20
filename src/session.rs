//! The session produced by the setup screen: the instrument, the experiment,
//! and the acquisition mode everything downstream works on.

use crate::instrument::Instrument;
use crate::ipts::IptsEntry;

/// Acquisition mode chosen with the two large buttons on the setup screen.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    WhiteBeam,
    Tof,
}

impl Mode {
    pub fn label(self) -> &'static str {
        match self {
            Mode::WhiteBeam => "White Beam",
            Mode::Tof => "TOF",
        }
    }

    /// Parse a config-file value; case, spaces and underscores are ignored so
    /// `TOF`, `tof`, `White Beam` and `WHITE_BEAM` all work.
    pub fn parse(s: &str) -> Option<Mode> {
        let key: String = s
            .chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .collect::<String>()
            .to_ascii_uppercase();
        match key.as_str() {
            "TOF" => Some(Mode::Tof),
            "WHITEBEAM" => Some(Mode::WhiteBeam),
            _ => None,
        }
    }
}

/// Everything the setup screen hands to the rest of the application.
pub struct Session {
    pub instrument: Instrument,
    pub ipts: IptsEntry,
    pub mode: Mode,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_parse_accepts_common_spellings() {
        assert_eq!(Mode::parse("TOF"), Some(Mode::Tof));
        assert_eq!(Mode::parse("tof"), Some(Mode::Tof));
        assert_eq!(Mode::parse("White Beam"), Some(Mode::WhiteBeam));
        assert_eq!(Mode::parse("WHITE_BEAM"), Some(Mode::WhiteBeam));
        assert_eq!(Mode::parse("whitebeam"), Some(Mode::WhiteBeam));
        assert_eq!(Mode::parse("monochromatic"), None);
    }
}
