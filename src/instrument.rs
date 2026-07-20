//! The neutron imaging instruments the application can work with.

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Instrument {
    Venus,
    Mars,
}

impl Instrument {
    pub const ALL: [Instrument; 2] = [Instrument::Venus, Instrument::Mars];

    pub fn name(self) -> &'static str {
        match self {
            Instrument::Venus => "VENUS",
            Instrument::Mars => "MARS",
        }
    }

    /// Where the experiment (IPTS) folders live for this instrument.
    pub fn root(self) -> &'static str {
        match self {
            Instrument::Venus => "/SNS/VENUS",
            Instrument::Mars => "/HFIR/CG1D",
        }
    }

    /// Parse a config-file value, case-insensitive.
    pub fn parse(s: &str) -> Option<Instrument> {
        match s.trim().to_ascii_uppercase().as_str() {
            "VENUS" => Some(Instrument::Venus),
            "MARS" => Some(Instrument::Mars),
            _ => None,
        }
    }

    /// Beamline caption shown under the instrument selector.
    pub fn description(self) -> &'static str {
        match self {
            Instrument::Venus => "SNS beamline 10 — /SNS/VENUS",
            Instrument::Mars => "HFIR beamline CG-1D — /HFIR/CG1D",
        }
    }
}
