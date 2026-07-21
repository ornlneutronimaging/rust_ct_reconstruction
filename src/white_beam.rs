//! White-beam workflow data discovery: the CCD detector in use and where its
//! sample / open-beam folders live under the experiment.
//!
//! Unlike TOF (folder per projection full of TOF-binned images), a
//! white-beam sample folder holds one TIFF file per projection, with the run
//! number (`Run_<n>`) and the angle (`Ang_<deg>_<millideg>`) in the file
//! name, e.g. `20260604_Run_21775_Trex_CT_300_000s_2_800AngsMin_Ang_000_000_1.tiff`.

use std::path::{Path, PathBuf};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum WbDetector {
    IkonXl,
    Qhy,
    Scmos,
}

impl WbDetector {
    pub const ALL: [WbDetector; 3] = [WbDetector::IkonXl, WbDetector::Qhy, WbDetector::Scmos];

    pub fn label(self) -> &'static str {
        match self {
            WbDetector::IkonXl => "IkonXL",
            WbDetector::Qhy => "QHY",
            WbDetector::Scmos => "sCMOS",
        }
    }

    /// Subdirectory of `<ipts>/images` this detector writes to. Only the
    /// IkonXL location is confirmed; adjust the others once their layouts
    /// are pinned down.
    fn images_subdir(self) -> &'static str {
        match self {
            WbDetector::IkonXl => "ikonxl",
            WbDetector::Qhy => "qhy",
            WbDetector::Scmos => "scmos",
        }
    }

    /// Where the CT sample folders live, e.g.
    /// `/SNS/VENUS/IPTS-36573/images/ikonxl/raw/ct`.
    pub fn ct_root(self, ipts: &Path) -> PathBuf {
        ipts.join("images").join(self.images_subdir()).join("raw/ct")
    }

    /// Where the open-beam folders live, e.g.
    /// `/SNS/VENUS/IPTS-36573/images/ikonxl/ob`.
    pub fn ob_root(self, ipts: &Path) -> PathBuf {
        ipts.join("images").join(self.images_subdir()).join("ob")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_roots() {
        let ipts = Path::new("/SNS/VENUS/IPTS-36573");
        assert_eq!(
            WbDetector::IkonXl.ct_root(ipts),
            Path::new("/SNS/VENUS/IPTS-36573/images/ikonxl/raw/ct")
        );
        assert_eq!(
            WbDetector::IkonXl.ob_root(ipts),
            Path::new("/SNS/VENUS/IPTS-36573/images/ikonxl/ob")
        );
        assert_eq!(
            WbDetector::Qhy.ct_root(ipts),
            Path::new("/SNS/VENUS/IPTS-36573/images/qhy/raw/ct")
        );
    }

    #[test]
    fn labels() {
        let labels: Vec<&str> = WbDetector::ALL.iter().map(|d| d.label()).collect();
        assert_eq!(labels, ["IkonXL", "QHY", "sCMOS"]);
    }
}
