//! Orientation of the raw detector frames on load.
//!
//! The Python pipeline loads Timepix TIFFs with `swapaxes(0, 1)` and CCD
//! FITS frames with `np.flipud`; the same convention applies here (see the
//! shared [`detector_orientation`] crate): Timepix → transposed, CCD
//! (iKon-XL) → flipped vertically, QHY → not decided yet, loaded as-is. The
//! manual 90° [`crate::rotate`] step stays available on top of it.
//!
//! The automatic guess follows the workflow: TOF means a Timepix detector,
//! white beam maps the chosen camera. The user can override it with the
//! "Orientation" combobox next to the detector one; the override is carried
//! by the [`crate::session::Session`] (prefilled from the debug config).

pub use detector_orientation::{Detector as OrientDetector, Orientation, Selection, Source};

use crate::instrument::Instrument;
use crate::white_beam::WbDetector;

/// Automatic guess for the TOF workflow: always a Timepix detector.
pub fn auto_for_tof(user: Option<OrientDetector>) -> Selection {
    Selection { auto: Some((OrientDetector::Timepix, Source::Path)), manual: user }
}

/// Automatic guess for the white-beam workflow: from the camera picked on
/// VENUS (MARS has a single CCD).
pub fn auto_for_white_beam(
    instrument: Instrument,
    detector: WbDetector,
    user: Option<OrientDetector>,
) -> Selection {
    let auto = match instrument {
        Instrument::Mars => OrientDetector::Ccd,
        Instrument::Venus => match detector {
            WbDetector::IkonXl => OrientDetector::Ccd,
            WbDetector::Qhy => OrientDetector::Qhy,
            WbDetector::Scmos => OrientDetector::Unknown,
        },
    };
    Selection { auto: Some((auto, Source::Path)), manual: user }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guesses_follow_workflow() {
        assert_eq!(auto_for_tof(None).orientation(), Orientation::Transpose);
        assert_eq!(
            auto_for_white_beam(Instrument::Venus, WbDetector::IkonXl, None).orientation(),
            Orientation::FlipVertical
        );
        assert_eq!(
            auto_for_white_beam(Instrument::Venus, WbDetector::Qhy, None).orientation(),
            Orientation::Identity
        );
        assert_eq!(
            auto_for_white_beam(Instrument::Mars, WbDetector::Scmos, None).orientation(),
            Orientation::FlipVertical
        );
        // the user override wins
        let s = auto_for_tof(Some(OrientDetector::Unknown));
        assert_eq!(s.orientation(), Orientation::Identity);
        assert!(!s.is_auto());
    }
}
