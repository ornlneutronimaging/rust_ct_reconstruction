//! Discovery and validation of the IPTS experiment folders the user can read.
//!
//! An experiment counts as accessible when its directory can actually be
//! listed (`read_dir` succeeds): the IPTS folders are protected by ACLs, so
//! permission bits alone cannot answer the question.

use crate::instrument::Instrument;
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};

/// One experiment folder the user is allowed to read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IptsEntry {
    /// Directory name, e.g. `IPTS-36967`.
    pub name: String,
    /// Numeric part, used to sort the list newest-first.
    pub number: u64,
    /// Full path, e.g. `/SNS/VENUS/IPTS-36967`.
    pub path: PathBuf,
}

/// A scan of an instrument root running on a background thread; probing
/// several hundred directories on the network filesystem is too slow for the
/// UI thread.
pub struct IptsScan {
    rx: Receiver<Result<Vec<IptsEntry>, String>>,
}

impl IptsScan {
    pub fn start(instrument: Instrument) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let _ = tx.send(scan_root(instrument));
        });
        Self { rx }
    }

    /// The scan result, once the background thread has finished.
    pub fn try_finish(&mut self) -> Option<Result<Vec<IptsEntry>, String>> {
        self.rx.try_recv().ok()
    }
}

fn scan_root(instrument: Instrument) -> Result<Vec<IptsEntry>, String> {
    let root = PathBuf::from(instrument.root());
    let dir = std::fs::read_dir(&root).map_err(|e| format!("cannot list {}: {e}", root.display()))?;
    let mut entries = Vec::new();
    for item in dir.flatten() {
        let name = item.file_name().to_string_lossy().into_owned();
        let Some(number) = ipts_number(&name) else {
            continue;
        };
        let path = item.path();
        if std::fs::read_dir(&path).is_ok() {
            entries.push(IptsEntry { name, number, path });
        }
    }
    entries.sort_by(|a, b| b.number.cmp(&a.number).then_with(|| a.name.cmp(&b.name)));
    Ok(entries)
}

/// `IPTS-<digits>` → the digits as a number, `None` for anything else.
fn ipts_number(name: &str) -> Option<u64> {
    let digits = name.strip_prefix("IPTS-")?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

/// Interpret manual input (`36967` or `IPTS-36967`, prefix case-insensitive)
/// as a canonical folder name. Typed leading zeros are kept so historic
/// folders such as `IPTS-0001` stay reachable.
pub fn canonical_name(input: &str) -> Option<String> {
    let t = input.trim();
    let digits = if t.len() > 5 && t[..5].eq_ignore_ascii_case("ipts-") {
        &t[5..]
    } else {
        t
    };
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_digit()))
        .then(|| format!("IPTS-{digits}"))
}

/// Turn manual input into a validated, readable entry under the instrument root.
pub fn manual_entry(instrument: Instrument, input: &str) -> Result<IptsEntry, String> {
    let name = canonical_name(input).ok_or_else(|| {
        format!("'{}' is not an IPTS number (e.g. IPTS-36967 or 36967)", input.trim())
    })?;
    let number = ipts_number(&name).ok_or_else(|| format!("'{name}' is out of range"))?;
    let path = PathBuf::from(instrument.root()).join(&name);
    if !path.is_dir() {
        return Err(format!("{} does not exist", path.display()));
    }
    if std::fs::read_dir(&path).is_err() {
        return Err(format!("you do not have read access to {}", path.display()));
    }
    Ok(IptsEntry { name, number, path })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_name_accepts_prefix_and_bare_numbers() {
        assert_eq!(canonical_name("36967").as_deref(), Some("IPTS-36967"));
        assert_eq!(canonical_name(" IPTS-36967 ").as_deref(), Some("IPTS-36967"));
        assert_eq!(canonical_name("ipts-36967").as_deref(), Some("IPTS-36967"));
        assert_eq!(canonical_name("0001").as_deref(), Some("IPTS-0001"));
    }

    #[test]
    fn canonical_name_rejects_garbage() {
        assert_eq!(canonical_name(""), None);
        assert_eq!(canonical_name("IPTS-"), None);
        assert_eq!(canonical_name("IPTS-12a4"), None);
        assert_eq!(canonical_name("hello"), None);
    }

    #[test]
    fn ipts_number_parses_directory_names() {
        assert_eq!(ipts_number("IPTS-36967"), Some(36967));
        assert_eq!(ipts_number("IPTS-0001"), Some(1));
        assert_eq!(ipts_number("shared"), None);
        assert_eq!(ipts_number("IPTS-"), None);
    }
}
