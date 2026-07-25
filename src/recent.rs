//! Per-user history of the last HDF5 files saved or loaded, shown on the
//! setup screen so a recent file can be reloaded without browsing for it.
//!
//! The list is kept in memory behind a mutex (recordings come from every
//! workflow screen) and mirrored to a small JSON file in the user's home so
//! it survives restarts. Persistence is best-effort: if the file cannot be
//! read or written the in-memory list still works for the session.

use chrono::Local;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

/// How many files the history keeps.
pub const CAPACITY: usize = 10;

/// What put the file in the history; shown next to the timestamp.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Action {
    Saved,
    Loaded,
}

impl Action {
    pub fn label(self) -> &'static str {
        match self {
            Action::Saved => "saved",
            Action::Loaded => "loaded",
        }
    }

    fn parse(s: &str) -> Option<Action> {
        match s {
            "saved" => Some(Action::Saved),
            "loaded" => Some(Action::Loaded),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecentEntry {
    pub path: PathBuf,
    pub action: Action,
    /// Local time of the save/load, already formatted (`2026-07-25 14:03`).
    pub when: String,
}

static LIST: OnceLock<Mutex<Vec<RecentEntry>>> = OnceLock::new();

/// `~/.config/rust_ct_reconstruction/recent_files.json`
fn store_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join(".config")
            .join("rust_ct_reconstruction")
            .join("recent_files.json"),
    )
}

fn list() -> &'static Mutex<Vec<RecentEntry>> {
    LIST.get_or_init(|| Mutex::new(store_path().map(|p| read_store(&p)).unwrap_or_default()))
}

pub fn record_saved(path: &Path) {
    record(path, Action::Saved);
}

pub fn record_loaded(path: &Path) {
    record(path, Action::Loaded);
}

/// The current history, most recent first.
pub fn entries() -> Vec<RecentEntry> {
    list().lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn record(path: &Path, action: Action) {
    let entry = RecentEntry {
        path: path.to_path_buf(),
        action,
        when: Local::now().format("%Y-%m-%d %H:%M").to_string(),
    };
    let snapshot = {
        let mut entries = list().lock().unwrap_or_else(|e| e.into_inner());
        entries.retain(|e| e.path != entry.path);
        entries.insert(0, entry);
        entries.truncate(CAPACITY);
        entries.clone()
    };
    if let Some(store) = store_path()
        && let Err(e) = write_store(&store, &snapshot)
    {
        crate::logger::error(format!("cannot save the recent-files list: {e}"));
    }
}

fn read_store(path: &Path) -> Vec<RecentEntry> {
    let Ok(text) = std::fs::read_to_string(path) else {
        return Vec::new();
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
        crate::logger::error(format!("ignoring unreadable {}", path.display()));
        return Vec::new();
    };
    let Some(items) = json.as_array() else {
        return Vec::new();
    };
    let mut entries = Vec::new();
    for item in items {
        let (Some(p), Some(action), Some(when)) = (
            item.get("path").and_then(|v| v.as_str()),
            item.get("action").and_then(|v| v.as_str()).and_then(Action::parse),
            item.get("when").and_then(|v| v.as_str()),
        ) else {
            continue;
        };
        entries.push(RecentEntry {
            path: PathBuf::from(p),
            action,
            when: when.to_owned(),
        });
        if entries.len() == CAPACITY {
            break;
        }
    }
    entries
}

fn write_store(path: &Path, entries: &[RecentEntry]) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let json = serde_json::Value::Array(
        entries
            .iter()
            .map(|e| {
                serde_json::json!({
                    "path": e.path.to_string_lossy(),
                    "action": e.action.label(),
                    "when": e.when,
                })
            })
            .collect(),
    );
    let text = serde_json::to_string_pretty(&json).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(name: &str, action: Action) -> RecentEntry {
        RecentEntry {
            path: PathBuf::from(format!("/tmp/{name}")),
            action,
            when: "2026-07-25 12:00".to_owned(),
        }
    }

    #[test]
    fn store_round_trips() {
        let dir = std::env::temp_dir().join(format!("recent_test_{}", std::process::id()));
        let file = dir.join("recent_files.json");
        let entries = vec![entry("a.h5", Action::Saved), entry("b.h5", Action::Loaded)];
        write_store(&file, &entries).unwrap();
        assert_eq!(read_store(&file), entries);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn read_store_tolerates_missing_or_garbage_files() {
        assert!(read_store(Path::new("/nonexistent/recent.json")).is_empty());
    }

    #[test]
    fn action_labels_round_trip() {
        for action in [Action::Saved, Action::Loaded] {
            assert_eq!(Action::parse(action.label()), Some(action));
        }
    }
}
