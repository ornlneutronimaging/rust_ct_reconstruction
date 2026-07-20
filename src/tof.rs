//! TOF workflow data discovery: the detector in use and the sample /
//! open-beam folder trees under the IPTS autoreduce images directory.
//!
//! On disk (e.g. VENUS): `<ipts>/shared/autoreduce/images/tpx1/raw/ct/`
//! holds one folder per sample; a sample folder holds one folder per
//! projection (angle); a projection folder holds the TOF-binned TIFF images.
//! Open-beam runs follow the same folder-of-image-folders shape under
//! `<ipts>/shared/autoreduce/images/tpx1/ob/`.

use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Detector {
    Tpx1UntilJuly2025,
    Tpx1FromAugust2025,
    Tpx3,
}

impl Detector {
    pub const ALL: [Detector; 3] = [
        Detector::Tpx1UntilJuly2025,
        Detector::Tpx1FromAugust2025,
        Detector::Tpx3,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Detector::Tpx1UntilJuly2025 => "tpx1 - until July 2025",
            Detector::Tpx1FromAugust2025 => "tpx1 - from August 2025",
            Detector::Tpx3 => "tpx3",
        }
    }

    /// Subdirectory of `shared/autoreduce/images` this detector writes to.
    /// The until-July-2025 tpx1 layout is assumed identical for now — adjust
    /// here once its actual folder structure is pinned down.
    fn images_subdir(self) -> &'static str {
        match self {
            Detector::Tpx1UntilJuly2025 | Detector::Tpx1FromAugust2025 => "tpx1",
            Detector::Tpx3 => "tpx3",
        }
    }

    /// Where the CT sample folders live, e.g.
    /// `/SNS/VENUS/IPTS-36202/shared/autoreduce/images/tpx1/raw/ct`.
    pub fn ct_root(self, ipts: &Path) -> PathBuf {
        ipts.join("shared/autoreduce/images")
            .join(self.images_subdir())
            .join("raw/ct")
    }

    /// Where the open-beam folders live, e.g.
    /// `/SNS/VENUS/IPTS-36202/shared/autoreduce/images/tpx1/ob`.
    pub fn ob_root(self, ipts: &Path) -> PathBuf {
        ipts.join("shared/autoreduce/images")
            .join(self.images_subdir())
            .join("ob")
    }
}

/// One folder of images (a projection run or an OB run): every image file
/// directly inside it, full paths, sorted.
#[derive(Clone, Debug)]
pub struct ImageFolder {
    pub name: String,
    pub path: PathBuf,
    pub images: Vec<PathBuf>,
}

pub fn is_image(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .is_some_and(|e| e.eq_ignore_ascii_case("tif") || e.eq_ignore_ascii_case("tiff"))
}

/// Immediate subdirectories of `root`, sorted by name.
pub fn list_subdirs(root: &Path) -> Result<Vec<PathBuf>, String> {
    let dir = std::fs::read_dir(root).map_err(|e| format!("cannot list {}: {e}", root.display()))?;
    let mut subdirs: Vec<PathBuf> = dir
        .flatten()
        .filter(|item| item.file_type().is_ok_and(|t| t.is_dir()))
        .map(|item| item.path())
        .collect();
    subdirs.sort();
    Ok(subdirs)
}

fn images_in(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries =
        std::fs::read_dir(dir).map_err(|e| format!("cannot list {}: {e}", dir.display()))?;
    let mut images: Vec<PathBuf> = entries
        .flatten()
        .map(|item| item.path())
        .filter(|p| p.is_file() && is_image(p))
        .collect();
    images.sort();
    Ok(images)
}

fn folder_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

enum ScanMsg {
    Progress { done: usize, total: usize },
    Done(Result<Vec<ImageFolder>, String>),
}

/// A background inventory of a selected folder: one [`ImageFolder`] per
/// subfolder (per projection for a sample, per run for OBs). Runs on a
/// thread — hundreds of folders holding thousands of images each on the
/// network filesystem is too slow for the UI thread.
pub struct FolderScan {
    pub root: PathBuf,
    rx: Receiver<ScanMsg>,
    pub done: usize,
    pub total: usize,
}

impl FolderScan {
    pub fn start(root: PathBuf) -> Self {
        let (tx, rx) = channel();
        let thread_root = root.clone();
        std::thread::spawn(move || scan_thread(thread_root, tx));
        Self {
            root,
            rx,
            done: 0,
            total: 0,
        }
    }

    /// Drain progress messages; `Some` once the scan has finished.
    pub fn poll(&mut self) -> Option<Result<Vec<ImageFolder>, String>> {
        loop {
            match self.rx.try_recv() {
                Ok(ScanMsg::Progress { done, total }) => {
                    self.done = done;
                    self.total = total;
                }
                Ok(ScanMsg::Done(result)) => return Some(result),
                Err(_) => return None,
            }
        }
    }
}

fn scan_thread(root: PathBuf, tx: Sender<ScanMsg>) {
    let result = (|| {
        let subdirs = list_subdirs(&root)?;
        // A folder whose images sit directly inside (no per-projection
        // subfolders) is inventoried as a single entry.
        if subdirs.is_empty() {
            let images = images_in(&root)?;
            if images.is_empty() {
                return Ok(Vec::new());
            }
            return Ok(vec![ImageFolder {
                name: folder_name(&root),
                path: root.clone(),
                images,
            }]);
        }
        let total = subdirs.len();
        let mut folders = Vec::with_capacity(total);
        for (i, dir) in subdirs.into_iter().enumerate() {
            let images = images_in(&dir)?;
            folders.push(ImageFolder {
                name: folder_name(&dir),
                path: dir,
                images,
            });
            let _ = tx.send(ScanMsg::Progress {
                done: i + 1,
                total,
            });
        }
        Ok(folders)
    })();
    let _ = tx.send(ScanMsg::Done(result));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detector_roots() {
        let ipts = Path::new("/SNS/VENUS/IPTS-36202");
        assert_eq!(
            Detector::Tpx1FromAugust2025.ct_root(ipts),
            Path::new("/SNS/VENUS/IPTS-36202/shared/autoreduce/images/tpx1/raw/ct")
        );
        assert_eq!(
            Detector::Tpx1FromAugust2025.ob_root(ipts),
            Path::new("/SNS/VENUS/IPTS-36202/shared/autoreduce/images/tpx1/ob")
        );
        assert_eq!(
            Detector::Tpx3.ct_root(ipts),
            Path::new("/SNS/VENUS/IPTS-36202/shared/autoreduce/images/tpx3/raw/ct")
        );
    }

    #[test]
    fn image_extensions() {
        assert!(is_image(Path::new("a/b/img_00001.tif")));
        assert!(is_image(Path::new("a/b/IMG.TIFF")));
        assert!(!is_image(Path::new("a/b/notes.txt")));
        assert!(!is_image(Path::new("a/b/no_extension")));
    }
}
