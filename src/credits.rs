//! Credits window: the third-party tools and libraries the pipeline relies
//! on, with links and versions. Python package versions are probed once, in
//! the background, from the same interpreters the pipeline actually runs
//! (so they always reflect the deployed environments); Rust crate versions
//! come from the Cargo.lock compiled into the binary.

use std::collections::HashMap;
use std::path::Path;
use std::sync::mpsc::{Receiver, channel};

use eframe::egui::{self, RichText};

struct Entry {
    name: &'static str,
    url: &'static str,
    /// Key into the probed-versions map; `None` for entries without one.
    key: Option<&'static str>,
    role: &'static str,
}

const PYTHON_TOOLS: &[Entry] = &[
    Entry {
        name: "TomoPy",
        url: "https://tomopy.readthedocs.io",
        key: Some("tomopy"),
        role: "stripe removal filters and the gridrec reconstruction; the outlier \
               removal step is a native port of its remove_outlier",
    },
    Entry {
        name: "svmbir",
        url: "https://github.com/cabouman/svmbir",
        key: Some("svmbir"),
        role: "model-based iterative reconstruction (MBIR)",
    },
    Entry {
        name: "MBIRJAX",
        url: "https://github.com/cabouman/mbirjax",
        key: Some("mbirjax"),
        role: "JAX-based model-based iterative reconstruction",
    },
    Entry {
        name: "Algotom",
        url: "https://github.com/algotom/algotom",
        key: Some("algotom"),
        role: "FBP and gridrec reconstruction methods",
    },
    Entry {
        name: "bm3dornl",
        url: "https://github.com/ornlneutronimaging/bm3dornl",
        key: Some("bm3dornl"),
        role: "BM3D ring artifact removal (the bm3dornl pre-processing step)",
    },
    Entry {
        name: "NeuNorm",
        url: "https://github.com/ornlneutronimaging/NeuNorm",
        key: Some("NeuNorm"),
        role: "normalization by open beam and dark current",
    },
    Entry {
        name: "NumPy",
        url: "https://numpy.org",
        key: Some("numpy"),
        role: "array backbone of every Python step",
    },
    Entry {
        name: "SciPy",
        url: "https://scipy.org",
        key: Some("scipy"),
        role: "scientific computing under the Python steps; the median-filter \
               cleaning method is a native port of its median_filter",
    },
    Entry {
        name: "tifffile",
        url: "https://github.com/cgohlke/tifffile",
        key: Some("tifffile"),
        role: "writing the reconstructed TIFF slices",
    },
];

const PORTED_TOOLS: &[Entry] = &[Entry {
    name: "MuhRec / imagingsuite",
    url: "https://github.com/neutronimaging/imagingsuite",
    key: None,
    role: "the morphological spot cleaning (MorphSpotClean) and the tilt-axis \
           algorithm are native ports of MuhRec's implementations",
}];

const RUST_LIBS: &[Entry] = &[
    Entry {
        name: "egui / eframe",
        url: "https://github.com/emilk/egui",
        key: Some("egui"),
        role: "the GUI framework",
    },
    Entry {
        name: "egui_plot",
        url: "https://github.com/emilk/egui_plot",
        key: Some("egui_plot"),
        role: "histograms, profiles and other plots",
    },
    Entry {
        name: "hdf5-metno",
        url: "https://github.com/metno/hdf5-rust",
        key: Some("hdf5-metno"),
        role: "reading and writing the HDF5 checkpoints",
    },
    Entry {
        name: "tiff",
        url: "https://github.com/image-rs/image-tiff",
        key: Some("tiff"),
        role: "reading the raw TIFF projections",
    },
    Entry {
        name: "rayon",
        url: "https://github.com/rayon-rs/rayon",
        key: Some("rayon"),
        role: "parallelism of the native processing steps",
    },
    Entry {
        name: "rfd",
        url: "https://github.com/PolyMeilex/rfd",
        key: Some("rfd"),
        role: "native file dialogs",
    },
];

/// Version of a crate in the Cargo.lock this binary was built from.
fn lock_version(name: &str) -> Option<&'static str> {
    const CARGO_LOCK: &str = include_str!("../Cargo.lock");
    let mut lines = CARGO_LOCK.lines();
    while let Some(line) = lines.next() {
        if line.strip_prefix("name = \"").and_then(|l| l.strip_suffix('"')) == Some(name) {
            return lines
                .next()?
                .strip_prefix("version = \"")?
                .strip_suffix('"');
        }
    }
    None
}

/// One `python -c` run printing `package version` lines (missing packages
/// are silently skipped).
fn probe_python(interpreter: &str, packages: &[&str]) -> HashMap<String, String> {
    let snippet = format!(
        "import importlib.metadata as m\n\
         for p in {packages:?}:\n    \
             try:\n        print(p, m.version(p))\n    \
             except Exception:\n        pass\n"
    );
    let mut map = HashMap::new();
    if let Ok(out) = std::process::Command::new(interpreter)
        .arg("-c")
        .arg(snippet)
        .output()
    {
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            if let Some((pkg, version)) = line.split_once(' ') {
                map.insert(pkg.to_owned(), version.trim().to_owned());
            }
        }
    }
    map
}

/// All the probed versions, collected off the UI thread.
fn probe_all() -> HashMap<String, String> {
    // The environments that actually run each step (see the respective
    // modules): reconstruction env, stripes env, and the PATH python that
    // runs NeuNorm.
    let mut map = probe_python(
        crate::recon_run::RECON_PYTHON,
        &[
            "numpy", "tifffile", "svmbir", "mbirjax", "algotom", "tomopy", "scipy",
        ],
    );
    let stripes = probe_python(crate::stripes::TOMOPY_PYTHON, &["tomopy"]);
    if let Some(stripes_tomopy) = stripes.get("tomopy") {
        match map.get("tomopy") {
            Some(recon_tomopy) if recon_tomopy != stripes_tomopy => {
                let merged = format!("{recon_tomopy} (recon) / {stripes_tomopy} (stripes)");
                map.insert("tomopy".to_owned(), merged);
            }
            Some(_) => {}
            None => {
                map.insert("tomopy".to_owned(), stripes_tomopy.clone());
            }
        }
    }
    map.extend(probe_python(crate::normalize::PYTHON, &["NeuNorm"]));

    // bm3dornl is the sibling rust_bm3dornl repo — read its pyproject.toml.
    // …/rust_bm3dornl/src/rust_core/target/release/bm3dornl-gui → repo root
    if let Some(root) = Path::new(crate::bm3dornl::BM3DORNL_GUI_BIN)
        .ancestors()
        .nth(5)
        && let Ok(pyproject) = std::fs::read_to_string(root.join("pyproject.toml"))
        && let Some(version) = pyproject.lines().find_map(|l| {
            l.trim()
                .strip_prefix("version = \"")
                .and_then(|v| v.strip_suffix('"'))
        })
    {
        map.insert("bm3dornl".to_owned(), version.to_owned());
    }
    map
}

#[derive(Default)]
pub struct Credits {
    probe: Option<Receiver<HashMap<String, String>>>,
    versions: Option<HashMap<String, String>>,
}

impl Credits {
    /// The floating credits window; call every frame, draws nothing while
    /// `open` is false. The version probe starts the first time it opens.
    pub fn window(&mut self, ctx: &egui::Context, open: &mut bool) {
        if !*open {
            return;
        }
        if self.versions.is_none() && self.probe.is_none() {
            let (tx, rx) = channel();
            self.probe = Some(rx);
            let ctx = ctx.clone();
            std::thread::spawn(move || {
                let _ = tx.send(probe_all());
                ctx.request_repaint();
            });
        }
        if let Some(rx) = &self.probe
            && let Ok(map) = rx.try_recv()
        {
            self.versions = Some(map);
            self.probe = None;
        }
        egui::Window::new("Credits")
            .open(open)
            .default_width(620.0)
            .show(ctx, |ui| {
                ui.label(
                    RichText::new(format!(
                        "CT Reconstruction {} — built on the following open-source tools",
                        env!("CARGO_PKG_VERSION")
                    ))
                    .strong(),
                );
                ui.add_space(6.0);
                egui::ScrollArea::vertical()
                    .max_height(480.0)
                    .show(ui, |ui| {
                        self.section(ui, "Python packages (versions from the deployed environments)", PYTHON_TOOLS);
                        self.section(ui, "Algorithms ported natively from", PORTED_TOOLS);
                        self.section(ui, "Rust libraries", RUST_LIBS);
                    });
            });
    }

    fn section(&self, ui: &mut egui::Ui, title: &str, entries: &[Entry]) {
        ui.add_space(4.0);
        ui.label(RichText::new(title).strong().size(13.0));
        ui.separator();
        for entry in entries {
            ui.horizontal(|ui| {
                ui.hyperlink_to(RichText::new(entry.name).strong(), entry.url);
                ui.label(RichText::new(self.version_of(entry)).monospace().size(12.0));
            });
            ui.label(RichText::new(entry.role).weak().size(11.0));
            ui.add_space(4.0);
        }
    }

    fn version_of(&self, entry: &Entry) -> String {
        let Some(key) = entry.key else {
            return String::new();
        };
        if let Some(version) = lock_version(key) {
            return version.to_owned();
        }
        match &self.versions {
            None => "…".to_owned(),
            Some(map) => map.get(key).cloned().unwrap_or_else(|| "—".to_owned()),
        }
    }
}
