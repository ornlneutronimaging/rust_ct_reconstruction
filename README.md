# CT Reconstruction GUI

GUI version of the neutron CT reconstruction workflows developed in
`all_ct_reconstruction_development`, for the VENUS (SNS BL-10) and MARS
(HFIR CG-1D) imaging instruments. Built in Rust with
[egui/eframe](https://github.com/emilk/egui), following the same layout as
`rust_roi_selector`.

## Running

```bash
./launch_ct_reconstruction.sh
```

The script rebuilds the release binary if any source file changed, then starts
the GUI (a graphical session such as ThinLinc is required).

## Setup screen

The application opens on a setup screen that drives the rest of the UI:

1. **Instrument** — VENUS (`/SNS/VENUS`) or MARS (`/HFIR/CG1D`).
2. **Experiment (IPTS)** — the IPTS folders the current user can read are
   discovered in the background and listed newest-first (an experiment counts
   as readable when its directory can be listed, which honors the ACLs).
   An IPTS number can also be typed manually (`IPTS-36967` or `36967`).
3. **Acquisition mode** — two large buttons, **White Beam** and **TOF**;
   clicking one selects the mode (it does not navigate).
4. **Next ➡** — bottom-right, enabled only once instrument, experiment and
   mode are all selected. It hands the session (instrument + IPTS + mode) to
   the corresponding workflow screen.

## TOF workflow

The TOF screen starts with the **detector** (`tpx1 - until July 2025`,
`tpx1 - from August 2025` — the default — or `tpx3`), which decides where
the data is looked for under the experiment:

- sample (projections): `<ipts>/shared/autoreduce/images/<detector>/raw/ct/`
- open beam: `<ipts>/shared/autoreduce/images/<detector>/ob/`

Both sections list the folders found at their root (with a Browse fallback);
selecting one inventories, on a background thread with progress, the images
(full paths) of each subfolder — one per projection angle for a sample, one
per run for open beams — and shows the per-folder and total counts.

Once both are selected, the selection summary (number of projections, sample
/ OB / nexus folders, detector) is logged and a preprocessing pass runs in
the background: empty run folders are rejected, and each run's proton charge
is read from its NeXus file (`<ipts>/nexus/<instrument>_<run>.nxs.h5`,
dataset `entry/proton_charge`, pC → C — same rules as the Python pipeline,
run number parsed from the `Run_<n>` part of the folder name). The sample
and OB proton charges are then drawn on one plot (charge in C vs run number)
so mismatched beam conditions stand out; rejections and missing proton
charges are flagged in the UI and the log.

Note: the until-July-2025 tpx1 layout currently maps to the same folders as
the post-August one (adjust `Detector::images_subdir` in `src/tof.rs` when
its real structure is pinned down).

## White Beam workflow

The White Beam screen starts with the CCD **detector** (`IkonXL` — the
default — `QHY` or `sCMOS`), which decides where the data is looked for:

- sample (projections): `<ipts>/images/<detector>/raw/ct/`
- open beam: `<ipts>/images/<detector>/ob/`

(only the `ikonxl` location is confirmed; `qhy`/`scmos` are provisional —
see `WbDetector::images_subdir` in `src/white_beam.rs`). Unlike TOF, a
white-beam sample folder holds one TIFF per projection, with the run number
and angle in the file name. Folder selection works like the TOF screen; the
per-projection processing steps are next.

## Pre-processing exports

Every section of the pre-processing screen (crop, remove outliers,
normalization, ring artifact removal, remove stripes, rotation, tilt
correction, log conversion) ends with an **📤 Export the images (TIFF)…**
button, enabled once the step was applied. It asks for a folder and creates
a sub-folder named after the step (`crop/`, `remove_outliers/`,
`normalization/`, `ring_artifact_removal/`, `remove_stripes/`, `rotation/`,
`tilt_correction/`, `log_conversion/`) holding the stack as it is at that
point — this step and everything applied before it — as float32 TIFFs, one
per projection in stack order (`0000_<name>.tif`, …). The crop and outlier
steps also write the open beams (and dark currents) into `ob/` and `dc/`
sub-folders, since they transform those too. A `provenance.txt` records the
step, the source file and the stack metadata. An existing step folder is
never reused: a second export goes to `<step>_2/`, then `<step>_3/`, ….

## Sinogram viewer

The ring-artifact removal (bm3dornl) and remove-stripes sections both start
with a **📈 Visualize the sinograms…** button: it writes the current stack
to a work HDF5 next to the loaded file and opens the sibling
`rust_sinogram_viewer` tool on it (any row, band average, contrast window,
hover readout), so the vertical streaks those steps target can be judged
before running them. The work file is deleted when the viewer closes;
nothing comes back into the pipeline. This replaced the in-app *Sinogram*
section.

## Admin section

A collapsible **🔧 Admin** section sits at the bottom of the setup screen,
protected by a password (only its SHA-256 hash is stored in the source; the
typed password is hashed and compared). Once unlocked it offers a **debug
mode** toggle: when on, the setup screen is prefilled from an HDF5 config
file — instrument and IPTS are selected automatically and the configured mode
button is highlighted.

The active config defaults to `config/config_venus_white_beam.h5`
(VENUS / IPTS-36573 / White Beam); `config/config_venus_tof.h5`
(VENUS / IPTS-37118 / TOF) is also generated. A **Browse…** button
in the unlocked admin section selects a different `.h5` file (taking effect
immediately when debug mode is already on). A config file holds three scalar
string datasets: `instrument` (`VENUS`/`MARS`), `ipts` (e.g. `IPTS-36202`)
and `mode` (`White Beam`/`TOF`). Regenerate or write variants with:

```bash
cargo run --release --bin gen_config            # writes both default configs
cargo run --release --bin gen_config -- other.h5
h5dump config/config_venus_white_beam.h5        # inspect
```

## Logging

Most user actions (instrument/IPTS/mode selections, scans, admin unlocks,
debug-config loads, navigation) are appended to
`/SNS/VENUS/shared/log/rust_ct_reconstruction_<user>.log`, following the
`<tool>_<user>.log` naming and line format of the other imaging tools that
log there. Logging is best-effort: the GUI keeps working if the file cannot
be opened.

A **📜 Log** toggle in the top-right corner opens a resizable side panel
showing the tail of the log, with a manual **⟳ Refresh** button and an
**auto** checkbox (on by default, refreshes every 2 s and sticks to the
bottom).

## Development

```bash
cargo test           # non-UI logic (IPTS parsing/validation)
cargo build --release
```

Modules: `instrument.rs` (instrument roots), `ipts.rs` (IPTS discovery and
validation), `session.rs` (mode + session types), `config.rs` (HDF5 debug
configs), `app.rs` (egui application: setup screen → workflow screens),
`main.rs` (thin entry point), `bin/gen_config.rs` (config writer).

HDF5 support uses the `hdf5-metno` crate against the system libhdf5 (1.12.1
on the analysis machines).
