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

The workflow screens themselves are not implemented yet.

## Admin section

A collapsible **🔧 Admin** section sits at the bottom of the setup screen,
protected by a password (only its SHA-256 hash is stored in the source; the
typed password is hashed and compared). Once unlocked it offers a **debug
mode** toggle: when on, the setup screen is prefilled from an HDF5 config
file — instrument and IPTS are selected automatically and the configured mode
button is highlighted.

The active config defaults to `config/config_jean.h5`; a **Browse…** button
in the unlocked admin section selects a different `.h5` file (taking effect
immediately when debug mode is already on). A config file holds three scalar
string datasets: `instrument` (`VENUS`/`MARS`), `ipts` (e.g. `IPTS-36202`)
and `mode` (`White Beam`/`TOF`). Regenerate or write variants with:

```bash
cargo run --release --bin gen_config            # default: VENUS / IPTS-36202 / TOF
cargo run --release --bin gen_config -- other.h5
h5dump config/config_jean.h5                    # inspect
```

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
