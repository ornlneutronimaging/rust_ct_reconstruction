//! Launch the btop system monitor in an external terminal window.

use std::path::PathBuf;

/// Fallback locations when `btop` is not on the PATH of the app process.
const BTOP_FALLBACKS: &[&str] = &[
    "/SNS/users/j35/bin/btop",
    "/usr/bin/btop",
    "/usr/local/bin/btop",
];

/// Find an executable on the PATH.
fn on_path(name: &str) -> Option<PathBuf> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|p| p.is_file())
}

/// Find the btop binary: PATH first, then the known shared locations.
fn find_btop() -> Option<PathBuf> {
    on_path("btop").or_else(|| {
        BTOP_FALLBACKS
            .iter()
            .map(PathBuf::from)
            .find(|p| p.is_file())
    })
}

/// Open btop in its own terminal window (the first terminal emulator found).
/// The window lives independently of the app; closing either does not affect
/// the other.
pub fn launch() -> Result<(), String> {
    let btop = find_btop().ok_or("btop was not found on this machine")?;
    // Terminal emulators and the flags that make them run a command.
    let candidates: &[(&str, &[&str])] = &[
        ("gnome-terminal", &["--title=btop", "--geometry=200x50", "--"]),
        ("xfce4-terminal", &["--title=btop", "--geometry=200x50", "-x"]),
        (
            "xterm",
            &[
                "-title",
                "btop",
                "-geometry",
                "200x50",
                "-fa",
                "DejaVu Sans Mono",
                "-fs",
                "10",
                "-e",
            ],
        ),
    ];
    for (term, args) in candidates {
        if on_path(term).is_none() {
            continue;
        }
        match std::process::Command::new(term)
            .args(*args)
            .arg(&btop)
            .spawn()
        {
            Ok(mut child) => {
                // Reap the launcher process in the background so it never
                // lingers as a zombie (gnome-terminal exits immediately).
                std::thread::spawn(move || {
                    let _ = child.wait();
                });
                return Ok(());
            }
            Err(_) => continue,
        }
    }
    Err("no terminal emulator found (tried gnome-terminal, xfce4-terminal, xterm)".to_owned())
}
