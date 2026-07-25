//! End-of-reconstruction email notification: duration, output folder and
//! every parameter used, sent through the analysis machine's local
//! `sendmail` — no external service or credentials needed.

use crate::recon_run::{RunStats, format_bytes, format_duration};
use std::io::Write;
use std::path::{Path, PathBuf};

const SENDMAIL: &str = "/usr/sbin/sendmail";

/// The user's ORNL address, prefilled in the notification settings.
pub fn default_email() -> String {
    format!("{}@ornl.gov", crate::logger::user_id())
}

/// Minimal sanity check before handing an address to sendmail.
pub fn valid_email(address: &str) -> bool {
    let a = address.trim();
    a.contains('@') && !a.starts_with('@') && !a.ends_with('@') && !a.contains(char::is_whitespace)
}

/// Headers must be plain ASCII (RFC 2047 encoding is the alternative);
/// anything else risks the message being silently discarded by the mail
/// filtering, so non-ASCII characters are replaced by spaces.
fn ascii_header(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_ascii() && !c.is_ascii_control() {
                c
            } else {
                ' '
            }
        })
        .collect()
}

/// Send one message through the local MTA. Returns once sendmail has
/// accepted (queued) it.
pub fn send_mail(to: &str, subject: &str, body: &str) -> Result<(), String> {
    let mut child = std::process::Command::new(SENDMAIL)
        .args(["-oi", "-t"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
        .map_err(|e| format!("cannot launch {SENDMAIL}: {e}"))?;
    // The MIME headers matter: the body is UTF-8 (the provenance can carry
    // degree signs and the like) and without a declared charset the mail
    // filtering may drop the message.
    let message = format!(
        "To: {to}\nFrom: {from}\nSubject: {subject}\nMIME-Version: 1.0\n\
         Content-Type: text/plain; charset=UTF-8\nContent-Transfer-Encoding: 8bit\n\n{body}\n",
        to = ascii_header(to),
        from = default_email(),
        subject = ascii_header(subject),
    );
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(message.as_bytes())
        .map_err(|e| format!("cannot write to sendmail: {e}"))?;
    let status = child
        .wait()
        .map_err(|e| format!("waiting for sendmail: {e}"))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("sendmail exited with {status}"))
    }
}

/// Everything the notification texts are built from, captured when the run
/// starts (the run spec itself is consumed by the job).
#[derive(Clone)]
pub struct RunContext {
    pub algo_label: String,
    pub params_json: String,
    pub slice_from: usize,
    pub slice_to: usize,
    pub n_jobs: usize,
    pub checkpoint: PathBuf,
    /// The checkpoint's provenance: every pre-processing step and parameter.
    pub metadata: Vec<(String, String)>,
}

/// ASCII only: a non-ASCII subject (an em dash, say) must be RFC
/// 2047-encoded to be legal, and the mail filtering drops it otherwise.
pub fn email_subject(ctx: &RunContext, result: &Result<RunStats, String>) -> String {
    match result {
        Ok(stats) => format!(
            "CT reconstruction done - {} in {}",
            ctx.algo_label,
            format_duration(stats.total_seconds)
        ),
        Err(_) => format!("CT reconstruction failed - {}", ctx.algo_label),
    }
}

pub fn email_body(ctx: &RunContext, result: &Result<RunStats, String>) -> String {
    let mut b = String::new();
    match result {
        Ok(stats) => {
            b.push_str("Your CT reconstruction is done.\n\n");
            b.push_str(&format!("Algorithm:      {}\n", ctx.algo_label));
            b.push_str(&format!(
                "Total time:     {}\n",
                format_duration(stats.total_seconds)
            ));
            b.push_str(&format!(
                "Slices:         {} to {} ({} job{})\n",
                ctx.slice_from,
                ctx.slice_to,
                ctx.n_jobs,
                if ctx.n_jobs == 1 { "" } else { "s" }
            ));
            b.push_str(&format!(
                "Output folder:  {}\n",
                stats.output_folder.display()
            ));
            b.push_str(&format!(
                "Files:          {} image_*.tiff, {} total ({} to {} each)\n",
                stats.n_files,
                format_bytes(stats.total_bytes),
                format_bytes(stats.file_bytes.0),
                format_bytes(stats.file_bytes.1)
            ));
            if !stats.job_times.is_empty() {
                b.push_str("\nPer-job timing:\n");
                for (from, to, seconds) in &stats.job_times {
                    b.push_str(&format!(
                        "  slices {from} to {to}: {}\n",
                        format_duration(*seconds)
                    ));
                }
            }
        }
        Err(e) => {
            b.push_str("Your CT reconstruction FAILED.\n\n");
            b.push_str(&format!("Algorithm:      {}\n", ctx.algo_label));
            b.push_str(&format!(
                "Slices:         {} to {}\n",
                ctx.slice_from, ctx.slice_to
            ));
            b.push_str(&format!("Error:          {e}\n"));
        }
    }
    b.push_str(&format!("\nCheckpoint file: {}\n", ctx.checkpoint.display()));

    b.push_str(&format!("\nParameters used ({}):\n", ctx.algo_label));
    match serde_json::from_str::<serde_json::Value>(&ctx.params_json) {
        Ok(serde_json::Value::Object(map)) => {
            for (name, value) in &map {
                b.push_str(&format!("  {name}: {value}\n"));
            }
        }
        _ => b.push_str(&format!("  {}\n", ctx.params_json)),
    }

    if !ctx.metadata.is_empty() {
        b.push_str("\nProvenance (every step recorded in the checkpoint):\n");
        for (name, value) in &ctx.metadata {
            b.push_str(&format!("  {name}: {value}\n"));
        }
    }

    b.push_str(&format!(
        "\n--\nSent by rust_ct_reconstruction on {} (user {})\n",
        hostname(),
        crate::logger::user_id()
    ));
    b
}

fn hostname() -> String {
    std::fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_owned())
        .unwrap_or_else(|_| "unknown host".to_owned())
}

// ---------------------------------------------------------------------------
// Saved notification settings (~/.config/rust_ct_reconstruction/notify.json)
// so the address and phone number survive restarts.

#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Settings {
    pub email_enabled: bool,
    /// Optional replacement address; empty = the default `<user>@ornl.gov`.
    pub email: String,
}

impl Settings {
    /// Where the notification actually goes: the custom address when one
    /// was typed, the user's ORNL address otherwise.
    pub fn recipient(&self) -> String {
        let custom = self.email.trim();
        if custom.is_empty() {
            default_email()
        } else {
            custom.to_owned()
        }
    }
}

fn settings_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(
        PathBuf::from(home)
            .join(".config")
            .join("rust_ct_reconstruction")
            .join("notify.json"),
    )
}

pub fn load_settings() -> Settings {
    settings_path()
        .map(|p| read_settings(&p))
        .unwrap_or_default()
}

pub fn save_settings(settings: &Settings) {
    if let Some(path) = settings_path()
        && let Err(e) = write_settings(&path, settings)
    {
        crate::logger::error(format!("cannot save the notification settings: {e}"));
    }
}

fn read_settings(path: &Path) -> Settings {
    let mut settings = Settings::default();
    let Ok(text) = std::fs::read_to_string(path) else {
        return settings;
    };
    let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) else {
        return settings;
    };
    if let Some(v) = json.get("email_enabled").and_then(|v| v.as_bool()) {
        settings.email_enabled = v;
    }
    if let Some(v) = json.get("email").and_then(|v| v.as_str())
        // Settings written before the address became an override stored the
        // default explicitly; treat it as "use the default".
        && v != default_email()
    {
        settings.email = v.to_owned();
    }
    settings
}

fn write_settings(path: &Path, settings: &Settings) -> Result<(), String> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    }
    let json = serde_json::json!({
        "email_enabled": settings.email_enabled,
        "email": settings.email,
    });
    let text = serde_json::to_string_pretty(&json).map_err(|e| e.to_string())?;
    std::fs::write(path, text).map_err(|e| format!("cannot write {}: {e}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn context() -> RunContext {
        RunContext {
            algo_label: "svmbir".to_owned(),
            params_json: "{\"sharpness\":0.5,\"snr_db\":30.0}".to_owned(),
            slice_from: 120,
            slice_to: 480,
            n_jobs: 3,
            checkpoint: PathBuf::from("/SNS/VENUS/IPTS-1/shared/sample_step_reconstruction.h5"),
            metadata: vec![("normalization".to_owned(), "ob + pc".to_owned())],
        }
    }

    fn stats() -> RunStats {
        RunStats {
            output_folder: PathBuf::from("/SNS/VENUS/IPTS-1/shared/recon"),
            n_files: 361,
            file_bytes: (3_300_000, 3_400_000),
            total_bytes: 1_200_000_000,
            total_seconds: 3725.0,
            job_times: vec![(120, 240, 1200.0)],
        }
    }

    #[test]
    fn recipient_falls_back_to_the_default_address() {
        let mut settings = Settings::default();
        assert_eq!(settings.recipient(), default_email());
        settings.email = "  ".to_owned();
        assert_eq!(settings.recipient(), default_email());
        settings.email = "someone@example.com".to_owned();
        assert_eq!(settings.recipient(), "someone@example.com");
    }

    #[test]
    fn subjects_are_pure_ascii() {
        let ctx = context();
        assert!(email_subject(&ctx, &Ok(stats())).is_ascii());
        assert!(email_subject(&ctx, &Err("boom".to_owned())).is_ascii());
        assert_eq!(ascii_header("done — svmbir"), "done   svmbir");
        assert_eq!(ascii_header("plain ascii"), "plain ascii");
    }

    #[test]
    fn email_validation_rejects_junk() {
        assert!(valid_email("j35@ornl.gov"));
        assert!(!valid_email("j35"));
        assert!(!valid_email("@ornl.gov"));
        assert!(!valid_email("j35@"));
        assert!(!valid_email("j 35@ornl.gov"));
    }

    #[test]
    fn email_body_covers_stats_params_and_provenance() {
        let body = email_body(&context(), &Ok(stats()));
        for needle in [
            "1h 2m 5s",
            "/SNS/VENUS/IPTS-1/shared/recon",
            "361 image_*.tiff",
            "sharpness",
            "normalization",
            "slices 120 to 240",
        ] {
            assert!(body.contains(needle), "missing {needle:?} in:\n{body}");
        }
        let failed = email_body(&context(), &Err("out of memory".to_owned()));
        assert!(failed.contains("FAILED"));
        assert!(failed.contains("out of memory"));
    }

    #[test]
    fn settings_round_trip() {
        let dir = std::env::temp_dir().join(format!("notify_test_{}", std::process::id()));
        let file = dir.join("notify.json");
        let settings = Settings {
            email_enabled: true,
            email: "someone@example.com".to_owned(),
        };
        write_settings(&file, &settings).unwrap();
        assert_eq!(read_settings(&file), settings);
        // A stored default address (written by the previous version, which
        // prefilled it) reads back as "use the default".
        let old = Settings {
            email_enabled: true,
            email: default_email(),
        };
        write_settings(&file, &old).unwrap();
        assert_eq!(read_settings(&file).email, "");
        std::fs::remove_dir_all(&dir).ok();
    }
}
