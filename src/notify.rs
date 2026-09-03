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
    /// The slice-range jobs: (first slice, one past the last slice).
    pub jobs: Vec<(usize, usize)>,
    /// How the per-job slice cap was decided (GPU memory model, fixed cap,
    /// check skipped by the user, no split).
    pub split_note: String,
    pub checkpoint: PathBuf,
    /// The checkpoint's provenance: every pre-processing step and parameter.
    pub metadata: Vec<(String, String)>,
    /// The data handed to the reconstruction.
    pub n_projections: usize,
    pub width: usize,
    pub height: usize,
    /// Smallest and largest projection angle (deg), when known.
    pub angle_range: Option<(f64, f64)>,
    /// The analysis machine as probed when the run started.
    pub machine: MachineInfo,
}

/// One GPU as reported by nvidia-smi.
#[derive(Clone, Debug, Default)]
pub struct GpuInfo {
    pub name: String,
    pub total_mib: u64,
    pub used_mib: u64,
}

/// The analysis machine the reconstruction runs on — what to look at when
/// a run fails with an out-of-memory error.
#[derive(Clone, Debug, Default)]
pub struct MachineInfo {
    pub hostname: String,
    /// Every GPU nvidia-smi lists (empty: no nvidia-smi or no GPU).
    pub gpus: Vec<GpuInfo>,
    /// `CUDA_VISIBLE_DEVICES`, when set — jax only sees those.
    pub cuda_visible_devices: Option<String>,
    pub cpus: usize,
    pub mem_total_kib: u64,
    pub mem_available_kib: u64,
}

impl MachineInfo {
    /// Probe nvidia-smi, /proc/meminfo and the environment (a few ms).
    pub fn probe() -> Self {
        let gpus = std::process::Command::new("nvidia-smi")
            .args([
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ])
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| {
                String::from_utf8_lossy(&out.stdout)
                    .lines()
                    .filter_map(|line| {
                        let mut parts = line.split(',').map(str::trim);
                        let name = parts.next()?.to_owned();
                        let total_mib = parts.next()?.parse().ok()?;
                        let used_mib = parts.next().and_then(|v| v.parse().ok()).unwrap_or(0);
                        Some(GpuInfo {
                            name,
                            total_mib,
                            used_mib,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        let meminfo = std::fs::read_to_string("/proc/meminfo").unwrap_or_default();
        let mem_kib = |key: &str| -> u64 {
            meminfo
                .lines()
                .find(|l| l.starts_with(key))
                .and_then(|l| l.split_whitespace().nth(1))
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        };
        Self {
            hostname: hostname(),
            gpus,
            cuda_visible_devices: std::env::var("CUDA_VISIBLE_DEVICES").ok(),
            cpus: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(0),
            mem_total_kib: mem_kib("MemTotal:"),
            mem_available_kib: mem_kib("MemAvailable:"),
        }
    }
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

/// The message body. `output_tail` is what the reconstruction process
/// printed (stdout + stderr); its last lines are quoted when the run failed.
pub fn email_body(
    ctx: &RunContext,
    result: &Result<RunStats, String>,
    output_tail: &str,
) -> String {
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
            const TAIL_LINES: usize = 40;
            let lines: Vec<&str> = output_tail.trim().lines().collect();
            if !lines.is_empty() {
                let skipped = lines.len().saturating_sub(TAIL_LINES);
                b.push_str(&format!(
                    "\nLast {} lines printed by the reconstruction{}:\n",
                    lines.len().min(TAIL_LINES),
                    if skipped > 0 {
                        format!(" ({skipped} earlier lines not shown)")
                    } else {
                        String::new()
                    }
                ));
                for line in &lines[skipped..] {
                    b.push_str(&format!("  | {line}\n"));
                }
            }
        }
    }
    b.push_str(&format!("\nCheckpoint file: {}\n", ctx.checkpoint.display()));

    // The data and how it was cut into jobs — the numbers that decide
    // whether a job fits in memory.
    let px = ctx.width * ctx.height;
    b.push_str("\nData handed to the reconstruction:\n");
    b.push_str(&format!(
        "  Projections:        {}{}\n",
        ctx.n_projections,
        match ctx.angle_range {
            Some((lo, hi)) => format!(" (angles {lo:.3} to {hi:.3} deg)"),
            None => String::new(),
        }
    ));
    b.push_str(&format!(
        "  Image size:         {} x {} px (width x height), {} per image (float32)\n",
        ctx.width,
        ctx.height,
        format_bytes(px as u64 * 4)
    ));
    b.push_str(&format!(
        "  Whole stack:        {} in memory\n",
        format_bytes((ctx.n_projections * px) as u64 * 4)
    ));
    b.push_str(&format!(
        "  Slices requested:   {} to {} ({} slices)\n",
        ctx.slice_from,
        ctx.slice_to,
        ctx.slice_to.saturating_sub(ctx.slice_from) + 1
    ));
    b.push_str(&format!("  Job split:          {}\n", ctx.split_note));
    if !ctx.jobs.is_empty() {
        let largest = ctx
            .jobs
            .iter()
            .map(|(a, z)| z.saturating_sub(*a))
            .max()
            .unwrap_or(0);
        b.push_str(&format!(
            "  Jobs:               {} — largest {largest} slices, i.e. a {} x {largest} x {} \
             sinogram block of {}\n",
            ctx.jobs.len(),
            ctx.n_projections,
            ctx.width,
            format_bytes((ctx.n_projections * largest * ctx.width) as u64 * 4)
        ));
        for (i, (a, z)) in ctx.jobs.iter().enumerate() {
            b.push_str(&format!(
                "    job {}: slices {a} to {} ({} slices)\n",
                i + 1,
                z.saturating_sub(1),
                z.saturating_sub(*a)
            ));
        }
    }

    let m = &ctx.machine;
    b.push_str(&format!("\nMachine ({}):\n", m.hostname));
    if m.gpus.is_empty() {
        b.push_str("  GPUs:               none found (nvidia-smi unavailable or no GPU)\n");
    } else {
        b.push_str(&format!("  GPUs:               {}\n", m.gpus.len()));
        for (i, g) in m.gpus.iter().enumerate() {
            b.push_str(&format!(
                "    GPU {i}: {} — {:.1} GB total, {:.1} GB in use when the run started\n",
                g.name,
                g.total_mib as f64 / 1024.0,
                g.used_mib as f64 / 1024.0
            ));
        }
    }
    b.push_str(&format!(
        "  CUDA_VISIBLE_DEVICES: {}\n",
        m.cuda_visible_devices
            .as_deref()
            .unwrap_or("not set (all GPUs visible)")
    ));
    b.push_str(&format!("  CPUs:               {}\n", m.cpus));
    b.push_str(&format!(
        "  RAM:                {:.0} GB total, {:.0} GB available when the run started\n",
        m.mem_total_kib as f64 / 1024.0 / 1024.0,
        m.mem_available_kib as f64 / 1024.0 / 1024.0
    ));

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
            jobs: vec![(120, 250), (240, 370), (360, 481)],
            split_note: "fixed cap of 50 slices per job".to_owned(),
            checkpoint: PathBuf::from("/SNS/VENUS/IPTS-1/shared/sample_step_reconstruction.h5"),
            metadata: vec![("normalization".to_owned(), "ob + pc".to_owned())],
            n_projections: 361,
            width: 2048,
            height: 512,
            angle_range: None,
            machine: MachineInfo::default(),
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
        let body = email_body(&context(), &Ok(stats()), "");
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
        let failed = email_body(&context(), &Err("out of memory".to_owned()), "");
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

    fn sample_context() -> RunContext {
        RunContext {
            algo_label: "mbirjax".to_owned(),
            params_json: r#"{"sharpness": 0.0}"#.to_owned(),
            slice_from: 0,
            slice_to: 1023,
            n_jobs: 3,
            jobs: vec![(0, 350), (340, 690), (680, 1024)],
            split_note: "GPU memory check SKIPPED by the user: at most 500 slices per job"
                .to_owned(),
            checkpoint: PathBuf::from("/tmp/ct_step_pre_processing.h5"),
            metadata: vec![("rebin".to_owned(), "2x2 (block mean), 4096x2048 -> 2048x1024".to_owned())],
            n_projections: 500,
            width: 2048,
            height: 1024,
            angle_range: Some((0.0, 359.5)),
            machine: MachineInfo {
                hostname: "bl10-analysis1".to_owned(),
                gpus: vec![
                    GpuInfo { name: "NVIDIA A100-PCIE-40GB".to_owned(), total_mib: 40960, used_mib: 1049 },
                    GpuInfo { name: "NVIDIA A100-PCIE-40GB".to_owned(), total_mib: 40960, used_mib: 1 },
                ],
                cuda_visible_devices: None,
                cpus: 112,
                mem_total_kib: 1_583_385_848,
                mem_available_kib: 1_033_412_740,
            },
        }
    }

    #[test]
    fn failure_email_carries_the_diagnostics() {
        let ctx = sample_context();
        let output: String = (1..=50).map(|i| format!("line {i}\n")).collect::<String>()
            + "jaxlib.xla_extension.XlaRuntimeError: RESOURCE_EXHAUSTED: Out of memory\n";
        let body = email_body(&ctx, &Err("reconstruction failed (exit status: 1)".to_owned()), &output);
        println!("{body}");
        for needle in [
            "Projections:        500 (angles 0.000 to 359.500 deg)",
            "Image size:         2048 x 1024 px",
            "Whole stack:        4.19 GB in memory",
            "GPU memory check SKIPPED",
            "Jobs:               3 — largest 350 slices",
            "job 1: slices 0 to 349 (350 slices)",
            "GPUs:               2",
            "GPU 0: NVIDIA A100-PCIE-40GB — 40.0 GB total, 1.0 GB in use",
            "CUDA_VISIBLE_DEVICES: not set",
            "CPUs:               112",
            "RAM:                1510 GB total, 986 GB available",
            "Last 40 lines printed by the reconstruction (11 earlier lines not shown)",
            "RESOURCE_EXHAUSTED: Out of memory",
            "rebin: 2x2 (block mean)",
        ] {
            assert!(body.contains(needle), "missing {needle:?} in:\n{body}");
        }
        assert!(!body.contains("| line 11\n"), "line 11 should be cut off");
        assert!(body.contains("| line 12\n"));
        // A success email carries the same data / machine sections, no tail.
        let ok = email_body(&ctx, &Ok(RunStats {
            output_folder: PathBuf::from("/tmp/out"),
            n_files: 1024,
            file_bytes: (1, 2),
            total_bytes: 3,
            total_seconds: 61.0,
            job_times: vec![],
        }), &output);
        assert!(ok.contains("Machine (bl10-analysis1)"));
        assert!(!ok.contains("lines printed by the reconstruction"));
    }
}

