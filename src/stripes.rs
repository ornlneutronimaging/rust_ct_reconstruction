//! "Remove stripes" — the tomopy stripe-removal algorithms of the Python
//! notebook (`RemoveStrips` / `test_algorithms_on_selected_range_of_data`),
//! run through the real tomopy library (the pixi environment of
//! `all_ct_reconstruction`), with the stack handed over as `.npy`.
//!
//! Like the notebook, the algorithms operate on the log data: the linear
//! transmission is converted to `-log(T)`, cleaned, and converted back.
//! The selected algorithms are applied in order.

use crate::combine::{LoadedStack, Projection};
use crate::crop::{read_npy, write_npy};
use crate::normalize::scratch_dir;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

/// The interpreter of the pixi environment that has tomopy installed.
pub const TOMOPY_PYTHON: &str =
    "/SNS/VENUS/shared/software/git/all_ct_reconstruction/.pixi/envs/default/bin/python";

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ParamValue {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: &'static str,
    pub value: ParamValue,
    /// An `Int` where 0 means "auto" (tomopy `None`), left out of the call.
    pub zero_means_auto: bool,
}

impl Param {
    fn int(name: &'static str, value: i64) -> Self {
        Self {
            name,
            value: ParamValue::Int(value),
            zero_means_auto: false,
        }
    }
    fn auto_int(name: &'static str, value: i64) -> Self {
        Self {
            name,
            value: ParamValue::Int(value),
            zero_means_auto: true,
        }
    }
    fn float(name: &'static str, value: f64) -> Self {
        Self {
            name,
            value: ParamValue::Float(value),
            zero_means_auto: false,
        }
    }
    fn boolean(name: &'static str, value: bool) -> Self {
        Self {
            name,
            value: ParamValue::Bool(value),
            zero_means_auto: false,
        }
    }
}

/// One tomopy algorithm the user can enable, with its parameters (defaults
/// from the Python notebook).
#[derive(Clone, Debug)]
pub struct StripeAlgo {
    /// The `tomopy.prep.stripe` function name.
    pub name: &'static str,
    pub help: &'static str,
    pub enabled: bool,
    pub params: Vec<Param>,
}

/// The notebook's algorithm list, in its order; `remove_stripe_sf` is the
/// notebook's default selection.
pub fn default_algorithms() -> Vec<StripeAlgo> {
    vec![
        StripeAlgo {
            name: "remove_stripe_ti",
            help: "Titarenko's approach",
            enabled: false,
            params: vec![Param::int("nblock", 0), Param::float("alpha", 1.5)],
        },
        StripeAlgo {
            name: "remove_stripe_sf",
            help: "smoothing-filter normalization",
            enabled: true,
            params: vec![Param::int("size", 5)],
        },
        StripeAlgo {
            name: "remove_stripe_based_sorting",
            help: "Vo's sorting approach (full and partial stripes)",
            enabled: false,
            params: vec![Param::auto_int("size", 0), Param::int("dim", 1)],
        },
        StripeAlgo {
            name: "remove_stripe_based_fitting",
            help: "Vo's fitting approach",
            enabled: false,
            params: vec![
                Param::int("order", 3),
                Param::int("sigma1", 5),
                Param::int("sigma2", 20),
            ],
        },
        StripeAlgo {
            name: "remove_large_stripe",
            help: "Vo's approach for large stripes",
            enabled: false,
            params: vec![
                Param::float("snr", 3.0),
                Param::int("size", 51),
                Param::float("drop_ratio", 0.1),
                Param::boolean("norm", true),
            ],
        },
        StripeAlgo {
            name: "remove_dead_stripe",
            help: "Vo's approach for unresponsive stripes",
            enabled: false,
            params: vec![
                Param::float("snr", 3.0),
                Param::int("size", 51),
                Param::boolean("norm", true),
            ],
        },
        StripeAlgo {
            name: "remove_stripe_based_interpolation",
            help: "interpolation-based, most stripe types",
            enabled: false,
            params: vec![
                Param::float("snr", 3.0),
                Param::int("size", 31),
                Param::float("drop_ratio", 0.1),
                Param::boolean("norm", true),
            ],
        },
    ]
}

/// `algo(param=value, …) + algo(…)` for the log and the provenance.
pub fn describe(algos: &[StripeAlgo]) -> String {
    algos
        .iter()
        .filter(|a| a.enabled)
        .map(|a| {
            let params: Vec<String> = a
                .params
                .iter()
                .map(|p| {
                    let value = match p.value {
                        ParamValue::Int(0) if p.zero_means_auto => "auto".to_owned(),
                        ParamValue::Int(v) => v.to_string(),
                        ParamValue::Float(v) => format!("{v}"),
                        ParamValue::Bool(v) => v.to_string(),
                    };
                    format!("{}={value}", p.name)
                })
                .collect();
            format!("{}({})", a.name, params.join(", "))
        })
        .collect::<Vec<_>>()
        .join(" + ")
}

const TOMOPY_SCRIPT: &str = r#"
import json
import sys

import numpy as np
from tomopy.prep import stripe

data_file, spec_file, out_file = sys.argv[1:4]
with open(spec_file) as f:
    spec = json.load(f)
data = np.load(data_file)
# Like the notebook: the algorithms run on the log data.
work = -np.log(np.clip(data, 1e-6, None)).astype(np.float32)
for step in spec["steps"]:
    kwargs = dict(step["kwargs"])
    if "sigma1" in kwargs:
        kwargs["sigma"] = (kwargs.pop("sigma1"), kwargs.pop("sigma2"))
    fn = getattr(stripe, step["algo"])
    work = fn(work, ncore=spec["ncore"], **kwargs)
np.save(out_file, np.exp(-work).astype(np.float32))
"#;

fn spec_json(algos: &[StripeAlgo]) -> String {
    let steps: Vec<serde_json::Value> = algos
        .iter()
        .filter(|a| a.enabled)
        .map(|a| {
            let mut kwargs = serde_json::Map::new();
            for p in &a.params {
                match p.value {
                    ParamValue::Int(0) if p.zero_means_auto => {}
                    ParamValue::Int(v) => {
                        kwargs.insert(p.name.to_owned(), v.into());
                    }
                    ParamValue::Float(v) => {
                        kwargs.insert(p.name.to_owned(), v.into());
                    }
                    ParamValue::Bool(v) => {
                        kwargs.insert(p.name.to_owned(), v.into());
                    }
                }
            }
            serde_json::json!({"algo": a.name, "kwargs": kwargs})
        })
        .collect();
    serde_json::json!({
        "ncore": std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8),
        "steps": steps,
    })
    .to_string()
}

/// Run the selected algorithms on a `(n, band_h, w)` volume through tomopy.
fn run_tomopy(
    stack: &LoadedStack,
    volume: &[f32],
    n: usize,
    band_h: usize,
    w: usize,
    algos: &[StripeAlgo],
) -> Result<Vec<f32>, String> {
    let dir = scratch_dir(stack, "stripes")?;
    let data_npy = dir.join("stripes_in.npy");
    let spec_file = dir.join("stripes_spec.json");
    let out_npy = dir.join("stripes_out.npy");
    let script = dir.join("stripes_run.py");
    let cleanup = || {
        for f in [&data_npy, &spec_file, &out_npy, &script] {
            let _ = std::fs::remove_file(f);
        }
        let _ = std::fs::remove_dir(&dir);
    };
    let run = || -> Result<Vec<f32>, String> {
        write_npy(
            &data_npy,
            &[n, band_h, w],
            volume.chunks(band_h * w),
        )?;
        std::fs::write(&spec_file, spec_json(algos))
            .map_err(|e| format!("write {}: {e}", spec_file.display()))?;
        std::fs::write(&script, TOMOPY_SCRIPT)
            .map_err(|e| format!("write {}: {e}", script.display()))?;
        let output = std::process::Command::new(TOMOPY_PYTHON)
            .arg(&script)
            .arg(&data_npy)
            .arg(&spec_file)
            .arg(&out_npy)
            .output()
            .map_err(|e| format!("cannot launch {TOMOPY_PYTHON}: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let tail: Vec<&str> = stderr.trim().lines().rev().take(4).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            return Err(format!(
                "tomopy failed ({}): {}",
                output.status,
                tail.join(" | ")
            ));
        }
        let (shape, values) = read_npy(&out_npy)?;
        if shape != [n, band_h, w] {
            return Err(format!(
                "tomopy returned shape {shape:?}, expected ({n}, {band_h}, {w})"
            ));
        }
        Ok(values)
    };
    let result = run();
    cleanup();
    result
}

/// A rows band `y0..=y1` of every sample projection as an `(n, band_h, w)`
/// volume.
fn band_volume(stack: &LoadedStack, y0: usize, y1: usize) -> (Vec<f32>, usize, usize, usize) {
    let first = &stack.sample[0];
    let (w, n) = (first.width, stack.sample.len());
    let band_h = y1 - y0 + 1;
    let mut volume = Vec::with_capacity(n * band_h * w);
    for p in &stack.sample {
        volume.extend_from_slice(&p.mean[y0 * w..(y1 + 1) * w]);
    }
    (volume, n, band_h, w)
}

/// Test run on a band of rows: resolves to `(before, after)` volumes plus
/// their dimensions.
pub struct StripeTestJob {
    rx: Receiver<Result<(Vec<f32>, Vec<f32>, usize, usize, usize), String>>,
}

impl StripeTestJob {
    pub fn start(stack: Arc<LoadedStack>, y0: usize, y1: usize, algos: Vec<StripeAlgo>) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let result = (|| {
                let (before, n, band_h, w) = band_volume(&stack, y0, y1);
                let after = run_tomopy(&stack, &before, n, band_h, w, &algos)?;
                Ok((before, after, n, band_h, w))
            })();
            let _ = tx.send(result);
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<(Vec<f32>, Vec<f32>, usize, usize, usize), String>> {
        self.rx.try_recv().ok()
    }
}

/// Apply the selected algorithms to the whole sample stack.
pub struct StripeApplyJob {
    rx: Receiver<Result<LoadedStack, String>>,
}

impl StripeApplyJob {
    pub fn start(stack: Arc<LoadedStack>, algos: Vec<StripeAlgo>) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let result = (|| {
                let first = stack
                    .sample
                    .first()
                    .ok_or("no sample projections in the stack")?;
                let h = first.height;
                let (volume, n, band_h, w) = band_volume(&stack, 0, h - 1);
                let cleaned = run_tomopy(&stack, &volume, n, band_h, w, &algos)?;
                let sample: Vec<Projection> = stack
                    .sample
                    .iter()
                    .enumerate()
                    .map(|(i, p)| {
                        let mean = cleaned[i * h * w..(i + 1) * h * w].to_vec();
                        let sum: f64 = mean.iter().map(|v| f64::from(*v)).sum();
                        Projection {
                            name: p.name.clone(),
                            run_number: p.run_number,
                            angle_deg: p.angle_deg,
                            n_images_used: p.n_images_used,
                            height: h,
                            width: w,
                            mean,
                            total_counts: sum * p.n_images_used.max(1) as f64,
                        }
                    })
                    .collect();
                let mut metadata = stack.metadata.clone();
                metadata.retain(|(name, _)| name != "remove_stripes");
                metadata.push(("remove_stripes".to_owned(), describe(&algos)));
                metadata.sort();
                Ok(LoadedStack {
                    path: stack.path.clone(),
                    sample,
                    ob: stack.ob.clone(),
                    metadata,
                    center_of_rotation: stack.center_of_rotation,
                })
            })();
            let _ = tx.send(result);
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<LoadedStack, String>> {
        self.rx.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_notebook() {
        let algos = default_algorithms();
        assert_eq!(algos.len(), 7);
        let enabled: Vec<&str> = algos
            .iter()
            .filter(|a| a.enabled)
            .map(|a| a.name)
            .collect();
        assert_eq!(enabled, ["remove_stripe_sf"]);
    }

    #[test]
    fn spec_and_description() {
        let mut algos = default_algorithms();
        algos[2].enabled = true; // sorting, with size=0 (auto)
        let spec: serde_json::Value = serde_json::from_str(&spec_json(&algos)).unwrap();
        let steps = spec["steps"].as_array().unwrap();
        assert_eq!(steps.len(), 2);
        assert_eq!(steps[0]["algo"], "remove_stripe_sf");
        assert_eq!(steps[0]["kwargs"]["size"], 5);
        // The auto size is omitted, dim stays.
        assert_eq!(steps[1]["algo"], "remove_stripe_based_sorting");
        assert!(steps[1]["kwargs"].get("size").is_none());
        assert_eq!(steps[1]["kwargs"]["dim"], 1);
        let text = describe(&algos);
        assert!(text.contains("remove_stripe_sf(size=5)"));
        assert!(text.contains("size=auto"));
    }
}
