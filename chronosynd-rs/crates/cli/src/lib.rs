//! Operator CLI for ChronosynD, the entry point `run` takes argv and a
//! writer for stdout, errors go to stderr via `eprintln!` and surface as
//! nonzero exit codes, the library form lets tests drive it deterministically

#![deny(unsafe_op_in_unsafe_fn)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{anyhow, bail, Context, Result};
use chronosynd_baseline::{Baseline, NaiveBaseline, Sediment};
use chronosynd_collector::extract_observations_from_recording;
use chronosynd_storage::{BaselineStore, MaintenanceWindow, StoredBaseline};
use clap::{Parser, Subcommand, ValueEnum};
use ndarray::Array2;

/// Exit code for a successful run
pub const EXIT_OK: i32 = 0;
/// Exit code for a CLI usage error or failed operation
pub const EXIT_ERR: i32 = 1;
/// Exit code for a tamper-evidence failure surfaced by `verify-store`
pub const EXIT_TAMPERED: i32 = 2;

#[derive(Parser, Debug)]
#[command(name = "chronosynd", version, about = "ChronosynD operator CLI")]
struct Cli {
    /// Path to the baseline store, can also be set via CHRONOSYND_STORE
    #[arg(long, env = "CHRONOSYND_STORE", default_value = "chronosynd.db")]
    store: PathBuf,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Inspect persisted baselines
    Baseline {
        #[command(subcommand)]
        action: BaselineAction,
    },
    /// Manage maintenance windows
    Maintenance {
        #[command(subcommand)]
        action: MaintenanceAction,
    },
    /// Walk the audit log and report tampering
    VerifyStore,
    /// Run the collector daemon end-to-end against the configured store
    Run {
        /// Populate the store with a demo baseline before running
        #[arg(long)]
        seed_demo: bool,
        /// Default per-process drift threshold above which an alert is emitted
        #[arg(long, default_value_t = 100.0)]
        default_threshold: f64,
        /// Disjoint window size the feature extractor closes on
        #[arg(long, default_value_t = 16)]
        window_size: usize,
        /// Use the BPF event source, requires `--features bpf`, Linux, and CAP_BPF or root
        #[arg(long)]
        bpf: bool,
        /// Append every observed event to a JSONL file for later replay
        #[arg(long)]
        record: Option<PathBuf>,
    },
    /// Fit a baseline from a CSV of training observations and persist it
    FitBaseline {
        /// Stable identifier for the process this baseline tracks
        process_key: String,
        /// Path to a CSV file, one observation per row, comma-separated f64 columns
        #[arg(long)]
        input: PathBuf,
        /// Estimator to fit
        #[arg(long, value_enum, default_value_t = EstimatorChoice::Sediment)]
        estimator: EstimatorChoice,
        /// Trim fraction for Sediment, ignored when `--estimator naive`
        #[arg(long, default_value_t = 0.3)]
        trim_fraction: f64,
        /// Epsilon added to per-feature std before scoring
        #[arg(long, default_value_t = 1e-6)]
        epsilon: f64,
    },
    /// Fit a baseline from a recorded JSONL event trace and persist it
    FitFromTrace {
        /// Stable identifier for the process this baseline tracks, only events
        /// whose `comm` equals this value contribute to the baseline
        process_key: String,
        /// Path to a JSONL recording produced by `chronosynd run --record`
        #[arg(long)]
        input: PathBuf,
        /// Disjoint window size used by the syscall n-gram extractor
        #[arg(long, default_value_t = 16)]
        window_size: usize,
        /// Estimator to fit on the replayed observations
        #[arg(long, value_enum, default_value_t = EstimatorChoice::Sediment)]
        estimator: EstimatorChoice,
        /// Trim fraction for Sediment, ignored when `--estimator naive`
        #[arg(long, default_value_t = 0.3)]
        trim_fraction: f64,
        /// Epsilon added to per-feature std before scoring
        #[arg(long, default_value_t = 1e-6)]
        epsilon: f64,
    },
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum EstimatorChoice {
    /// Mean and standard deviation per feature, the prior-work reference
    Naive,
    /// Bias-corrected trimmed-mean estimator, the paper's contribution
    Sediment,
}

#[derive(Subcommand, Debug)]
enum BaselineAction {
    /// List every baseline currently in the store
    List,
    /// Show one baseline by its process key
    Show {
        /// Process key recorded when the baseline was fitted
        process_key: String,
    },
}

#[derive(Subcommand, Debug)]
enum MaintenanceAction {
    /// Open a new maintenance window
    Start {
        /// Free-text note for the window
        #[arg(long)]
        note: Option<String>,
    },
    /// Close an open maintenance window by id
    End {
        /// Identifier returned from `maintenance start`
        id: i64,
    },
    /// Show the currently-open window if any
    Current,
}

/// CLI entry point, returns the process exit code
pub fn run<I, T, W>(args: I, mut out: W) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
    W: Write,
{
    let cli = match Cli::try_parse_from(args) {
        Ok(parsed) => parsed,
        Err(err) => {
            eprintln!("{err}");
            return match err.kind() {
                clap::error::ErrorKind::DisplayHelp
                | clap::error::ErrorKind::DisplayVersion => EXIT_OK,
                _ => EXIT_ERR,
            };
        }
    };
    match dispatch(cli, &mut out) {
        Ok(code) => code,
        Err(err) => {
            eprintln!("error: {err:#}");
            EXIT_ERR
        }
    }
}

fn dispatch<W: Write>(cli: Cli, out: &mut W) -> Result<i32> {
    let mut store =
        BaselineStore::open(&cli.store).with_context(|| format!("opening {:?}", cli.store))?;
    match cli.command {
        Command::Baseline { action } => match action {
            BaselineAction::List => baseline_list(&store, out)?,
            BaselineAction::Show { process_key } => baseline_show(&store, &process_key, out)?,
        },
        Command::Maintenance { action } => match action {
            MaintenanceAction::Start { note } => {
                maintenance_start(&mut store, note.as_deref(), out)?
            }
            MaintenanceAction::End { id } => maintenance_end(&mut store, id, out)?,
            MaintenanceAction::Current => maintenance_current(&store, out)?,
        },
        Command::VerifyStore => return verify_store(&store, out),
        Command::Run {
            seed_demo,
            default_threshold,
            window_size,
            bpf,
            record,
        } => {
            // Drop our own handle so the daemon can open the store itself
            drop(store);
            return Ok(run_collector_daemon(
                &cli.store,
                seed_demo,
                default_threshold,
                window_size,
                bpf,
                record.as_deref(),
                out,
            ));
        }
        Command::FitBaseline {
            process_key,
            input,
            estimator,
            trim_fraction,
            epsilon,
        } => fit_baseline(
            &mut store,
            &process_key,
            &input,
            estimator,
            trim_fraction,
            epsilon,
            out,
        )?,
        Command::FitFromTrace {
            process_key,
            input,
            window_size,
            estimator,
            trim_fraction,
            epsilon,
        } => fit_from_trace(
            &mut store,
            &process_key,
            &input,
            window_size,
            estimator,
            trim_fraction,
            epsilon,
            out,
        )?,
    }
    Ok(EXIT_OK)
}

fn baseline_list<W: Write>(store: &BaselineStore, out: &mut W) -> Result<()> {
    let baselines = store.list_baselines()?;
    if baselines.is_empty() {
        writeln!(out, "no baselines recorded")?;
        return Ok(());
    }
    writeln!(out, "{:<40}  {:<22}  {:>10}  {:>14}", "PROCESS_KEY", "ESTIMATOR", "SAMPLES", "FITTED_AT_NS")?;
    for baseline in baselines {
        writeln!(
            out,
            "{:<40}  {:<22}  {:>10}  {:>14}",
            baseline.process_key,
            baseline.estimator_kind,
            baseline.sample_count,
            baseline.fitted_at_ns,
        )?;
    }
    Ok(())
}

fn baseline_show<W: Write>(store: &BaselineStore, key: &str, out: &mut W) -> Result<()> {
    let baseline = store
        .get_baseline(key)?
        .ok_or_else(|| anyhow!("no baseline for process_key {key}"))?;
    write_baseline_detail(&baseline, out)
}

fn write_baseline_detail<W: Write>(baseline: &StoredBaseline, out: &mut W) -> Result<()> {
    writeln!(out, "process_key:    {}", baseline.process_key)?;
    writeln!(out, "estimator:      {}", baseline.estimator_kind)?;
    writeln!(out, "feature_dim:    {}", baseline.feature_dim)?;
    writeln!(out, "sample_count:   {}", baseline.sample_count)?;
    writeln!(out, "fitted_at_ns:   {}", baseline.fitted_at_ns)?;
    writeln!(out, "mean:           {:?}", baseline.mean)?;
    writeln!(out, "std:            {:?}", baseline.std)?;
    Ok(())
}

fn maintenance_start<W: Write>(
    store: &mut BaselineStore,
    note: Option<&str>,
    out: &mut W,
) -> Result<()> {
    let id = store.start_maintenance_window(now_ns(), note)?;
    writeln!(out, "started maintenance window {id}")?;
    Ok(())
}

fn maintenance_end<W: Write>(store: &mut BaselineStore, id: i64, out: &mut W) -> Result<()> {
    store.end_maintenance_window(id, now_ns())?;
    writeln!(out, "ended maintenance window {id}")?;
    Ok(())
}

fn maintenance_current<W: Write>(store: &BaselineStore, out: &mut W) -> Result<()> {
    match store.current_maintenance_window()? {
        None => writeln!(out, "no open maintenance window")?,
        Some(window) => write_window(&window, out)?,
    }
    Ok(())
}

fn write_window<W: Write>(window: &MaintenanceWindow, out: &mut W) -> Result<()> {
    writeln!(out, "id:        {}", window.id)?;
    writeln!(out, "start_ns:  {}", window.start_ns)?;
    match window.end_ns {
        None => writeln!(out, "end_ns:    open")?,
        Some(value) => writeln!(out, "end_ns:    {value}")?,
    }
    if let Some(note) = window.note.as_deref() {
        writeln!(out, "note:      {note}")?;
    }
    Ok(())
}

fn verify_store<W: Write>(store: &BaselineStore, out: &mut W) -> Result<i32> {
    let report = store.verify_audit_log()?;
    writeln!(out, "rows:    {}", report.row_count)?;
    writeln!(out, "last_seq: {}", report.last_seq)?;
    if report.valid {
        writeln!(out, "status:  CLEAN")?;
        return Ok(EXIT_OK);
    }
    writeln!(
        out,
        "status:  TAMPERED at seq {}",
        report
            .broken_at_seq
            .map(|n| n.to_string())
            .unwrap_or_else(|| "unknown".into()),
    )?;
    Ok(EXIT_TAMPERED)
}

fn fit_baseline<W: Write>(
    store: &mut BaselineStore,
    process_key: &str,
    input: &std::path::Path,
    estimator: EstimatorChoice,
    trim_fraction: f64,
    epsilon: f64,
    out: &mut W,
) -> Result<()> {
    let observations = read_observations_csv(input)
        .with_context(|| format!("reading observations from {}", input.display()))?;
    let (n_samples, n_features) = (observations.nrows(), observations.ncols());
    if n_samples == 0 {
        bail!("input contained zero observations");
    }

    let (mean, std, estimator_kind) =
        fit_moments(&observations, estimator, trim_fraction, epsilon)?;

    let stored = StoredBaseline {
        process_key: process_key.to_string(),
        feature_dim: n_features,
        mean,
        std,
        estimator_kind,
        fitted_at_ns: now_ns(),
        sample_count: n_samples as u64,
    };
    store.put_baseline(&stored)?;

    writeln!(
        out,
        "fit baseline for {} from {} samples ({} features) with {}",
        stored.process_key, stored.sample_count, stored.feature_dim, stored.estimator_kind,
    )?;
    Ok(())
}

fn read_observations_csv(path: &std::path::Path) -> Result<Array2<f64>> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(false)
        .flexible(false)
        .from_path(path)?;
    let mut rows: Vec<Vec<f64>> = Vec::new();
    for (line_idx, record) in reader.records().enumerate() {
        let record = record.with_context(|| format!("parsing CSV row {}", line_idx + 1))?;
        let parsed: std::result::Result<Vec<f64>, _> = record
            .iter()
            .map(|cell| cell.trim().parse::<f64>())
            .collect();
        let parsed = parsed
            .with_context(|| format!("parsing floats on row {}", line_idx + 1))?;
        if let Some(first) = rows.first() {
            if parsed.len() != first.len() {
                bail!(
                    "row {} has {} columns, expected {}",
                    line_idx + 1,
                    parsed.len(),
                    first.len()
                );
            }
        }
        rows.push(parsed);
    }
    if rows.is_empty() {
        return Ok(Array2::<f64>::zeros((0, 0)));
    }
    let n_features = rows[0].len();
    if n_features == 0 {
        bail!("input rows have zero columns, expected at least one feature per sample");
    }
    let flat: Vec<f64> = rows.into_iter().flatten().collect();
    let n_samples = flat.len() / n_features;
    Array2::from_shape_vec((n_samples, n_features), flat)
        .context("assembling observations into a 2-D array")
}

fn fit_moments(
    observations: &Array2<f64>,
    estimator: EstimatorChoice,
    trim_fraction: f64,
    epsilon: f64,
) -> Result<(Vec<f64>, Vec<f64>, String)> {
    match estimator {
        EstimatorChoice::Naive => {
            let mut baseline =
                NaiveBaseline::with_epsilon(epsilon).context("constructing NaiveBaseline")?;
            baseline
                .fit(observations.view())
                .context("fitting NaiveBaseline")?;
            let (mean, std) = baseline
                .fitted_moments()
                .ok_or_else(|| anyhow!("NaiveBaseline produced no moments after fit"))?;
            Ok((mean, std, "naive".to_string()))
        }
        EstimatorChoice::Sediment => {
            let mut baseline = Sediment::with_params(trim_fraction, epsilon)
                .context("constructing Sediment")?;
            baseline
                .fit(observations.view())
                .context("fitting Sediment")?;
            let (mean, std) = baseline
                .fitted_moments()
                .ok_or_else(|| anyhow!("Sediment produced no moments after fit"))?;
            let kind = format!("sediment_trim{:02}", (trim_fraction * 100.0).round() as u32);
            Ok((mean, std, kind))
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn fit_from_trace<W: Write>(
    store: &mut BaselineStore,
    process_key: &str,
    input: &std::path::Path,
    window_size: usize,
    estimator: EstimatorChoice,
    trim_fraction: f64,
    epsilon: f64,
    out: &mut W,
) -> Result<()> {
    let replay = extract_observations_from_recording(input, process_key, window_size)
        .with_context(|| format!("replaying recording {}", input.display()))?;

    if replay.rows.is_empty() {
        bail!(
            "no closed windows for process_key {process_key} in {} ({} matching events, \
             {} total), capture longer or lower --window-size",
            input.display(),
            replay.events_for_process,
            replay.total_events_read,
        );
    }

    let n_features = replay.rows[0].len();
    let n_samples = replay.rows.len();
    let flat: Vec<f64> = replay.rows.into_iter().flatten().collect();
    let observations = Array2::from_shape_vec((n_samples, n_features), flat)
        .context("assembling replayed observations into a 2-D array")?;

    let (mean, std, estimator_kind) =
        fit_moments(&observations, estimator, trim_fraction, epsilon)?;

    let stored = StoredBaseline {
        process_key: process_key.to_string(),
        feature_dim: n_features,
        mean,
        std,
        estimator_kind,
        fitted_at_ns: now_ns(),
        sample_count: n_samples as u64,
    };
    store.put_baseline(&stored)?;

    writeln!(
        out,
        "fit baseline for {} from {} samples ({} features) with {} \
         (replayed {} events, {} for this process)",
        stored.process_key,
        stored.sample_count,
        stored.feature_dim,
        stored.estimator_kind,
        replay.total_events_read,
        replay.events_for_process,
    )?;
    Ok(())
}

fn run_collector_daemon<W: Write>(
    store: &std::path::Path,
    seed_demo: bool,
    default_threshold: f64,
    window_size: usize,
    bpf: bool,
    record: Option<&std::path::Path>,
    out: &mut W,
) -> i32 {
    let mut argv: Vec<String> = vec![
        "chronosynd-collector".into(),
        "--store".into(),
        store.to_string_lossy().into_owned(),
        "--default-threshold".into(),
        default_threshold.to_string(),
        "--window-size".into(),
        window_size.to_string(),
    ];
    if seed_demo {
        argv.push("--seed-demo".into());
    }
    if bpf {
        argv.push("--bpf".into());
    }
    if let Some(path) = record {
        argv.push("--record".into());
        argv.push(path.to_string_lossy().into_owned());
    }
    chronosynd_collector::run_daemon(argv, out)
}

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}
