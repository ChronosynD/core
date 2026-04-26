//! Daemon main loop wiring the full runtime path together, event source
//! to feature extractor to scorer (warmed from storage) to alert, the
//! synthetic event stream is the demo path used when --bpf is not set

use std::ffi::OsString;
use std::io::Write;
use std::path::PathBuf;

use anyhow::{Context, Result};
use chronosynd_features::{default_syscall_vocab, EmittedFeatures, SyscallNgramExtractor};
use chronosynd_scoring::{warm_from_store, Scorer, ScoringRequest, ScoringResult};
use chronosynd_storage::{BaselineStore, StoredBaseline};
use clap::Parser;

use crate::event::{sanitize_for_log, Event, EventKind};
use crate::wire::EventRecorder;

/// Successful run, the synthetic stream drained without error
pub const EXIT_OK: i32 = 0;

/// Run failed before completing, see stderr
pub const EXIT_ERR: i32 = 1;

const DEMO_TS_NS: u64 = 1_700_000_000_000;
const DEFAULT_WINDOW_SIZE: usize = 16;
const NGINX_PROCESS_KEY: &str = "nginx";

#[derive(Parser, Debug)]
#[command(
    name = "chronosynd-collector",
    version,
    about = "ChronosynD collector daemon"
)]
struct Args {
    /// Path to the baseline store, also reads CHRONOSYND_STORE
    #[arg(long, env = "CHRONOSYND_STORE", default_value = "chronosynd.db")]
    store: PathBuf,

    /// Default per-process drift threshold above which an alert is emitted
    #[arg(long, default_value_t = 100.0)]
    default_threshold: f64,

    /// Populate the store with a demo baseline before running
    #[arg(long)]
    seed_demo: bool,

    /// Disjoint window size the feature extractor closes on
    #[arg(long, default_value_t = DEFAULT_WINDOW_SIZE)]
    window_size: usize,

    /// Use the BPF event source instead of the synthetic stream, requires
    /// the binary built with `--features bpf`, Linux, and CAP_BPF or root
    #[arg(long)]
    bpf: bool,

    /// Append every observed event to this JSONL file for later replay,
    /// the file is created if missing and appended to if it already exists
    #[arg(long)]
    record: Option<PathBuf>,
}

/// Daemon entry point, returns the process exit code
pub fn run_daemon<I, T, W>(args: I, mut out: W) -> i32
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
    W: Write,
{
    let parsed = match Args::try_parse_from(args) {
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
    match try_run(&parsed, &mut out) {
        Ok(()) => EXIT_OK,
        Err(err) => {
            eprintln!("error: {err:#}");
            EXIT_ERR
        }
    }
}

fn try_run<W: Write>(args: &Args, out: &mut W) -> Result<()> {
    let vocab = default_syscall_vocab();
    let feature_dim = vocab.len() + 1;

    let mut store = BaselineStore::open(&args.store)
        .with_context(|| format!("opening {:?}", args.store))?;
    if args.seed_demo {
        seed_demo_baseline(&mut store, feature_dim, &vocab)?;
    }

    let mut scorer = Scorer::new(1e-6, args.default_threshold)
        .map_err(|err| anyhow::anyhow!("constructing scorer: {err}"))?;
    let warmed = warm_from_store(&mut scorer, &store)?;
    writeln!(
        out,
        "warmed {warmed} baselines from {}",
        args.store.display()
    )?;

    let mut extractor = SyscallNgramExtractor::new(vocab, args.window_size)
        .map_err(|err| anyhow::anyhow!("constructing extractor: {err}"))?;

    let mut recorder = match args.record.as_deref() {
        Some(path) => {
            let recorder = EventRecorder::create(path)
                .with_context(|| format!("opening recording file {}", path.display()))?;
            writeln!(out, "recording events to {}", path.display())?;
            Some(recorder)
        }
        None => None,
    };

    if args.bpf {
        run_bpf_loop(&scorer, &mut extractor, recorder.as_mut(), out)
    } else {
        for event in synthetic_event_stream() {
            if let Some(rec) = recorder.as_mut() {
                rec.record(&event)?;
            }
            if let Some(emitted) = extractor.accumulate(&event.comm, event.syscall_nr) {
                score_emission(&scorer, &emitted, out)?;
            }
        }
        if let Some(mut rec) = recorder {
            rec.flush()?;
        }
        Ok(())
    }
}

#[cfg(all(target_os = "linux", feature = "bpf"))]
fn run_bpf_loop<W: Write>(
    scorer: &Scorer,
    extractor: &mut SyscallNgramExtractor,
    mut recorder: Option<&mut EventRecorder>,
    out: &mut W,
) -> Result<()> {
    use crate::source::EventSource;
    use crate::sources::bpf::BpfEventSource;

    let mut source = BpfEventSource::new()
        .map_err(|err| anyhow::anyhow!("opening BPF event source: {err}"))?;
    writeln!(
        out,
        "BPF event source attached, polling for syscall events, Ctrl-C to stop"
    )?;
    loop {
        match source.next_event() {
            Ok(event) => {
                if let Some(rec) = recorder.as_deref_mut() {
                    rec.record(&event)?;
                }
                if let Some(emitted) = extractor.accumulate(&event.comm, event.syscall_nr) {
                    score_emission(scorer, &emitted, out)?;
                }
            }
            Err(err) => {
                writeln!(out, "[err] BPF source: {err}")?;
                if let Some(rec) = recorder.as_deref_mut() {
                    rec.flush()?;
                }
                return Ok(());
            }
        }
    }
}

#[cfg(not(all(target_os = "linux", feature = "bpf")))]
fn run_bpf_loop<W: Write>(
    _scorer: &Scorer,
    _extractor: &mut SyscallNgramExtractor,
    _recorder: Option<&mut EventRecorder>,
    _out: &mut W,
) -> Result<()> {
    Err(anyhow::anyhow!(
        "--bpf was passed but this binary was built without `--features bpf` on a non-Linux host, \
         rebuild with `cargo build --features bpf` on Linux"
    ))
}

fn score_emission<W: Write>(
    scorer: &Scorer,
    emitted: &EmittedFeatures,
    out: &mut W,
) -> Result<()> {
    let request = ScoringRequest {
        process_key: &emitted.process_key,
        observation: &emitted.feature_vector,
    };
    let safe_key = sanitize_for_log(&emitted.process_key);
    match scorer.score(&request) {
        Ok(ScoringResult {
            score,
            threshold,
            alert,
        }) => {
            let label = if alert.is_some() { "ALERT" } else { "ok" };
            writeln!(
                out,
                "[{label}] {safe_key} score={score:.3} threshold={threshold:.3}",
            )?;
        }
        Err(err) => {
            writeln!(out, "[err] {safe_key}: {err}")?;
        }
    }
    Ok(())
}

/// Crafted clean-nginx baseline whose mean matches the histogram a window
/// of the synthetic clean stream produces, std is small but non-zero so
/// the anomalous window's score lands well above the default threshold
fn seed_demo_baseline(
    store: &mut BaselineStore,
    feature_dim: usize,
    vocab: &[u32],
) -> Result<()> {
    let mut mean = vec![0.0; feature_dim];
    set_vocab_value(&mut mean, vocab, 0, 7.0 / 16.0);
    set_vocab_value(&mut mean, vocab, 1, 5.0 / 16.0);
    set_vocab_value(&mut mean, vocab, 3, 2.0 / 16.0);
    set_vocab_value(&mut mean, vocab, 257, 2.0 / 16.0);
    let std = vec![0.02_f64; feature_dim];

    store
        .put_baseline(&StoredBaseline {
            process_key: NGINX_PROCESS_KEY.into(),
            feature_dim,
            mean,
            std,
            estimator_kind: "demo_naive".into(),
            fitted_at_ns: DEMO_TS_NS,
            sample_count: 64,
        })
        .with_context(|| format!("seeding {NGINX_PROCESS_KEY}"))?;
    Ok(())
}

fn set_vocab_value(out: &mut [f64], vocab: &[u32], syscall_nr: u32, value: f64) {
    if let Some(idx) = vocab.iter().position(|nr| *nr == syscall_nr) {
        out[idx] = value;
    }
}

/// Synthetic deterministic event stream, two windows of clean nginx
/// behavior followed by one window of adversarial behavior dominated by
/// exec and connect calls a legitimate nginx never makes
fn synthetic_event_stream() -> Vec<Event> {
    let clean_pattern: [u32; 16] = [0, 0, 1, 0, 1, 3, 0, 1, 257, 1, 0, 3, 0, 1, 257, 0];
    let anomalous_pattern: [u32; 16] = [
        59, 42, 59, 56, 42, 59, 56, 42, 59, 42, 56, 59, 42, 56, 59, 42,
    ];

    let mut events = Vec::new();
    let mut ts = 1_000_000_000_u64;

    for _ in 0..2 {
        for &syscall_nr in &clean_pattern {
            events.push(make_event(ts, syscall_nr, EventKind::FileOpen));
            ts += 1_000_000;
        }
    }
    for &syscall_nr in &anomalous_pattern {
        events.push(make_event(ts, syscall_nr, EventKind::Exec));
        ts += 1_000_000;
    }
    events
}

fn make_event(ts_ns: u64, syscall_nr: u32, kind: EventKind) -> Event {
    Event {
        ts_ns,
        pid: 4242,
        tgid: 4242,
        uid: 33,
        syscall_nr,
        kind,
        comm: NGINX_PROCESS_KEY.into(),
        arg0: String::new(),
    }
}
