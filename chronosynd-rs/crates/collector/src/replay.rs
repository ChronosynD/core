//! Replay recorded events through the canonical feature extractor and
//! return one observation row per closed window, the helper turns a real
//! captured trace into the same shape the harness fits a baseline from

use std::path::Path;

use anyhow::{Context, Result};
use chronosynd_features::{default_syscall_vocab, SyscallNgramExtractor};

use crate::wire::read_recording_filter;

/// Replay results scoped to a single process key, the rows are exactly
/// what `chronosynd-baseline` expects for `fit`, one feature vector per row
pub struct ReplayObservations {
    /// Stable identifier the events were filtered by
    pub process_key: String,
    /// Closed-window feature vectors in chronological order
    pub rows: Vec<Vec<f64>>,
    /// Total events read from the recording before filtering
    pub total_events_read: usize,
    /// Events whose comm matched `process_key` and were fed to the extractor
    pub events_for_process: usize,
}

/// Read `recording_path`, replay every event whose `comm` equals `process_key`
/// through a fresh `SyscallNgramExtractor` of `window_size`, and return one
/// row per emitted window plus counters for diagnostics
pub fn extract_observations_from_recording(
    recording_path: &Path,
    process_key: &str,
    window_size: usize,
) -> Result<ReplayObservations> {
    let vocab = default_syscall_vocab();
    let mut extractor = SyscallNgramExtractor::new(vocab, window_size)
        .map_err(|err| anyhow::anyhow!("constructing extractor: {err}"))?;

    let mut total = 0usize;
    let events = read_recording_filter(recording_path, |event| {
        total += 1;
        event.comm == process_key
    })
    .with_context(|| format!("reading recording {}", recording_path.display()))?;

    let mut rows: Vec<Vec<f64>> = Vec::new();
    for event in &events {
        if let Some(emitted) = extractor.accumulate(&event.comm, event.syscall_nr) {
            rows.push(emitted.feature_vector);
        }
    }

    Ok(ReplayObservations {
        process_key: process_key.to_string(),
        rows,
        total_events_read: total,
        events_for_process: events.len(),
    })
}
