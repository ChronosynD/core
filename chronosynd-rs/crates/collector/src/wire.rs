//! On-disk JSONL format for recorded event streams, the wire struct is
//! flat plain-data so it does not bind callers to the typed `EventKind`
//! enum and stays forward-compatible with new kind codes

use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::event::{Event, EventKind};

/// Plain-data on-disk representation of one captured event, kind is stored
/// as the raw kernel code so unknown future codes round-trip cleanly
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WireEvent {
    /// Monotonic nanosecond timestamp the kernel attached to the event
    pub ts_ns: u64,
    /// Process id the event belongs to
    pub pid: u32,
    /// Thread group id, equals `pid` for single-threaded processes
    pub tgid: u32,
    /// Effective user id at the time of the event
    pub uid: u32,
    /// Syscall number that triggered the event
    pub syscall_nr: u32,
    /// Raw kernel kind code, decoded into `EventKind` on read
    pub kind_code: u32,
    /// Process command name as captured by the kernel
    pub comm: String,
    /// First syscall argument decoded as a UTF-8 string
    pub arg0: String,
}

impl WireEvent {
    /// Build a wire-format record from a normalized in-memory event
    pub fn from_event(event: &Event) -> Self {
        Self {
            ts_ns: event.ts_ns,
            pid: event.pid,
            tgid: event.tgid,
            uid: event.uid,
            syscall_nr: event.syscall_nr,
            kind_code: kind_to_code(event.kind),
            comm: event.comm.clone(),
            arg0: event.arg0.clone(),
        }
    }

    /// Reconstruct a typed in-memory event from a wire-format record
    pub fn into_event(self) -> Event {
        Event {
            ts_ns: self.ts_ns,
            pid: self.pid,
            tgid: self.tgid,
            uid: self.uid,
            syscall_nr: self.syscall_nr,
            kind: EventKind::from_code(self.kind_code),
            comm: self.comm,
            arg0: self.arg0,
        }
    }
}

fn kind_to_code(kind: EventKind) -> u32 {
    use chronosynd_bpf::{
        RAW_EVENT_KIND_EXEC, RAW_EVENT_KIND_FILE_OPEN, RAW_EVENT_KIND_NET_CONNECT,
        RAW_EVENT_KIND_OTHER_SYSCALL, RAW_EVENT_KIND_PROCESS_EXIT,
    };
    match kind {
        EventKind::Exec => RAW_EVENT_KIND_EXEC,
        EventKind::FileOpen => RAW_EVENT_KIND_FILE_OPEN,
        EventKind::NetConnect => RAW_EVENT_KIND_NET_CONNECT,
        EventKind::ProcessExit => RAW_EVENT_KIND_PROCESS_EXIT,
        EventKind::OtherSyscall => RAW_EVENT_KIND_OTHER_SYSCALL,
        EventKind::Unknown(code) => code,
    }
}

/// Append-mode writer that serializes events as JSON lines, one per row,
/// flushed on drop so a partial recording is still readable
pub struct EventRecorder {
    writer: BufWriter<File>,
}

impl EventRecorder {
    /// Open `path` for append-mode writing, creating it if missing
    pub fn create(path: &Path) -> Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .with_context(|| format!("opening recording file {}", path.display()))?;
        Ok(Self {
            writer: BufWriter::new(file),
        })
    }

    /// Serialize one event to the underlying file as a JSON line
    pub fn record(&mut self, event: &Event) -> Result<()> {
        let wire = WireEvent::from_event(event);
        serde_json::to_writer(&mut self.writer, &wire)
            .context("serializing event to recording file")?;
        self.writer
            .write_all(b"\n")
            .context("writing newline to recording file")?;
        Ok(())
    }

    /// Force any buffered bytes out to the underlying file
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush().context("flushing recording file")
    }
}

impl Drop for EventRecorder {
    fn drop(&mut self) {
        // best-effort flush, the OS still closes the fd
        let _ = self.writer.flush();
    }
}

/// Read every event from a recording file, the file must contain one JSON
/// object per line, blank lines and trailing whitespace are tolerated
pub fn read_recording(path: &Path) -> Result<Vec<Event>> {
    read_recording_filter(path, |_| true)
}

/// Read every event whose comm matches `predicate`, used to fit a per-process
/// baseline from a multi-process recording without loading everything first.
/// Tolerates a malformed trailing line (the daemon-was-killed-mid-write case)
/// by silently dropping it; a malformed line followed by more lines is real
/// corruption and aborts the read
pub fn read_recording_filter<F>(path: &Path, mut predicate: F) -> Result<Vec<Event>>
where
    F: FnMut(&Event) -> bool,
{
    let file = File::open(path)
        .with_context(|| format!("opening recording file {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut events: Vec<Event> = Vec::new();
    let mut pending_failure: Option<(usize, String)> = None;
    for (line_idx, line_result) in reader.lines().enumerate() {
        let line = line_result
            .with_context(|| format!("reading line {} of {}", line_idx + 1, path.display()))?;
        if let Some((prev_idx, prev_line)) = pending_failure.take() {
            let err = serde_json::from_str::<WireEvent>(&prev_line)
                .err()
                .map(|e| e.to_string())
                .unwrap_or_else(|| "unknown parse error".into());
            return Err(anyhow::anyhow!(
                "mid-file corruption at line {} of {}: {}",
                prev_idx + 1,
                path.display(),
                err,
            ));
        }
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<WireEvent>(&line) {
            Ok(wire) => {
                let event = wire.into_event();
                if predicate(&event) {
                    events.push(event);
                }
            }
            Err(_) => {
                pending_failure = Some((line_idx, line));
            }
        }
    }
    Ok(events)
}
