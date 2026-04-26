//! BPF-backed event source gated behind the `bpf` feature on Linux,
//! loads the compiled syscall probe from `chronosynd-bpf` and attaches
//! it to `raw_syscalls/sys_enter`, requires CAP_BPF to load and attach

use std::time::Duration;

use aya::maps::{MapData, RingBuf};
use aya::programs::TracePoint;
use aya::Ebpf;
use chronosynd_bpf::{RawEvent, SYSCALL_PROBE_OBJ};

use crate::event::{Event, EventKind};
use crate::source::{EventSource, EventSourceError};

const RAW_EVENT_SIZE: usize = 112;
const COMM_OFFSET: usize = 32;
const COMM_SIZE: usize = 16;
const ARG0_OFFSET: usize = 48;
const ARG0_SIZE: usize = 64;

/// Default polling interval when the ring buffer is empty
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(5);

/// BPF-backed event source, owns the loaded program and the ring-buffer reader
pub struct BpfEventSource {
    // Field order matters, the ring is a borrow into bpf and must drop first
    ring: RingBuf<MapData>,
    _bpf: Ebpf,
    poll_interval: Duration,
}

impl BpfEventSource {
    /// Load the embedded BPF object, attach the tracepoint, and prepare the ring reader
    pub fn new() -> Result<Self, EventSourceError> {
        if SYSCALL_PROBE_OBJ.is_empty() {
            return Err(EventSourceError::Backend(
                "embedded BPF object is empty, build the bpf crate on Linux with clang and libbpf-dev".into(),
            ));
        }

        let mut bpf = Ebpf::load(SYSCALL_PROBE_OBJ)
            .map_err(|err| EventSourceError::Backend(format!("Ebpf::load: {err}")))?;

        let program: &mut TracePoint = bpf
            .program_mut("handle_syscall")
            .ok_or_else(|| {
                EventSourceError::Backend("BPF object missing program 'handle_syscall'".into())
            })?
            .try_into()
            .map_err(|err| EventSourceError::Backend(format!("program kind mismatch: {err}")))?;
        program
            .load()
            .map_err(|err| EventSourceError::Backend(format!("program.load: {err}")))?;
        program
            .attach("raw_syscalls", "sys_enter")
            .map_err(|err| EventSourceError::Backend(format!("program.attach: {err}")))?;

        let events_map = bpf
            .take_map("events")
            .ok_or_else(|| EventSourceError::Backend("BPF object missing map 'events'".into()))?;
        let ring = RingBuf::try_from(events_map)
            .map_err(|err| EventSourceError::Backend(format!("RingBuf::try_from: {err}")))?;

        Ok(Self {
            ring,
            _bpf: bpf,
            poll_interval: DEFAULT_POLL_INTERVAL,
        })
    }

    /// Override the poll interval used when the ring is empty
    pub fn with_poll_interval(mut self, interval: Duration) -> Self {
        self.poll_interval = interval;
        self
    }
}

impl EventSource for BpfEventSource {
    fn next_event(&mut self) -> Result<Event, EventSourceError> {
        loop {
            if let Some(item) = self.ring.next() {
                let bytes: &[u8] = &item;
                let raw = decode_raw_event(bytes)?;
                return Ok(raw_to_event(&raw));
            }
            std::thread::sleep(self.poll_interval);
        }
    }
}

fn decode_raw_event(bytes: &[u8]) -> Result<RawEvent, EventSourceError> {
    if bytes.len() < RAW_EVENT_SIZE {
        return Err(EventSourceError::Backend(format!(
            "ring item too small, got {} bytes need {RAW_EVENT_SIZE}",
            bytes.len()
        )));
    }
    let mut comm = [0_u8; COMM_SIZE];
    comm.copy_from_slice(&bytes[COMM_OFFSET..COMM_OFFSET + COMM_SIZE]);
    let mut arg0 = [0_u8; ARG0_SIZE];
    arg0.copy_from_slice(&bytes[ARG0_OFFSET..ARG0_OFFSET + ARG0_SIZE]);
    Ok(RawEvent {
        ts_ns: u64::from_ne_bytes(slice_to_array(&bytes[0..8])),
        pid: u32::from_ne_bytes(slice_to_array(&bytes[8..12])),
        tgid: u32::from_ne_bytes(slice_to_array(&bytes[12..16])),
        uid: u32::from_ne_bytes(slice_to_array(&bytes[16..20])),
        syscall_nr: u32::from_ne_bytes(slice_to_array(&bytes[20..24])),
        kind: u32::from_ne_bytes(slice_to_array(&bytes[24..28])),
        _padding: u32::from_ne_bytes(slice_to_array(&bytes[28..32])),
        comm,
        arg0,
    })
}

fn slice_to_array<const N: usize>(slice: &[u8]) -> [u8; N] {
    slice
        .try_into()
        .expect("caller already validated slice length")
}

fn raw_to_event(raw: &RawEvent) -> Event {
    Event {
        ts_ns: raw.ts_ns,
        pid: raw.pid,
        tgid: raw.tgid,
        uid: raw.uid,
        syscall_nr: raw.syscall_nr,
        kind: EventKind::from_code(raw.kind),
        comm: null_terminated_string(&raw.comm),
        arg0: null_terminated_string(&raw.arg0),
    }
}

fn null_terminated_string(bytes: &[u8]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    String::from_utf8_lossy(&bytes[..end]).into_owned()
}
