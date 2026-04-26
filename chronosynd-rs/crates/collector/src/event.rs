//! Normalized event the collector hands to the feature layer, owns its
//! strings and presents the kind as an enum so downstream code does not
//! need to know the kernel ABI details from `chronosynd_bpf::RawEvent`

use chronosynd_bpf::{
    RAW_EVENT_KIND_EXEC, RAW_EVENT_KIND_FILE_OPEN, RAW_EVENT_KIND_NET_CONNECT,
    RAW_EVENT_KIND_OTHER_SYSCALL, RAW_EVENT_KIND_PROCESS_EXIT,
};

/// A behavioral event captured from the kernel by the collector
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
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
    /// What the kernel observed
    pub kind: EventKind,
    /// Process command name, lossily decoded from kernel UTF-8
    pub comm: String,
    /// First syscall argument, decoded as a UTF-8 string
    pub arg0: String,
}

/// Behavioral event categories the collector reports
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    /// Process replaced its image via execve
    Exec,
    /// Process opened a file
    FileOpen,
    /// Process initiated an outbound network connection
    NetConnect,
    /// Process exited
    ProcessExit,
    /// Any syscall outside the explicitly classified kinds above, the
    /// generic bucket the BPF probe assigns when a syscall in the vocab
    /// does not map to one of the named categories
    OtherSyscall,
    /// Kind code did not match any known category
    Unknown(u32),
}

impl EventKind {
    /// Decode the wire-format kind code into the typed enum
    pub fn from_code(code: u32) -> Self {
        match code {
            RAW_EVENT_KIND_EXEC => Self::Exec,
            RAW_EVENT_KIND_FILE_OPEN => Self::FileOpen,
            RAW_EVENT_KIND_NET_CONNECT => Self::NetConnect,
            RAW_EVENT_KIND_PROCESS_EXIT => Self::ProcessExit,
            RAW_EVENT_KIND_OTHER_SYSCALL => Self::OtherSyscall,
            other => Self::Unknown(other),
        }
    }
}

/// Sanitize a user-controlled string before writing it to a log line.
/// Process names come from the kernel's `comm` field, which an unprivileged
/// caller can set to arbitrary bytes via `prctl(PR_SET_NAME)`. Without
/// this, an attacker could spoof daemon output by embedding newlines and
/// fake `[ALERT]` prefixes in their own process name. Replaces every
/// control character with `?` and caps the result at 64 bytes
pub fn sanitize_for_log(s: &str) -> String {
    const MAX_LEN: usize = 64;
    let mut out = String::with_capacity(s.len().min(MAX_LEN));
    for c in s.chars().take(MAX_LEN) {
        if c.is_control() {
            out.push('?');
        } else {
            out.push(c);
        }
    }
    out
}
