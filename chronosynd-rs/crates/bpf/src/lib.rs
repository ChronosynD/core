//! In-kernel BPF programs for ChronosynD compiled from restricted C, the
//! crate's `build.rs` produces ELF objects on Linux hosts with the
//! toolchain present and embeds them, see `bpf/README.md` for prerequisites

#![deny(unsafe_op_in_unsafe_fn)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

/// Wrapper that forces 8-byte alignment on the contained bytes via a
/// leading zero-sized `[Align; 0]` field, needed because Aya's ELF parser
/// reads `FileHeader64` via direct casts that demand 8-byte alignment
#[cfg(chronosynd_bpf_compiled)]
#[repr(C)]
struct AlignedAs<Align, Bytes: ?Sized> {
    _align: [Align; 0],
    bytes: Bytes,
}

#[cfg(chronosynd_bpf_compiled)]
static SYSCALL_PROBE_DATA: &AlignedAs<u64, [u8]> = &AlignedAs {
    _align: [],
    bytes: *include_bytes!(concat!(env!("OUT_DIR"), "/syscall_probe.bpf.o")),
};

/// Compiled bytes of `bpf/syscall_probe.bpf.c`, populated when the build
/// pipeline produces an object on a Linux host and empty otherwise, the
/// returned slice is 8-byte aligned so Aya's ELF parser is happy
#[cfg(chronosynd_bpf_compiled)]
pub const SYSCALL_PROBE_OBJ: &[u8] = &SYSCALL_PROBE_DATA.bytes;

/// Empty placeholder used when the BPF build pipeline did not run on this host
#[cfg(not(chronosynd_bpf_compiled))]
pub const SYSCALL_PROBE_OBJ: &[u8] = &[];

/// Whether the current build embedded a real compiled BPF object
pub const fn syscall_probe_available() -> bool {
    !SYSCALL_PROBE_OBJ.is_empty()
}

/// Wire-format event the kernel writes into the ring buffer, kept in lock
/// step with `bpf/event.h`, sizes and field order are part of the ABI
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RawEvent {
    /// Monotonic nanosecond timestamp from the kernel
    pub ts_ns: u64,
    /// Process id
    pub pid: u32,
    /// Thread group id
    pub tgid: u32,
    /// Effective user id at the time of the event
    pub uid: u32,
    /// Syscall number that triggered the event
    pub syscall_nr: u32,
    /// One of the `RAW_EVENT_KIND_*` constants
    pub kind: u32,
    /// Reserved for alignment, always zero
    pub _padding: u32,
    /// Process command name, null-terminated, max 16 bytes
    pub comm: [u8; 16],
    /// First syscall argument captured as a string, max 64 bytes
    pub arg0: [u8; 64],
}

/// Process spawned a new image via execve
pub const RAW_EVENT_KIND_EXEC: u32 = 1;
/// Process opened a file
pub const RAW_EVENT_KIND_FILE_OPEN: u32 = 2;
/// Process initiated an outbound network connection
pub const RAW_EVENT_KIND_NET_CONNECT: u32 = 3;
/// Process exited
pub const RAW_EVENT_KIND_PROCESS_EXIT: u32 = 4;
/// Any syscall outside the explicitly classified kinds above
pub const RAW_EVENT_KIND_OTHER_SYSCALL: u32 = 5;
