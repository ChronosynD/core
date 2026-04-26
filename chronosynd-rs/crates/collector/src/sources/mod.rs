//! Event source implementations, `synthetic` replays a deterministic
//! event stream for tests and Windows hosts, `bpf` is gated behind
//! `feature = "bpf"` plus `target_os = "linux"` and reads from the kernel

#[cfg(all(target_os = "linux", feature = "bpf"))]
pub mod bpf;

pub mod synthetic;
