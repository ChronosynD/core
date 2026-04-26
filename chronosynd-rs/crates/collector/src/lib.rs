//! Userspace half of the ChronosynD collector, loads BPF programs and
//! drains the kernel ring buffer into normalized events, the daemon entry
//! point in `daemon` ties storage, scoring, and the event source together

#![deny(unsafe_op_in_unsafe_fn)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod daemon;
mod event;
mod source;
pub mod sources;
pub mod wire;

pub use daemon::{run_daemon, EXIT_ERR, EXIT_OK};
pub use event::{Event, EventKind};
pub use source::{EventSource, EventSourceError};
pub use wire::{read_recording, read_recording_filter, EventRecorder, WireEvent};

pub mod replay;
pub use replay::extract_observations_from_recording;
