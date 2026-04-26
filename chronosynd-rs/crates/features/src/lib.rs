//! Behavioral feature extractors, turn a stream of events into fixed-shape
//! feature vectors the scoring engine consumes, the canonical implementation
//! lives here and the Python side mirrors it for parity

#![deny(unsafe_op_in_unsafe_fn)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod syscall_ngram;
mod types;
mod vocab;

pub use error::FeatureError;
pub use syscall_ngram::SyscallNgramExtractor;
pub use types::EmittedFeatures;
pub use vocab::default_syscall_vocab;
