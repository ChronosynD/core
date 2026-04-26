//! Real-time drift scoring engine, scores feature vectors against cached
//! baselines and emits alerts above the per-process threshold, the cache
//! is warmed externally to keep the scoring path I/O-free

#![deny(unsafe_op_in_unsafe_fn)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod error;
mod scorer;
mod sync;
mod types;

pub use error::ScoringError;
pub use scorer::Scorer;
pub use sync::warm_from_store;
pub use types::{DriftAlert, ScoringRequest, ScoringResult};
