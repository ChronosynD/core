//! Error type for the scoring engine

use thiserror::Error;

/// Anything the scorer can fail with on a per-request basis
#[derive(Debug, Error)]
pub enum ScoringError {
    /// No baseline has been cached for the requested process key
    #[error("no baseline cached for process key {0}")]
    BaselineNotCached(String),

    /// Observation feature dimension does not match the cached baseline
    #[error(
        "dimension mismatch for {process_key}, observation has {got} features, baseline expects {expected}"
    )]
    DimensionMismatch {
        /// Process key being scored
        process_key: String,
        /// Feature count of the supplied observation
        got: usize,
        /// Feature count the cached baseline was fitted on
        expected: usize,
    },

    /// Observation contains NaN or infinity
    #[error("observation for {0} contains non-finite values")]
    NonFiniteObservation(String),

    /// Estimator was constructed with an out-of-range parameter
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),
}

/// Result alias used throughout the crate
pub type Result<T> = core::result::Result<T, ScoringError>;
