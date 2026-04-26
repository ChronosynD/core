//! Error types for baseline estimators, mirrors the Python exception
//! hierarchy in `chronosynd_py.core` so parity tests can match errors

use thiserror::Error;

/// Every error a baseline estimator can produce
#[derive(Debug, Error)]
pub enum BaselineError {
    /// Estimator was scored before being fit
    #[error("call fit before score")]
    NotFitted,

    /// An array shape did not match what the estimator expected
    #[error("dimension mismatch: {0}")]
    DimensionMismatch(String),

    /// Fit was attempted with zero observations
    #[error("empty learning window: {0}")]
    EmptyLearningWindow(String),

    /// Constructor was given an out-of-range parameter
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// Observation values were malformed, NaN or inf
    #[error("invalid observation values: {0}")]
    InvalidObservation(String),
}

/// Result alias used throughout the crate
pub type Result<T> = core::result::Result<T, BaselineError>;
