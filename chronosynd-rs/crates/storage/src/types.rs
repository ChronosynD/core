//! Public data types persisted by the storage layer

use serde::{Deserialize, Serialize};

/// A fitted baseline as it lives on disk
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StoredBaseline {
    /// Stable identifier for the process the baseline tracks
    pub process_key: String,
    /// Number of features in the per-feature mean and std vectors
    pub feature_dim: usize,
    /// Per-feature mean vector
    pub mean: Vec<f64>,
    /// Per-feature standard deviation vector
    pub std: Vec<f64>,
    /// Which estimator produced this baseline
    pub estimator_kind: String,
    /// Monotonic timestamp at which the baseline was fitted
    pub fitted_at_ns: u64,
    /// How many observations went into the fit
    pub sample_count: u64,
}

/// An operator-tagged maintenance window
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaintenanceWindow {
    /// Auto-assigned identifier, stable across opens
    pub id: i64,
    /// Monotonic timestamp at which the window started
    pub start_ns: u64,
    /// Monotonic timestamp at which the window ended, `None` while still active
    pub end_ns: Option<u64>,
    /// Free-text note the operator attached to the window
    pub note: Option<String>,
}
