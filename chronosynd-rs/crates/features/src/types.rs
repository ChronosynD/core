//! Public output type produced by feature extractors

/// A feature vector emitted when a per-process window closes,
/// `process_key` identifies the process and `feature_vector` is the
/// dense numeric representation the scoring engine expects
#[derive(Debug, Clone, PartialEq)]
pub struct EmittedFeatures {
    /// Stable identifier for the process this window came from
    pub process_key: String,
    /// Dense numeric features, the scoring engine consumes this as is
    pub feature_vector: Vec<f64>,
}
