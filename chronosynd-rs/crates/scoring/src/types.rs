//! Public scoring types

/// Single scoring call against the cached baseline for a process
#[derive(Debug, Clone)]
pub struct ScoringRequest<'a> {
    /// Process the observation belongs to
    pub process_key: &'a str,
    /// Feature vector to score, shape `(n_features,)`
    pub observation: &'a [f64],
}

/// Outcome of a scoring call
#[derive(Debug, Clone, PartialEq)]
pub struct ScoringResult {
    /// Drift score, sum of squared standardized residuals against the cached baseline
    pub score: f64,
    /// Threshold the score was compared against
    pub threshold: f64,
    /// Set when `score > threshold`, carries the routable alert payload
    pub alert: Option<DriftAlert>,
}

/// Drift alert emitted when the score crosses the per-process threshold
#[derive(Debug, Clone, PartialEq)]
pub struct DriftAlert {
    /// Process the alert is for
    pub process_key: String,
    /// Score that tripped the threshold
    pub score: f64,
    /// Threshold the score was compared against
    pub threshold: f64,
}
