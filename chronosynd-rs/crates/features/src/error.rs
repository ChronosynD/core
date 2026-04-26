//! Error type for the feature extractors

use thiserror::Error;

/// Anything a feature extractor can fail with at construction time
#[derive(Debug, Error)]
pub enum FeatureError {
    /// Constructor parameter was out of range
    #[error("invalid parameter: {0}")]
    InvalidParameter(String),

    /// Vocabulary contained duplicate syscall numbers
    #[error("duplicate syscall number {0} in vocabulary")]
    DuplicateVocabEntry(u32),
}

/// Result alias used throughout the crate
pub type Result<T> = core::result::Result<T, FeatureError>;
