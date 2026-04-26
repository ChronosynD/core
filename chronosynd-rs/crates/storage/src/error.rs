//! Error type for the storage layer

use thiserror::Error;

/// Anything the storage layer can fail with
#[derive(Debug, Error)]
pub enum StorageError {
    /// Underlying SQLite operation failed
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),

    /// Serializing or deserializing a payload failed
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// A caller-supplied value did not pass validation
    #[error("invalid argument: {0}")]
    InvalidArgument(String),

    /// The audit log's hash chain is broken at the indicated sequence number
    #[error("audit log tampered at seq {seq}")]
    AuditTampered {
        /// First sequence number where the recomputed hash disagrees with the stored hash
        seq: i64,
    },

    /// A required record was not present in the store
    #[error("not found: {0}")]
    NotFound(String),

    /// A row read from the database held a value outside its allowed domain
    /// (negative width or count, oversized payload), the audit chain catches
    /// deliberate tampering, this catches field-level corruption inside a row
    #[error("corrupt row: {0}")]
    CorruptRow(String),
}

/// Result alias used throughout the crate
pub type Result<T> = core::result::Result<T, StorageError>;
