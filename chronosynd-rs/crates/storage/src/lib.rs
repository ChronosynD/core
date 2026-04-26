//! Baseline persistence with tamper evidence, baselines live in SQLite
//! and every state-changing operation is appended to a hash-chained audit
//! log so corruption is detectable on the next verify pass

#![deny(unsafe_op_in_unsafe_fn)]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod audit;
mod error;
mod schema;
mod store;
mod types;

pub use audit::AuditVerification;
pub use error::StorageError;
pub use store::BaselineStore;
pub use types::{MaintenanceWindow, StoredBaseline};
