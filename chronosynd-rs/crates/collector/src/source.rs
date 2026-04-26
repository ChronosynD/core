//! Generic event-source interface the collector pulls from, abstracts
//! over the synthetic source for testing and the BPF source for runtime
//! so the pipeline does not depend on Linux at the trait level

use thiserror::Error;

use crate::event::Event;

/// Anything an event source can fail with
#[derive(Debug, Error)]
pub enum EventSourceError {
    /// The source has no more events and will not produce any in the future
    #[error("event source is closed")]
    Closed,

    /// Source-specific failure, owns the upstream error message
    #[error("event source failed: {0}")]
    Backend(String),
}

/// A pull-based source of behavioral events, `next_event` blocks until an
/// event arrives or returns `Closed` when the source is exhausted, or
/// `Backend` for transient failures the caller may want to retry
pub trait EventSource {
    /// Block until the next event arrives, or report the source's terminal state
    fn next_event(&mut self) -> Result<Event, EventSourceError>;
}
