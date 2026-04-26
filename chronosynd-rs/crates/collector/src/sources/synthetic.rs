//! Deterministic in-process event source used for testing, wraps a
//! pre-built sequence of events and yields them one at a time, reports
//! `Closed` once the sequence is exhausted

use crate::event::Event;
use crate::source::{EventSource, EventSourceError};

/// In-memory event source that replays a fixed sequence
#[derive(Debug)]
pub struct SyntheticSource {
    events: Vec<Event>,
    cursor: usize,
}

impl SyntheticSource {
    /// Build from a fixed sequence of events, yielded in order
    pub fn new(events: Vec<Event>) -> Self {
        Self { events, cursor: 0 }
    }

    /// How many events remain to be yielded
    pub fn remaining(&self) -> usize {
        self.events.len().saturating_sub(self.cursor)
    }
}

impl EventSource for SyntheticSource {
    fn next_event(&mut self) -> Result<Event, EventSourceError> {
        if self.cursor >= self.events.len() {
            return Err(EventSourceError::Closed);
        }
        let event = self.events[self.cursor].clone();
        self.cursor += 1;
        Ok(event)
    }
}
