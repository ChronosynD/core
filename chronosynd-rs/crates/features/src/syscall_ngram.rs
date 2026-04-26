//! Syscall 1-gram histogram extractor over per-process disjoint windows,
//! every `window_size` events the extractor emits a normalized histogram
//! sized `vocab.len() + 1` (the trailing slot is the out-of-vocab bucket)

use std::collections::HashMap;

use crate::error::{FeatureError, Result};
use crate::types::EmittedFeatures;

/// Disjoint-window 1-gram histogram extractor over a fixed syscall vocabulary
#[derive(Debug)]
pub struct SyscallNgramExtractor {
    vocab_index: HashMap<u32, usize>,
    feature_dim: usize,
    window_size: usize,
    per_process: HashMap<String, Vec<f64>>,
    pending: HashMap<String, usize>,
}

impl SyscallNgramExtractor {
    /// Build with an explicit syscall vocabulary and disjoint window size,
    /// the emitted feature vector has dimension `vocab.len() + 1` with the
    /// trailing slot counting out-of-vocab syscalls
    pub fn new(vocab: Vec<u32>, window_size: usize) -> Result<Self> {
        if window_size == 0 {
            return Err(FeatureError::InvalidParameter(
                "window_size must be at least 1".into(),
            ));
        }
        let mut vocab_index = HashMap::with_capacity(vocab.len());
        for (idx, syscall_nr) in vocab.iter().enumerate() {
            if vocab_index.insert(*syscall_nr, idx).is_some() {
                return Err(FeatureError::DuplicateVocabEntry(*syscall_nr));
            }
        }
        let feature_dim = vocab.len() + 1;
        Ok(Self {
            vocab_index,
            feature_dim,
            window_size,
            per_process: HashMap::new(),
            pending: HashMap::new(),
        })
    }

    /// Dimension of every emitted feature vector, stays constant for the
    /// lifetime of the extractor
    pub fn feature_dim(&self) -> usize {
        self.feature_dim
    }

    /// Index in the feature vector reserved for syscalls outside the vocabulary
    pub fn other_bucket_index(&self) -> usize {
        self.feature_dim - 1
    }

    /// Number of processes the extractor is currently tracking
    pub fn tracked_process_count(&self) -> usize {
        self.per_process.len()
    }

    /// Feed one observed syscall, returns `Some(EmittedFeatures)` when the
    /// per-process window closes, otherwise `None`
    pub fn accumulate(&mut self, process_key: &str, syscall_nr: u32) -> Option<EmittedFeatures> {
        let counts = self
            .per_process
            .entry(process_key.to_string())
            .or_insert_with(|| vec![0.0; self.feature_dim]);
        let bucket = self
            .vocab_index
            .get(&syscall_nr)
            .copied()
            .unwrap_or(self.feature_dim - 1);
        counts[bucket] += 1.0;

        let pending = self
            .pending
            .entry(process_key.to_string())
            .or_insert(0);
        *pending += 1;

        if *pending < self.window_size {
            return None;
        }

        let denom = self.window_size as f64;
        let normalized: Vec<f64> = counts.iter().map(|count| count / denom).collect();
        let emitted = EmittedFeatures {
            process_key: process_key.to_string(),
            feature_vector: normalized,
        };

        self.per_process.insert(process_key.to_string(), vec![0.0; self.feature_dim]);
        self.pending.insert(process_key.to_string(), 0);
        Some(emitted)
    }

    /// Discard any in-progress window for a process, the next event for
    /// that process starts a fresh window
    pub fn reset_process(&mut self, process_key: &str) {
        self.per_process.remove(process_key);
        self.pending.remove(process_key);
    }
}
