//! In-memory scorer holding the working set of baselines, caches
//! `StoredBaseline` records keyed by process key and scores synchronously,
//! callers warm the cache via `upsert_baseline` or `sync::warm_from_store`

use std::collections::HashMap;

use chronosynd_storage::StoredBaseline;

use crate::error::{Result, ScoringError};
use crate::types::{DriftAlert, ScoringRequest, ScoringResult};

/// Cached working set of baselines and per-process thresholds
#[derive(Debug)]
pub struct Scorer {
    baselines: HashMap<String, StoredBaseline>,
    thresholds: HashMap<String, f64>,
    epsilon: f64,
    default_threshold: f64,
}

impl Scorer {
    /// Build a scorer with explicit `epsilon` and `default_threshold`
    pub fn new(epsilon: f64, default_threshold: f64) -> Result<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(ScoringError::InvalidParameter(format!(
                "epsilon must be a positive finite float, got {epsilon}"
            )));
        }
        if !default_threshold.is_finite() || default_threshold < 0.0 {
            return Err(ScoringError::InvalidParameter(format!(
                "default_threshold must be a finite non-negative float, got {default_threshold}"
            )));
        }
        Ok(Self {
            baselines: HashMap::new(),
            thresholds: HashMap::new(),
            epsilon,
            default_threshold,
        })
    }

    /// Insert or replace the cached baseline for a process
    pub fn upsert_baseline(&mut self, baseline: StoredBaseline) {
        self.baselines.insert(baseline.process_key.clone(), baseline);
    }

    /// Drop the cached baseline and any per-process threshold
    pub fn forget(&mut self, process_key: &str) {
        self.baselines.remove(process_key);
        self.thresholds.remove(process_key);
    }

    /// Number of baselines currently cached
    pub fn baseline_count(&self) -> usize {
        self.baselines.len()
    }

    /// Set or replace the threshold for a specific process
    pub fn set_threshold(&mut self, process_key: &str, threshold: f64) -> Result<()> {
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(ScoringError::InvalidParameter(format!(
                "threshold must be a finite non-negative float, got {threshold}"
            )));
        }
        self.thresholds.insert(process_key.to_string(), threshold);
        Ok(())
    }

    /// Effective threshold for `process_key`, falling back to the default
    pub fn threshold_for(&self, process_key: &str) -> f64 {
        self.thresholds
            .get(process_key)
            .copied()
            .unwrap_or(self.default_threshold)
    }

    /// Score an observation against the cached baseline
    pub fn score(&self, request: &ScoringRequest<'_>) -> Result<ScoringResult> {
        let baseline = self
            .baselines
            .get(request.process_key)
            .ok_or_else(|| ScoringError::BaselineNotCached(request.process_key.to_string()))?;

        // Defense in depth, a tampered store could hand us mismatched mean
        // and std lengths or a feature_dim that does not match either, the
        // store's audit chain catches the row tamper but a corrupt cache
        // load should still fail closed rather than score against a
        // truncated zip
        if baseline.mean.len() != baseline.feature_dim
            || baseline.std.len() != baseline.feature_dim
        {
            return Err(ScoringError::DimensionMismatch {
                process_key: request.process_key.to_string(),
                got: baseline.mean.len(),
                expected: baseline.feature_dim,
            });
        }
        if request.observation.len() != baseline.feature_dim {
            return Err(ScoringError::DimensionMismatch {
                process_key: request.process_key.to_string(),
                got: request.observation.len(),
                expected: baseline.feature_dim,
            });
        }
        if request.observation.iter().any(|v| !v.is_finite()) {
            return Err(ScoringError::NonFiniteObservation(
                request.process_key.to_string(),
            ));
        }

        let score = score_against_moments(
            request.observation,
            &baseline.mean,
            &baseline.std,
            self.epsilon,
        );
        let threshold = self.threshold_for(request.process_key);
        let alert = (score > threshold).then(|| DriftAlert {
            process_key: request.process_key.to_string(),
            score,
            threshold,
        });

        Ok(ScoringResult {
            score,
            threshold,
            alert,
        })
    }
}

fn score_against_moments(observation: &[f64], mean: &[f64], std: &[f64], epsilon: f64) -> f64 {
    let mut total = 0.0;
    for ((value, m), s) in observation.iter().zip(mean.iter()).zip(std.iter()) {
        let standardized = (value - m) / (s + epsilon);
        total += standardized * standardized;
    }
    total
}
