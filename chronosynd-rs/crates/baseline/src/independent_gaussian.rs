//! Shared plumbing for baselines that fit a per-feature independent Gaussian,
//! NaiveBaseline and Sediment share validation and standardized-residual
//! scoring through this module and only differ in how moments are computed

use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

use crate::error::{BaselineError, Result};

/// Public interface every baseline estimator implements
pub trait Baseline {
    /// Fit the baseline on a learning-window batch of shape `(n_samples, n_features)`
    fn fit(&mut self, observations: ArrayView2<'_, f64>) -> Result<()>;

    /// Score one observation of shape `(n_features,)`, higher means more anomalous
    fn score(&self, observation: ArrayView1<'_, f64>) -> Result<f64>;

    /// Score every row of a batch, returning one drift score per row
    fn score_batch(&self, observations: ArrayView2<'_, f64>) -> Result<Array1<f64>>;
}

/// Per-feature mean and standard deviation produced by a concrete fit
#[derive(Debug)]
pub(crate) struct FittedMoments {
    pub mean: Array1<f64>,
    pub std: Array1<f64>,
}

/// Strategy used by the public estimators to compute moments from a batch,
/// concrete implementations live in `naive` and `sediment`
pub(crate) trait MomentsFitter: core::fmt::Debug {
    fn fit_moments(&self, observations: ArrayView2<'_, f64>) -> Result<FittedMoments>;
}

/// Container holding the shared state for an independent-Gaussian estimator,
/// generic over the moment-fitting strategy
#[derive(Debug)]
pub(crate) struct IndependentGaussianState<F: MomentsFitter> {
    fitter: F,
    epsilon: f64,
    moments: Option<FittedMoments>,
}

impl<F: MomentsFitter> IndependentGaussianState<F> {
    pub fn new(fitter: F, epsilon: f64) -> Result<Self> {
        if !epsilon.is_finite() || epsilon <= 0.0 {
            return Err(BaselineError::InvalidParameter(format!(
                "epsilon must be a positive finite float, got {epsilon}"
            )));
        }
        Ok(Self {
            fitter,
            epsilon,
            moments: None,
        })
    }

    pub fn fit(&mut self, observations: ArrayView2<'_, f64>) -> Result<()> {
        validate_batch(observations)?;
        self.moments = Some(self.fitter.fit_moments(observations)?);
        Ok(())
    }

    pub fn score(&self, observation: ArrayView1<'_, f64>) -> Result<f64> {
        let moments = self
            .moments
            .as_ref()
            .ok_or(BaselineError::NotFitted)?;
        validate_observation(observation, moments.mean.len())?;
        Ok(score_one(observation, moments, self.epsilon))
    }

    pub fn score_batch(&self, observations: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        let moments = self
            .moments
            .as_ref()
            .ok_or(BaselineError::NotFitted)?;
        validate_batch_for_score(observations, moments.mean.len())?;

        let mut scores = Array1::<f64>::zeros(observations.nrows());
        for (i, row) in observations.axis_iter(Axis(0)).enumerate() {
            scores[i] = score_one(row, moments, self.epsilon);
        }
        Ok(scores)
    }

    /// Cloned per-feature mean and std once the baseline has been fitted, used
    /// by callers that need to persist or transport the fitted moments
    pub(crate) fn fitted_moments_clone(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        self.moments
            .as_ref()
            .map(|m| (m.mean.to_vec(), m.std.to_vec()))
    }
}

fn score_one(observation: ArrayView1<'_, f64>, moments: &FittedMoments, epsilon: f64) -> f64 {
    let mut total = 0.0;
    for ((value, mean), std) in observation
        .iter()
        .zip(moments.mean.iter())
        .zip(moments.std.iter())
    {
        let standardized = (value - mean) / (std + epsilon);
        total += standardized * standardized;
    }
    total
}

fn validate_batch(observations: ArrayView2<'_, f64>) -> Result<()> {
    if observations.nrows() == 0 {
        return Err(BaselineError::EmptyLearningWindow(
            "cannot fit on zero observations, need at least one sample".into(),
        ));
    }
    if observations.ncols() == 0 {
        return Err(BaselineError::DimensionMismatch(
            "cannot fit on zero-feature observations, need at least one column".into(),
        ));
    }
    if observations.iter().any(|v| !v.is_finite()) {
        return Err(BaselineError::InvalidObservation(
            "observations contain non-finite values, NaN or inf".into(),
        ));
    }
    Ok(())
}

fn validate_observation(observation: ArrayView1<'_, f64>, expected_dim: usize) -> Result<()> {
    if observation.len() != expected_dim {
        return Err(BaselineError::DimensionMismatch(format!(
            "observation len {} does not match fitted feature dimension {expected_dim}",
            observation.len(),
        )));
    }
    if observation.iter().any(|v| !v.is_finite()) {
        return Err(BaselineError::InvalidObservation(
            "observation contains non-finite values, NaN or inf".into(),
        ));
    }
    Ok(())
}

fn validate_batch_for_score(observations: ArrayView2<'_, f64>, expected_dim: usize) -> Result<()> {
    if observations.ncols() != expected_dim {
        return Err(BaselineError::DimensionMismatch(format!(
            "batch feature dim {} does not match fitted feature dimension {expected_dim}",
            observations.ncols(),
        )));
    }
    if observations.iter().any(|v| !v.is_finite()) {
        return Err(BaselineError::InvalidObservation(
            "observations contain non-finite values, NaN or inf".into(),
        ));
    }
    Ok(())
}
