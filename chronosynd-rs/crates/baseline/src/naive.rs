//! Naive baseline, one Gaussian per feature under an independence assumption,
//! the prior-work reference point Sediment is compared against in the paper
//! and known to be highly vulnerable to learning-window poisoning

use ndarray::{Array1, ArrayView1, ArrayView2, Axis};

use crate::error::Result;
use crate::independent_gaussian::{
    Baseline, FittedMoments, IndependentGaussianState, MomentsFitter,
};

/// Mean-and-standard-deviation baseline matching the prior-work shape
#[derive(Debug)]
pub struct NaiveBaseline {
    state: IndependentGaussianState<NaiveFitter>,
}

impl NaiveBaseline {
    /// Build a NaiveBaseline with the default epsilon
    pub fn new() -> Result<Self> {
        Self::with_epsilon(1e-6)
    }

    /// Build a NaiveBaseline with a custom epsilon, must be positive and finite
    pub fn with_epsilon(epsilon: f64) -> Result<Self> {
        Ok(Self {
            state: IndependentGaussianState::new(NaiveFitter, epsilon)?,
        })
    }
}

impl NaiveBaseline {
    /// Cloned per-feature mean and std after fitting, returns None until `fit` runs
    pub fn fitted_moments(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        self.state.fitted_moments_clone()
    }
}

impl Baseline for NaiveBaseline {
    fn fit(&mut self, observations: ArrayView2<'_, f64>) -> Result<()> {
        self.state.fit(observations)
    }

    fn score(&self, observation: ArrayView1<'_, f64>) -> Result<f64> {
        self.state.score(observation)
    }

    fn score_batch(&self, observations: ArrayView2<'_, f64>) -> Result<Array1<f64>> {
        self.state.score_batch(observations)
    }
}

#[derive(Debug)]
pub(crate) struct NaiveFitter;

impl MomentsFitter for NaiveFitter {
    fn fit_moments(&self, observations: ArrayView2<'_, f64>) -> Result<FittedMoments> {
        let n_samples = observations.nrows();
        let mean = observations
            .mean_axis(Axis(0))
            .expect("mean_axis cannot fail on a non-empty 2-D batch");
        let std = if n_samples > 1 {
            observations.std_axis(Axis(0), 1.0)
        } else {
            observations.std_axis(Axis(0), 0.0)
        };
        Ok(FittedMoments { mean, std })
    }
}
