//! Sediment, the poisoning-resistant baseline, a symmetric trimmed mean and
//! bias-corrected trimmed std per feature parameterized by `trim_fraction`,
//! mirrors the Python reference and is parity-checked against it

use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use statrs::distribution::{ContinuousCDF, Normal};

use crate::error::{BaselineError, Result};
use crate::independent_gaussian::{
    Baseline, FittedMoments, IndependentGaussianState, MomentsFitter,
};

const MIN_TRIM: f64 = 0.0;
const MAX_TRIM: f64 = 1.0;

/// Trimmed-mean estimator that holds up under bounded learning-window poisoning
#[derive(Debug)]
pub struct Sediment {
    state: IndependentGaussianState<SedimentFitter>,
    trim_fraction: f64,
}

impl Sediment {
    /// Build with the default trim_fraction (0.1) and epsilon (1e-6)
    pub fn new() -> Result<Self> {
        Self::with_params(0.1, 1e-6)
    }

    /// Build with explicit `trim_fraction` and `epsilon`, `trim_fraction` is
    /// the total fraction of samples trimmed per feature split evenly across
    /// the low and high tails, must lie in `[0, 1)`
    pub fn with_params(trim_fraction: f64, epsilon: f64) -> Result<Self> {
        if !trim_fraction.is_finite() || !(MIN_TRIM..MAX_TRIM).contains(&trim_fraction) {
            return Err(BaselineError::InvalidParameter(format!(
                "trim_fraction must be finite and in [{MIN_TRIM}, {MAX_TRIM}), got {trim_fraction}"
            )));
        }
        let std_correction = gaussian_trim_std_correction(trim_fraction);
        let fitter = SedimentFitter {
            trim_fraction,
            std_correction,
        };
        Ok(Self {
            state: IndependentGaussianState::new(fitter, epsilon)?,
            trim_fraction,
        })
    }

    /// Fraction of samples this estimator drops from the tails of each feature
    pub fn trim_fraction(&self) -> f64 {
        self.trim_fraction
    }

    /// Cloned per-feature mean and std after fitting, returns None until `fit` runs
    pub fn fitted_moments(&self) -> Option<(Vec<f64>, Vec<f64>)> {
        self.state.fitted_moments_clone()
    }
}

impl Baseline for Sediment {
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
pub(crate) struct SedimentFitter {
    trim_fraction: f64,
    std_correction: f64,
}

impl MomentsFitter for SedimentFitter {
    fn fit_moments(&self, observations: ArrayView2<'_, f64>) -> Result<FittedMoments> {
        let n_samples = observations.nrows();
        let n_features = observations.ncols();

        if self.trim_fraction == 0.0 {
            let mean = observations
                .mean_axis(Axis(0))
                .expect("mean_axis cannot fail on a non-empty 2-D batch");
            let ddof = if n_samples > 1 { 1.0 } else { 0.0 };
            let std = observations.std_axis(Axis(0), ddof);
            return Ok(FittedMoments { mean, std });
        }

        let per_tail = (self.trim_fraction / 2.0 * n_samples as f64).round() as usize;
        let survivor_count = n_samples.saturating_sub(2 * per_tail);
        if survivor_count < 1 {
            return Err(BaselineError::EmptyLearningWindow(format!(
                "trim_fraction={} leaves 0 samples for n_samples={}, collect more data or trim less",
                self.trim_fraction, n_samples
            )));
        }

        let mut mean = Array1::<f64>::zeros(n_features);
        let mut std = Array1::<f64>::zeros(n_features);
        let mut sorted = vec![0.0_f64; n_samples];

        for j in 0..n_features {
            for i in 0..n_samples {
                sorted[i] = observations[[i, j]];
            }
            sorted.sort_by(|a, b| a.partial_cmp(b).expect("validation rejected NaN inputs"));
            let trimmed = &sorted[per_tail..n_samples - per_tail];
            let trimmed_mean = trimmed.iter().sum::<f64>() / trimmed.len() as f64;
            let trimmed_var = if trimmed.len() > 1 {
                let sum_sq = trimmed
                    .iter()
                    .map(|v| (v - trimmed_mean).powi(2))
                    .sum::<f64>();
                sum_sq / (trimmed.len() - 1) as f64
            } else {
                0.0
            };
            mean[j] = trimmed_mean;
            std[j] = trimmed_var.sqrt() * self.std_correction;
        }

        Ok(FittedMoments { mean, std })
    }
}

/// Multiplicative factor that unbiases a trimmed std against a Gaussian null,
/// the trimmed-sample std underestimates sigma because the dropped tails
/// carry disproportionate variance, this factor unwinds that truncation
fn gaussian_trim_std_correction(trim_fraction: f64) -> f64 {
    if trim_fraction <= 0.0 {
        return 1.0;
    }
    let standard = Normal::new(0.0, 1.0).expect("standard normal is well-defined");
    let cutoff = standard.inverse_cdf(1.0 - trim_fraction / 2.0);
    let phi_cutoff = standard_normal_pdf(cutoff);
    let phi_bulk = 1.0 - trim_fraction;
    let truncated_variance = 1.0 - 2.0 * cutoff * phi_cutoff / phi_bulk;
    1.0 / truncated_variance.sqrt()
}

fn standard_normal_pdf(x: f64) -> f64 {
    let coefficient = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    coefficient * (-0.5 * x * x).exp()
}
