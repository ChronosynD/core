//! Unit tests for the Rust baselines, mirrors the Python tests in
//! `chronosynd-py/tests/baseline/`, parity tests live in `parity_tests.rs`

use chronosynd_baseline::{Baseline, BaselineError, NaiveBaseline, Sediment};
use ndarray::{array, Array2};

fn rng_normal(rows: usize, cols: usize, seed: u64) -> Array2<f64> {
    let mut state = seed.wrapping_mul(0x9E3779B97F4A7C15);
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u1 = (state as f64 / u64::MAX as f64).clamp(1e-300, 1.0 - 1e-15);
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u2 = (state as f64 / u64::MAX as f64).clamp(1e-300, 1.0);
        (-2.0_f64 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    };
    Array2::from_shape_fn((rows, cols), |_| next())
}

#[test]
fn naive_observation_at_the_mean_scores_near_zero() {
    let window = rng_normal(200, 5, 42);
    let mut baseline = NaiveBaseline::new().unwrap();
    baseline.fit(window.view()).unwrap();
    let mean_obs = window.mean_axis(ndarray::Axis(0)).unwrap();
    let score = baseline.score(mean_obs.view()).unwrap();
    assert!(score < 0.1, "score at mean should be near zero, got {score}");
}

#[test]
fn naive_clearly_anomalous_observation_scores_high() {
    let window = rng_normal(200, 5, 42);
    let mut baseline = NaiveBaseline::new().unwrap();
    baseline.fit(window.view()).unwrap();
    let anomaly = ndarray::Array1::from_elem(5, 100.0);
    let score = baseline.score(anomaly.view()).unwrap();
    assert!(score > 1000.0, "anomaly score should be huge, got {score}");
}

#[test]
fn naive_score_before_fit_raises_not_fitted() {
    let baseline = NaiveBaseline::new().unwrap();
    let observation = ndarray::Array1::zeros(3);
    let err = baseline.score(observation.view()).unwrap_err();
    assert!(matches!(err, BaselineError::NotFitted));
}

#[test]
fn naive_fit_rejects_empty_window() {
    let mut baseline = NaiveBaseline::new().unwrap();
    let empty = Array2::<f64>::zeros((0, 5));
    let err = baseline.fit(empty.view()).unwrap_err();
    assert!(matches!(err, BaselineError::EmptyLearningWindow(_)));
}

#[test]
fn naive_fit_rejects_non_finite_values() {
    let mut baseline = NaiveBaseline::new().unwrap();
    let bad = array![[1.0, f64::NAN, 3.0], [4.0, 5.0, 6.0]];
    let err = baseline.fit(bad.view()).unwrap_err();
    assert!(matches!(err, BaselineError::InvalidObservation(_)));
}

#[test]
fn naive_score_rejects_dimension_mismatch() {
    let mut baseline = NaiveBaseline::new().unwrap();
    baseline.fit(Array2::<f64>::zeros((10, 5)).view()).unwrap();
    let wrong_dim = ndarray::Array1::zeros(3);
    let err = baseline.score(wrong_dim.view()).unwrap_err();
    assert!(matches!(err, BaselineError::DimensionMismatch(_)));
}

#[test]
fn naive_score_batch_matches_row_by_row() {
    let window = rng_normal(200, 5, 42);
    let test_batch = rng_normal(50, 5, 7);
    let mut baseline = NaiveBaseline::new().unwrap();
    baseline.fit(window.view()).unwrap();

    let vectorized = baseline.score_batch(test_batch.view()).unwrap();
    for i in 0..test_batch.nrows() {
        let row_score = baseline.score(test_batch.row(i)).unwrap();
        assert!(
            (vectorized[i] - row_score).abs() < 1e-12,
            "row {i}: vectorized {} vs scalar {}",
            vectorized[i],
            row_score
        );
    }
}

#[test]
fn naive_constructor_rejects_invalid_epsilon() {
    let err = NaiveBaseline::with_epsilon(0.0).unwrap_err();
    assert!(matches!(err, BaselineError::InvalidParameter(_)));
    let err = NaiveBaseline::with_epsilon(-1.0).unwrap_err();
    assert!(matches!(err, BaselineError::InvalidParameter(_)));
    let err = NaiveBaseline::with_epsilon(f64::NAN).unwrap_err();
    assert!(matches!(err, BaselineError::InvalidParameter(_)));
}

#[test]
fn sediment_constructs_with_defaults() {
    let baseline = Sediment::new().unwrap();
    assert_eq!(baseline.trim_fraction(), 0.1);
}

#[test]
fn sediment_constructs_with_custom_trim_fraction() {
    let baseline = Sediment::with_params(0.25, 1e-6).unwrap();
    assert_eq!(baseline.trim_fraction(), 0.25);
}

#[test]
fn sediment_rejects_invalid_trim_fraction() {
    for bad in [-0.01, 1.0, 1.5, f64::NAN, f64::INFINITY] {
        let err = Sediment::with_params(bad, 1e-6).unwrap_err();
        assert!(
            matches!(err, BaselineError::InvalidParameter(_)),
            "expected InvalidParameter for trim_fraction={bad}, got {err:?}"
        );
    }
}

#[test]
fn sediment_trim_zero_matches_naive() {
    let window = rng_normal(150, 4, 42);
    let probe = rng_normal(1, 4, 99).row(0).to_owned();

    let mut naive = NaiveBaseline::new().unwrap();
    naive.fit(window.view()).unwrap();
    let naive_score = naive.score(probe.view()).unwrap();

    let mut sediment = Sediment::with_params(0.0, 1e-6).unwrap();
    sediment.fit(window.view()).unwrap();
    let sediment_score = sediment.score(probe.view()).unwrap();

    assert!(
        (naive_score - sediment_score).abs() < 1e-12,
        "trim=0 must equal naive, naive={naive_score} sediment={sediment_score}"
    );
}

#[test]
fn sediment_rejects_trim_too_aggressive_for_window() {
    let mut baseline = Sediment::with_params(0.9, 1e-6).unwrap();
    let tiny = Array2::<f64>::zeros((2, 3));
    let err = baseline.fit(tiny.view()).unwrap_err();
    assert!(matches!(err, BaselineError::EmptyLearningWindow(_)));
}
