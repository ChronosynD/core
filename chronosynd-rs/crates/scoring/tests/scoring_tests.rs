//! Tests for the in-memory scorer

use chronosynd_scoring::{warm_from_store, ScoringError, ScoringRequest, Scorer};
use chronosynd_storage::{BaselineStore, StoredBaseline};

fn baseline(key: &str, dim: usize, mean: f64, std: f64) -> StoredBaseline {
    StoredBaseline {
        process_key: key.into(),
        feature_dim: dim,
        mean: vec![mean; dim],
        std: vec![std; dim],
        estimator_kind: "naive".into(),
        fitted_at_ns: 1,
        sample_count: 100,
    }
}

#[test]
fn observation_at_the_mean_scores_zero() {
    let mut scorer = Scorer::new(1e-6, 100.0).unwrap();
    scorer.upsert_baseline(baseline("nginx", 4, 0.0, 1.0));

    let observation = vec![0.0; 4];
    let request = ScoringRequest {
        process_key: "nginx",
        observation: &observation,
    };
    let result = scorer.score(&request).unwrap();
    assert!(result.score < 1e-9, "expected near-zero score, got {}", result.score);
    assert!(result.alert.is_none());
}

#[test]
fn anomalous_observation_triggers_an_alert() {
    let mut scorer = Scorer::new(1e-6, 50.0).unwrap();
    scorer.upsert_baseline(baseline("nginx", 4, 0.0, 1.0));

    let observation = vec![10.0; 4];
    let request = ScoringRequest {
        process_key: "nginx",
        observation: &observation,
    };
    let result = scorer.score(&request).unwrap();
    assert!(result.score > 50.0);
    let alert = result.alert.unwrap();
    assert_eq!(alert.process_key, "nginx");
    assert_eq!(alert.score, result.score);
    assert_eq!(alert.threshold, 50.0);
}

#[test]
fn per_process_threshold_overrides_default() {
    let mut scorer = Scorer::new(1e-6, 1000.0).unwrap();
    scorer.upsert_baseline(baseline("nginx", 4, 0.0, 1.0));
    scorer.set_threshold("nginx", 1.0).unwrap();

    let observation = vec![1.0; 4];
    let request = ScoringRequest {
        process_key: "nginx",
        observation: &observation,
    };
    let result = scorer.score(&request).unwrap();
    assert_eq!(result.threshold, 1.0);
    assert!(result.alert.is_some(), "expected alert for low threshold");
}

#[test]
fn missing_baseline_returns_not_cached_error() {
    let scorer = Scorer::new(1e-6, 1.0).unwrap();
    let observation = vec![0.0; 4];
    let request = ScoringRequest {
        process_key: "ghost",
        observation: &observation,
    };
    let err = scorer.score(&request).unwrap_err();
    assert!(matches!(err, ScoringError::BaselineNotCached(ref name) if name == "ghost"));
}

#[test]
fn dimension_mismatch_is_rejected() {
    let mut scorer = Scorer::new(1e-6, 1.0).unwrap();
    scorer.upsert_baseline(baseline("nginx", 4, 0.0, 1.0));

    let observation = vec![0.0; 5];
    let request = ScoringRequest {
        process_key: "nginx",
        observation: &observation,
    };
    let err = scorer.score(&request).unwrap_err();
    assert!(matches!(
        err,
        ScoringError::DimensionMismatch {
            got: 5,
            expected: 4,
            ..
        }
    ));
}

#[test]
fn non_finite_observation_is_rejected() {
    let mut scorer = Scorer::new(1e-6, 1.0).unwrap();
    scorer.upsert_baseline(baseline("nginx", 3, 0.0, 1.0));

    let observation = vec![1.0, f64::NAN, 1.0];
    let request = ScoringRequest {
        process_key: "nginx",
        observation: &observation,
    };
    let err = scorer.score(&request).unwrap_err();
    assert!(matches!(err, ScoringError::NonFiniteObservation(_)));
}

#[test]
fn forget_drops_baseline_and_threshold() {
    let mut scorer = Scorer::new(1e-6, 1.0).unwrap();
    scorer.upsert_baseline(baseline("nginx", 3, 0.0, 1.0));
    scorer.set_threshold("nginx", 5.0).unwrap();

    scorer.forget("nginx");
    assert_eq!(scorer.baseline_count(), 0);
    assert_eq!(scorer.threshold_for("nginx"), 1.0);
}

#[test]
fn upsert_replaces_existing_baseline_for_same_key() {
    let mut scorer = Scorer::new(1e-6, 1000.0).unwrap();
    scorer.upsert_baseline(baseline("nginx", 4, 0.0, 1.0));
    scorer.upsert_baseline(baseline("nginx", 4, 5.0, 1.0));

    assert_eq!(scorer.baseline_count(), 1);
    let observation = vec![5.0; 4];
    let result = scorer
        .score(&ScoringRequest {
            process_key: "nginx",
            observation: &observation,
        })
        .unwrap();
    assert!(result.score < 1e-9, "expected near-zero score after replacement");
}

#[test]
fn invalid_epsilon_is_rejected_at_construction() {
    assert!(matches!(
        Scorer::new(0.0, 1.0).unwrap_err(),
        ScoringError::InvalidParameter(_)
    ));
    assert!(matches!(
        Scorer::new(-1.0, 1.0).unwrap_err(),
        ScoringError::InvalidParameter(_)
    ));
    assert!(matches!(
        Scorer::new(f64::NAN, 1.0).unwrap_err(),
        ScoringError::InvalidParameter(_)
    ));
}

#[test]
fn invalid_default_threshold_is_rejected_at_construction() {
    assert!(matches!(
        Scorer::new(1e-6, -1.0).unwrap_err(),
        ScoringError::InvalidParameter(_)
    ));
    assert!(matches!(
        Scorer::new(1e-6, f64::NAN).unwrap_err(),
        ScoringError::InvalidParameter(_)
    ));
}

#[test]
fn invalid_per_process_threshold_is_rejected() {
    let mut scorer = Scorer::new(1e-6, 1.0).unwrap();
    assert!(matches!(
        scorer.set_threshold("nginx", -0.5).unwrap_err(),
        ScoringError::InvalidParameter(_)
    ));
}

#[test]
fn warm_from_store_loads_every_baseline() {
    let mut store = BaselineStore::open_in_memory().unwrap();
    store.put_baseline(&baseline("a", 3, 0.0, 1.0)).unwrap();
    store.put_baseline(&baseline("b", 3, 0.0, 1.0)).unwrap();
    store.put_baseline(&baseline("c", 3, 0.0, 1.0)).unwrap();

    let mut scorer = Scorer::new(1e-6, 1.0).unwrap();
    let count = warm_from_store(&mut scorer, &store).unwrap();
    assert_eq!(count, 3);
    assert_eq!(scorer.baseline_count(), 3);
}

#[test]
fn warm_from_store_replaces_cached_versions() {
    let mut store = BaselineStore::open_in_memory().unwrap();
    let mut scorer = Scorer::new(1e-6, 1000.0).unwrap();

    scorer.upsert_baseline(baseline("nginx", 4, 0.0, 100.0));
    store.put_baseline(&baseline("nginx", 4, 0.0, 1.0)).unwrap();

    let count = warm_from_store(&mut scorer, &store).unwrap();
    assert_eq!(count, 1);

    let observation = vec![5.0; 4];
    let result = scorer
        .score(&ScoringRequest {
            process_key: "nginx",
            observation: &observation,
        })
        .unwrap();
    // After warm_from_store the std becomes 1.0, so 5.0 over four features scores around 100
    assert!(result.score > 50.0, "warm_from_store did not replace, got score {}", result.score);
}
