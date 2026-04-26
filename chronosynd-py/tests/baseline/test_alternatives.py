"""Unit tests for ConsensusBaseline and AnomalyWithinBaseline"""

from __future__ import annotations

import numpy as np
import pytest

from chronosynd_py.baseline.alternatives import (
    AnomalyWithinBaseline,
    ConsensusBaseline,
)
from chronosynd_py.baseline.naive import NaiveBaseline
from chronosynd_py.core import (
    BaselineNotFittedError,
    DimensionMismatchError,
    EmptyLearningWindowError,
    InvalidObservationError,
    InvalidParameterError,
)

# ConsensusBaseline


def test_consensus_observation_at_the_mean_scores_near_zero() -> None:
    rng = np.random.default_rng(seed=42)
    window = rng.normal(size=(400, 5))

    baseline = ConsensusBaseline(n_sub_windows=4)
    baseline.fit(window)

    score_at_mean = baseline.score(window.mean(axis=0))
    assert score_at_mean < 0.1


def test_consensus_drops_features_with_disagreeing_sub_windows() -> None:
    # feature 0 is clean, feature 1 has its mean shifted in the last sub-window
    rng = np.random.default_rng(seed=7)
    n_samples = 400
    feature_clean = rng.normal(size=n_samples)
    feature_shifted = rng.normal(size=n_samples)
    feature_shifted[300:] += 50.0  # massive shift in the last quarter
    window = np.column_stack([feature_clean, feature_shifted])

    baseline = ConsensusBaseline(n_sub_windows=4, disagreement_threshold=3.0)
    baseline.fit(window)

    mask = baseline.consensus_mask
    assert bool(mask[0]) is True
    assert bool(mask[1]) is False


def test_consensus_keeps_all_features_when_subwindows_agree() -> None:
    rng = np.random.default_rng(seed=11)
    window = rng.normal(size=(400, 6))

    baseline = ConsensusBaseline(n_sub_windows=4)
    baseline.fit(window)

    assert baseline.consensus_mask.all()


def test_consensus_score_uses_only_consensus_features() -> None:
    rng = np.random.default_rng(seed=2)
    n_samples = 400
    feature_clean = rng.normal(size=n_samples)
    feature_drifted = rng.normal(size=n_samples)
    feature_drifted[300:] += 50.0
    window = np.column_stack([feature_clean, feature_drifted])

    baseline = ConsensusBaseline(n_sub_windows=4, disagreement_threshold=3.0)
    baseline.fit(window)

    # the dropped feature contributes nothing, the clean one drives the score
    score_anomaly_in_dropped_only = baseline.score(np.array([0.0, 1000.0]))
    assert score_anomaly_in_dropped_only < 1.0


def test_consensus_score_batch_matches_score_row_by_row() -> None:
    rng = np.random.default_rng(seed=4)
    window = rng.normal(size=(200, 4))
    test_batch = rng.normal(size=(20, 4))

    baseline = ConsensusBaseline(n_sub_windows=4)
    baseline.fit(window)

    batch_scores = baseline.score_batch(test_batch)
    row_scores = np.array([baseline.score(row) for row in test_batch])
    np.testing.assert_allclose(batch_scores, row_scores, rtol=1e-12, atol=1e-12)


def test_consensus_score_before_fit_raises() -> None:
    baseline = ConsensusBaseline()
    with pytest.raises(BaselineNotFittedError):
        baseline.score(np.zeros(3))


def test_consensus_consensus_mask_before_fit_raises() -> None:
    baseline = ConsensusBaseline()
    with pytest.raises(BaselineNotFittedError):
        _ = baseline.consensus_mask


def test_consensus_fit_rejects_too_few_samples() -> None:
    baseline = ConsensusBaseline(n_sub_windows=4)
    with pytest.raises(EmptyLearningWindowError):
        baseline.fit(np.zeros((3, 5)))


def test_consensus_fit_rejects_non_finite_values() -> None:
    baseline = ConsensusBaseline()
    with pytest.raises(InvalidObservationError):
        baseline.fit(np.array([[1.0, np.nan], [4.0, 5.0], [7.0, 8.0], [9.0, 10.0]]))


def test_consensus_score_rejects_dimension_mismatch() -> None:
    baseline = ConsensusBaseline()
    baseline.fit(np.ones((20, 5)) + np.random.default_rng(0).normal(size=(20, 5)))
    with pytest.raises(DimensionMismatchError):
        baseline.score(np.zeros(3))


@pytest.mark.parametrize("bad_n", [0, 1, -3])
def test_consensus_constructor_rejects_invalid_n_sub_windows(bad_n: int) -> None:
    with pytest.raises(InvalidParameterError):
        ConsensusBaseline(n_sub_windows=bad_n)


@pytest.mark.parametrize(
    "bad_threshold", [0.0, -1.0, float("nan"), float("inf")]
)
def test_consensus_constructor_rejects_invalid_threshold(bad_threshold: float) -> None:
    with pytest.raises(InvalidParameterError):
        ConsensusBaseline(disagreement_threshold=bad_threshold)


# AnomalyWithinBaseline


def test_anomaly_within_observation_at_the_mean_scores_near_zero() -> None:
    rng = np.random.default_rng(seed=42)
    window = rng.normal(size=(200, 5))

    baseline = AnomalyWithinBaseline(drop_fraction=0.1)
    baseline.fit(window)

    score_at_mean = baseline.score(window.mean(axis=0))
    assert score_at_mean < 0.1


def test_anomaly_within_resists_a_handful_of_obvious_outliers() -> None:
    # ten extreme injections in a clean window should be dropped before refit
    rng = np.random.default_rng(seed=123)
    target = np.full(4, 50.0)
    clean = rng.normal(size=(200, 4))
    poisoned = np.vstack([clean, np.tile(target, (10, 1))])

    naive = NaiveBaseline()
    naive.fit(poisoned)

    robust = AnomalyWithinBaseline(drop_fraction=0.1)
    robust.fit(poisoned)

    # naive's score on the target collapses. AnomalyWithin holds it up
    assert robust.score(target) > 5.0 * naive.score(target)


def test_anomaly_within_with_zero_drop_matches_naive() -> None:
    rng = np.random.default_rng(seed=9)
    window = rng.normal(size=(150, 4))
    test_point = rng.normal(size=4)

    naive = NaiveBaseline()
    naive.fit(window)

    drop_zero = AnomalyWithinBaseline(drop_fraction=0.0)
    drop_zero.fit(window)

    assert drop_zero.score(test_point) == pytest.approx(
        naive.score(test_point), rel=1e-12, abs=1e-12
    )


def test_anomaly_within_score_batch_matches_score_row_by_row() -> None:
    rng = np.random.default_rng(seed=4)
    window = rng.normal(size=(200, 4))
    test_batch = rng.normal(size=(20, 4))

    baseline = AnomalyWithinBaseline(drop_fraction=0.1)
    baseline.fit(window)

    batch_scores = baseline.score_batch(test_batch)
    row_scores = np.array([baseline.score(row) for row in test_batch])
    np.testing.assert_allclose(batch_scores, row_scores, rtol=1e-12, atol=1e-12)


def test_anomaly_within_score_before_fit_raises() -> None:
    baseline = AnomalyWithinBaseline()
    with pytest.raises(BaselineNotFittedError):
        baseline.score(np.zeros(3))


def test_anomaly_within_fit_rejects_empty_window() -> None:
    baseline = AnomalyWithinBaseline()
    with pytest.raises(EmptyLearningWindowError):
        baseline.fit(np.zeros((0, 5)))


def test_anomaly_within_fit_rejects_non_finite_values() -> None:
    baseline = AnomalyWithinBaseline()
    with pytest.raises(InvalidObservationError):
        baseline.fit(np.array([[1.0, np.nan], [4.0, 5.0]]))


def test_anomaly_within_drop_fraction_too_large_raises() -> None:
    baseline = AnomalyWithinBaseline(drop_fraction=0.99)
    with pytest.raises(EmptyLearningWindowError):
        baseline.fit(np.ones((1, 3)))


@pytest.mark.parametrize(
    "bad_drop", [-0.1, 1.0, 1.5, float("nan"), float("inf")]
)
def test_anomaly_within_constructor_rejects_invalid_drop_fraction(bad_drop: float) -> None:
    with pytest.raises(InvalidParameterError):
        AnomalyWithinBaseline(drop_fraction=bad_drop)


def test_anomaly_within_score_rejects_dimension_mismatch() -> None:
    baseline = AnomalyWithinBaseline()
    baseline.fit(np.random.default_rng(0).normal(size=(50, 5)))
    with pytest.raises(DimensionMismatchError):
        baseline.score(np.zeros(3))
