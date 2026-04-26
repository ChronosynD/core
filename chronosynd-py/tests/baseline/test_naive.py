"""Unit tests for NaiveBaseline"""

from __future__ import annotations

import numpy as np
import pytest

from chronosynd_py.baseline.naive import NaiveBaseline
from chronosynd_py.core import (
    BaselineNotFittedError,
    DimensionMismatchError,
    EmptyLearningWindowError,
    InvalidObservationError,
    InvalidParameterError,
)


def test_observation_at_the_mean_scores_near_zero() -> None:
    rng = np.random.default_rng(seed=42)
    window = rng.normal(loc=0.0, scale=1.0, size=(200, 5))

    baseline = NaiveBaseline()
    baseline.fit(window)

    score_at_mean = baseline.score(window.mean(axis=0))
    assert score_at_mean < 0.1


def test_clearly_anomalous_observation_scores_high() -> None:
    rng = np.random.default_rng(seed=42)
    window = rng.normal(loc=0.0, scale=1.0, size=(200, 5))

    baseline = NaiveBaseline()
    baseline.fit(window)

    wildly_anomalous = np.full(5, 100.0)
    assert baseline.score(wildly_anomalous) > 1000.0


def test_score_grows_monotonically_with_distance_from_mean() -> None:
    rng = np.random.default_rng(seed=7)
    window = rng.normal(size=(100, 3))

    baseline = NaiveBaseline()
    baseline.fit(window)

    close = baseline.score(np.array([0.1, 0.1, 0.1]))
    far = baseline.score(np.array([5.0, 5.0, 5.0]))
    farther = baseline.score(np.array([50.0, 50.0, 50.0]))
    assert close < far < farther


def test_score_before_fit_raises() -> None:
    baseline = NaiveBaseline()
    with pytest.raises(BaselineNotFittedError):
        baseline.score(np.zeros(3))


def test_fit_rejects_1d_input() -> None:
    baseline = NaiveBaseline()
    with pytest.raises(DimensionMismatchError):
        baseline.fit(np.zeros(10))


def test_fit_rejects_empty_window() -> None:
    baseline = NaiveBaseline()
    with pytest.raises(EmptyLearningWindowError):
        baseline.fit(np.zeros((0, 5)))


def test_fit_rejects_non_finite_values() -> None:
    baseline = NaiveBaseline()
    with pytest.raises(InvalidObservationError):
        baseline.fit(np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]))
    with pytest.raises(InvalidObservationError):
        baseline.fit(np.array([[1.0, np.inf, 3.0], [4.0, 5.0, 6.0]]))


def test_score_rejects_dimension_mismatch() -> None:
    baseline = NaiveBaseline()
    baseline.fit(np.zeros((10, 5)))
    with pytest.raises(DimensionMismatchError):
        baseline.score(np.zeros(3))


def test_score_rejects_wrong_rank() -> None:
    baseline = NaiveBaseline()
    baseline.fit(np.zeros((10, 5)))
    with pytest.raises(DimensionMismatchError):
        baseline.score(np.zeros((1, 5)))


def test_score_rejects_non_finite_observation() -> None:
    baseline = NaiveBaseline()
    baseline.fit(np.ones((10, 3)))
    with pytest.raises(InvalidObservationError):
        baseline.score(np.array([1.0, np.nan, 1.0]))


@pytest.mark.parametrize("bad_epsilon", [0.0, -1e-9, -1.0, float("nan"), float("inf")])
def test_constructor_rejects_invalid_epsilon(bad_epsilon: float) -> None:
    with pytest.raises(InvalidParameterError):
        NaiveBaseline(epsilon=bad_epsilon)


def test_single_sample_fit_falls_back_to_zero_ddof() -> None:
    baseline = NaiveBaseline()
    baseline.fit(np.array([[1.0, 2.0, 3.0]]))
    # stddev is zero for a single sample, epsilon keeps scoring finite
    assert np.isfinite(baseline.score(np.array([1.0, 2.0, 3.0])))


def test_zero_variance_feature_does_not_divide_by_zero() -> None:
    # one feature is constant across the window, a naive divide would blow up
    window = np.column_stack(
        [
            np.random.default_rng(seed=1).normal(size=100),
            np.zeros(100),
        ]
    )
    baseline = NaiveBaseline(epsilon=1e-6)
    baseline.fit(window)

    # off-constant value in the zero-variance feature should score high without raising
    score = baseline.score(np.array([0.0, 1.0]))
    assert np.isfinite(score)
    assert score > 0.0


def test_fit_is_idempotent_when_called_twice_on_same_data() -> None:
    rng = np.random.default_rng(seed=3)
    window = rng.normal(size=(50, 4))

    first = NaiveBaseline()
    first.fit(window)
    score_once = first.score(window[0])

    second = NaiveBaseline()
    second.fit(window)
    second.fit(window)
    score_twice = second.score(window[0])

    assert score_once == pytest.approx(score_twice, rel=1e-12, abs=1e-12)
