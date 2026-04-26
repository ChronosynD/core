"""Unit tests for the Sediment estimator"""

from __future__ import annotations

import numpy as np
import pytest

from chronosynd_py.baseline.naive import NaiveBaseline
from chronosynd_py.baseline.sediment import Sediment
from chronosynd_py.core import (
    BaselineNotFittedError,
    DimensionMismatchError,
    EmptyLearningWindowError,
    InvalidObservationError,
    InvalidParameterError,
)
from chronosynd_py.evaluation.attacks.pre_seed import inject_pre_seed


def test_constructs_with_defaults() -> None:
    baseline = Sediment()
    assert baseline.trim_fraction == 0.1


def test_constructs_with_custom_trim_fraction() -> None:
    baseline = Sediment(trim_fraction=0.25)
    assert baseline.trim_fraction == 0.25


@pytest.mark.parametrize(
    "bad_trim",
    [-0.01, -1.0, 1.0, 1.5, float("nan"), float("inf")],
)
def test_constructor_rejects_invalid_trim_fraction(bad_trim: float) -> None:
    with pytest.raises(InvalidParameterError):
        Sediment(trim_fraction=bad_trim)


def test_constructor_rejects_invalid_epsilon() -> None:
    with pytest.raises(InvalidParameterError):
        Sediment(epsilon=0.0)


def test_trim_fraction_zero_matches_naive_baseline() -> None:
    rng = np.random.default_rng(seed=42)
    window = rng.normal(size=(150, 4))
    probe = rng.normal(size=4)

    naive = NaiveBaseline()
    naive.fit(window)

    sediment = Sediment(trim_fraction=0.0)
    sediment.fit(window)

    assert sediment.score(probe) == pytest.approx(naive.score(probe), rel=1e-12)


def test_scores_observation_near_mean_as_low() -> None:
    rng = np.random.default_rng(seed=7)
    window = rng.normal(size=(200, 5))

    baseline = Sediment(trim_fraction=0.1)
    baseline.fit(window)

    assert baseline.score(window.mean(axis=0)) < 1.0


def test_scores_wildly_anomalous_observation_as_high() -> None:
    rng = np.random.default_rng(seed=7)
    window = rng.normal(size=(200, 5))

    baseline = Sediment(trim_fraction=0.1)
    baseline.fit(window)

    assert baseline.score(np.full(5, 100.0)) > 1000.0


def test_score_grows_monotonically_with_distance() -> None:
    rng = np.random.default_rng(seed=7)
    window = rng.normal(size=(100, 3))

    baseline = Sediment(trim_fraction=0.1)
    baseline.fit(window)

    close = baseline.score(np.array([0.1, 0.1, 0.1]))
    far = baseline.score(np.array([5.0, 5.0, 5.0]))
    farther = baseline.score(np.array([50.0, 50.0, 50.0]))
    assert close < far < farther


def test_score_before_fit_raises() -> None:
    baseline = Sediment()
    with pytest.raises(BaselineNotFittedError):
        baseline.score(np.zeros(3))


def test_fit_rejects_1d_input() -> None:
    baseline = Sediment()
    with pytest.raises(DimensionMismatchError):
        baseline.fit(np.zeros(10))


def test_fit_rejects_empty_window() -> None:
    baseline = Sediment()
    with pytest.raises(EmptyLearningWindowError):
        baseline.fit(np.zeros((0, 5)))


def test_fit_rejects_non_finite_values() -> None:
    baseline = Sediment()
    with pytest.raises(InvalidObservationError):
        baseline.fit(np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]))


def test_fit_rejects_trim_too_aggressive_for_window_size() -> None:
    baseline = Sediment(trim_fraction=0.9)
    with pytest.raises(EmptyLearningWindowError):
        baseline.fit(np.zeros((2, 3)))


def test_score_rejects_dimension_mismatch() -> None:
    baseline = Sediment()
    baseline.fit(np.zeros((10, 5)))
    with pytest.raises(DimensionMismatchError):
        baseline.score(np.zeros(3))


def test_trimming_recovers_tight_fit_after_outlier_contamination() -> None:
    """Naive under-flags the outliers it was trained on. Sediment trims them
    out and correctly flags the same point as highly anomalous
    """
    rng = np.random.default_rng(seed=3)
    benign = rng.normal(loc=0.0, scale=1.0, size=(100, 2))
    outliers = np.full((10, 2), 50.0)
    mixed = np.vstack([benign, outliers])

    naive_on_mixed = NaiveBaseline()
    naive_on_mixed.fit(mixed)
    sediment = Sediment(trim_fraction=0.3)
    sediment.fit(mixed)

    outlier_point = np.array([50.0, 50.0])
    sediment_score = sediment.score(outlier_point)
    naive_mixed_score = naive_on_mixed.score(outlier_point)

    # Sediment flags the outlier orders of magnitude harder than the contaminated naive fit
    assert sediment_score > naive_mixed_score * 50.0


def test_sediment_resists_pre_seed_poisoning_at_matched_budget() -> None:
    """The paper's main result in miniature. Sediment with trim_fraction at
    least twice the budget keeps the target score high while naive collapses
    """
    rng = np.random.default_rng(seed=42)
    benign = rng.normal(size=(200, 4))
    target = np.full(4, 8.0)

    poisoned = inject_pre_seed(
        benign, target, budget_fraction=0.15, rng=np.random.default_rng(seed=11)
    )

    clean_naive = NaiveBaseline()
    clean_naive.fit(benign)
    clean_score = clean_naive.score(target)

    poisoned_naive = NaiveBaseline()
    poisoned_naive.fit(poisoned.observations)
    poisoned_naive_score = poisoned_naive.score(target)

    sediment = Sediment(trim_fraction=0.3)
    sediment.fit(poisoned.observations)
    sediment_score = sediment.score(target)

    # Sanity check: the poisoned naive score must drop meaningfully or the attack is not biting
    assert poisoned_naive_score < clean_score / 5.0

    # The main claim: Sediment's poisoned score must stay close to the clean naive score
    assert sediment_score > 5.0 * poisoned_naive_score
    assert sediment_score > clean_score / 3.0


def test_sediment_clean_fpr_comparable_to_naive() -> None:
    """Sediment must not pay a meaningful FPR overhead on unpoisoned data"""
    rng = np.random.default_rng(seed=99)
    window = rng.normal(size=(500, 6))
    held_out = rng.normal(size=6)

    naive = NaiveBaseline()
    naive.fit(window)
    sediment = Sediment(trim_fraction=0.1)
    sediment.fit(window)

    naive_score = naive.score(held_out)
    sediment_score = sediment.score(held_out)

    # Naive must score a benign held-out point as non-trivial, otherwise the
    # ratio comparison below is meaningless and the test would pass vacuously
    assert naive_score > 1e-6
    # Scores on a typical benign observation must agree within a factor of two
    ratio = sediment_score / naive_score
    assert 0.5 < ratio < 2.0
