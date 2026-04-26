"""Unit tests for the pre-seed poisoning attack"""

from __future__ import annotations

import numpy as np
import pytest

from chronosynd_py.baseline.naive import NaiveBaseline
from chronosynd_py.core import (
    DimensionMismatchError,
    InvalidObservationError,
    InvalidParameterError,
    PoisonedTrace,
)
from chronosynd_py.evaluation.attacks.pre_seed import inject_pre_seed


def _benign_window(
    *, samples: int = 200, features: int = 4, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.normal(loc=0.0, scale=1.0, size=(samples, features))


def test_zero_budget_returns_benign_trace_unchanged() -> None:
    benign = _benign_window()
    target = np.full(benign.shape[1], 10.0)

    result = inject_pre_seed(benign, target, budget_fraction=0.0)

    np.testing.assert_array_equal(result.observations, benign)
    assert result.adversarial_count == 0
    assert result.sample_count == benign.shape[0]


def test_small_budget_injects_expected_count() -> None:
    benign = _benign_window(samples=100)
    target = np.full(benign.shape[1], 5.0)

    result = inject_pre_seed(benign, target, budget_fraction=0.1)

    assert result.sample_count == benign.shape[0] + 10
    assert result.adversarial_count == 10


def test_full_budget_doubles_the_window() -> None:
    benign = _benign_window(samples=50)
    target = np.full(benign.shape[1], 7.0)

    result = inject_pre_seed(benign, target, budget_fraction=1.0)

    assert result.sample_count == 2 * benign.shape[0]
    assert result.adversarial_count == benign.shape[0]


def test_injected_rows_are_exact_copies_without_jitter() -> None:
    benign = _benign_window(samples=100)
    target = np.array([1.0, 2.0, 3.0, 4.0])

    result = inject_pre_seed(benign, target, budget_fraction=0.1)

    adversarial_rows = result.observations[result.is_adversarial]
    for row in adversarial_rows:
        np.testing.assert_array_equal(row, target)


def test_jitter_produces_near_but_not_exact_injections() -> None:
    benign = _benign_window(samples=100)
    target = np.array([1.0, 2.0, 3.0, 4.0])
    rng = np.random.default_rng(seed=1)

    result = inject_pre_seed(
        benign, target, budget_fraction=0.2, jitter_scale=0.01, rng=rng
    )

    adversarial_rows = result.observations[result.is_adversarial]
    distances = np.linalg.norm(adversarial_rows - target, axis=1)
    assert np.all(distances > 0.0)
    assert np.all(distances < 0.2)


def test_reproducible_under_seeded_rng() -> None:
    benign = _benign_window()
    target = np.full(benign.shape[1], 3.0)

    first = inject_pre_seed(
        benign,
        target,
        budget_fraction=0.15,
        jitter_scale=0.05,
        rng=np.random.default_rng(seed=99),
    )
    second = inject_pre_seed(
        benign,
        target,
        budget_fraction=0.15,
        jitter_scale=0.05,
        rng=np.random.default_rng(seed=99),
    )

    np.testing.assert_array_equal(first.observations, second.observations)
    np.testing.assert_array_equal(first.is_adversarial, second.is_adversarial)


def test_different_seeds_produce_different_permutations() -> None:
    benign = _benign_window()
    target = np.full(benign.shape[1], 3.0)

    first = inject_pre_seed(
        benign, target, budget_fraction=0.2, rng=np.random.default_rng(seed=1)
    )
    second = inject_pre_seed(
        benign, target, budget_fraction=0.2, rng=np.random.default_rng(seed=2)
    )

    assert not np.array_equal(first.is_adversarial, second.is_adversarial)


def test_returns_poisoned_trace_instance() -> None:
    benign = _benign_window()
    target = np.zeros(benign.shape[1])
    result = inject_pre_seed(benign, target, budget_fraction=0.05)
    assert isinstance(result, PoisonedTrace)


def test_rejects_1d_benign_trace() -> None:
    with pytest.raises(DimensionMismatchError):
        inject_pre_seed(np.zeros(10), np.zeros(5), budget_fraction=0.1)


def test_rejects_2d_target() -> None:
    with pytest.raises(DimensionMismatchError):
        inject_pre_seed(np.zeros((10, 5)), np.zeros((1, 5)), budget_fraction=0.1)


def test_rejects_target_with_wrong_feature_dim() -> None:
    with pytest.raises(DimensionMismatchError):
        inject_pre_seed(np.zeros((10, 5)), np.zeros(3), budget_fraction=0.1)


@pytest.mark.parametrize("bad_budget", [-0.01, -1.0, 1.0001, 10.0, float("nan")])
def test_rejects_out_of_range_budget(bad_budget: float) -> None:
    with pytest.raises(InvalidParameterError):
        inject_pre_seed(np.zeros((10, 5)), np.zeros(5), budget_fraction=bad_budget)


@pytest.mark.parametrize("bad_jitter", [-0.01, -1.0, float("nan"), float("inf")])
def test_rejects_invalid_jitter_scale(bad_jitter: float) -> None:
    with pytest.raises(InvalidParameterError):
        inject_pre_seed(
            np.zeros((10, 5)),
            np.zeros(5),
            budget_fraction=0.1,
            jitter_scale=bad_jitter,
        )


def test_rejects_non_finite_benign_trace() -> None:
    benign = np.zeros((10, 3))
    benign[0, 0] = np.nan
    with pytest.raises(InvalidObservationError):
        inject_pre_seed(benign, np.zeros(3), budget_fraction=0.1)


def test_rejects_non_finite_target() -> None:
    with pytest.raises(InvalidObservationError):
        inject_pre_seed(
            np.zeros((10, 3)),
            np.array([1.0, np.inf, 3.0]),
            budget_fraction=0.1,
        )


def test_attack_lowers_naive_baseline_score_for_the_target() -> None:
    """Paper's motivation result in miniature. Clean baseline scores the
    distant target as highly anomalous; modest poisoning collapses the score
    """
    benign = _benign_window(samples=200, seed=7)
    target = np.full(benign.shape[1], 8.0)

    clean = NaiveBaseline()
    clean.fit(benign)
    clean_target_score = clean.score(target)

    poisoned = inject_pre_seed(
        benign, target, budget_fraction=0.15, rng=np.random.default_rng(seed=11)
    )
    attacked = NaiveBaseline()
    attacked.fit(poisoned.observations)
    attacked_target_score = attacked.score(target)

    assert attacked_target_score < clean_target_score / 5.0
