"""Unit tests for the burst poisoning attack"""

from __future__ import annotations

import numpy as np
import pytest

from chronosynd_py.core import (
    DimensionMismatchError,
    InvalidObservationError,
    InvalidParameterError,
)
from chronosynd_py.evaluation.attacks.burst import inject_burst


def _benign_window(
    *, samples: int = 200, features: int = 4, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.normal(loc=0.0, scale=1.0, size=(samples, features))


def test_zero_budget_returns_benign_trace_unchanged() -> None:
    benign = _benign_window()
    target = np.full(benign.shape[1], 10.0)

    result = inject_burst(benign, target, budget_fraction=0.0)

    np.testing.assert_array_equal(result.observations, benign)
    assert result.adversarial_count == 0
    assert result.sample_count == benign.shape[0]


def test_small_budget_injects_expected_count() -> None:
    benign = _benign_window(samples=100)
    target = np.full(benign.shape[1], 5.0)

    result = inject_burst(benign, target, budget_fraction=0.1)

    assert result.adversarial_count == 10
    assert result.sample_count == 100


def test_injections_are_contiguous() -> None:
    benign = _benign_window(samples=200)
    target = np.full(benign.shape[1], 5.0)

    result = inject_burst(
        benign,
        target,
        budget_fraction=0.1,
        rng=np.random.default_rng(seed=1),
    )

    adversarial_indices = np.flatnonzero(result.is_adversarial)
    assert adversarial_indices.size == 20
    assert np.all(np.diff(adversarial_indices) == 1)


def test_burst_position_zero_places_burst_at_start() -> None:
    benign = _benign_window(samples=100)
    target = np.full(benign.shape[1], 5.0)

    result = inject_burst(
        benign,
        target,
        budget_fraction=0.1,
        burst_position=0.0,
    )

    assert bool(result.is_adversarial[0]) is True
    assert bool(result.is_adversarial[9]) is True
    assert bool(result.is_adversarial[10]) is False


def test_burst_position_one_places_burst_at_end() -> None:
    benign = _benign_window(samples=100)
    target = np.full(benign.shape[1], 5.0)

    result = inject_burst(
        benign,
        target,
        budget_fraction=0.1,
        burst_position=1.0,
    )

    assert bool(result.is_adversarial[-1]) is True
    assert bool(result.is_adversarial[-10]) is True
    assert bool(result.is_adversarial[-11]) is False


def test_full_takeover_at_budget_one() -> None:
    benign = _benign_window(samples=50)
    target = np.full(benign.shape[1], 5.0)

    result = inject_burst(benign, target, budget_fraction=1.0)

    assert result.adversarial_count == 50
    assert np.all(result.observations == target)


def test_jitter_makes_injections_non_identical() -> None:
    benign = _benign_window(samples=100)
    target = np.full(benign.shape[1], 5.0)

    result = inject_burst(
        benign,
        target,
        budget_fraction=0.1,
        burst_position=0.0,
        jitter_scale=0.5,
        rng=np.random.default_rng(seed=2),
    )

    injected = result.observations[:10]
    assert not np.allclose(injected, injected[0])


def test_random_position_uses_provided_rng() -> None:
    benign = _benign_window(samples=200)
    target = np.full(benign.shape[1], 5.0)

    first = inject_burst(
        benign, target, budget_fraction=0.1, rng=np.random.default_rng(seed=7)
    )
    second = inject_burst(
        benign, target, budget_fraction=0.1, rng=np.random.default_rng(seed=7)
    )
    np.testing.assert_array_equal(first.is_adversarial, second.is_adversarial)


def test_rejects_invalid_budget() -> None:
    benign = _benign_window()
    target = np.full(benign.shape[1], 0.0)
    with pytest.raises(InvalidParameterError):
        inject_burst(benign, target, budget_fraction=-0.1)
    with pytest.raises(InvalidParameterError):
        inject_burst(benign, target, budget_fraction=1.5)


def test_rejects_invalid_burst_position() -> None:
    benign = _benign_window()
    target = np.full(benign.shape[1], 0.0)
    with pytest.raises(InvalidParameterError):
        inject_burst(benign, target, budget_fraction=0.1, burst_position=-0.1)
    with pytest.raises(InvalidParameterError):
        inject_burst(benign, target, budget_fraction=0.1, burst_position=1.5)


def test_rejects_dimension_mismatch() -> None:
    benign = _benign_window(features=4)
    target = np.full(3, 0.0)
    with pytest.raises(DimensionMismatchError):
        inject_burst(benign, target, budget_fraction=0.1)


def test_rejects_non_finite_inputs() -> None:
    benign = _benign_window()
    bad_benign = benign.copy()
    bad_benign[0, 0] = np.nan
    target = np.zeros(benign.shape[1])
    with pytest.raises(InvalidObservationError):
        inject_burst(bad_benign, target, budget_fraction=0.1)
    bad_target = target.copy()
    bad_target[0] = np.inf
    with pytest.raises(InvalidObservationError):
        inject_burst(benign, bad_target, budget_fraction=0.1)
