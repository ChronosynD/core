"""Unit tests for the targeted (white-box) poisoning attack"""

from __future__ import annotations

import numpy as np
import pytest

from chronosynd_py.baseline.naive import NaiveBaseline
from chronosynd_py.baseline.sediment import Sediment
from chronosynd_py.core import (
    DimensionMismatchError,
    InvalidObservationError,
    InvalidParameterError,
)
from chronosynd_py.evaluation.attacks.pre_seed import inject_pre_seed
from chronosynd_py.evaluation.attacks.targeted import inject_targeted


def _benign_window(
    *, samples: int = 500, features: int = 4, seed: int = 42
) -> np.ndarray:
    rng = np.random.default_rng(seed=seed)
    return rng.normal(loc=0.0, scale=1.0, size=(samples, features))


def test_zero_budget_returns_benign_trace_unchanged() -> None:
    benign = _benign_window()
    target = np.full(benign.shape[1], 5.0)

    result = inject_targeted(benign, target, budget_fraction=0.0)

    np.testing.assert_array_equal(result.observations, benign)
    assert result.adversarial_count == 0


def test_against_naive_defender_injects_at_target_value() -> None:
    benign = _benign_window(samples=200)
    target = np.full(benign.shape[1], 7.0)

    result = inject_targeted(
        benign,
        target,
        budget_fraction=0.1,
        defender_trim_fraction=0.0,
        rng=np.random.default_rng(seed=1),
    )

    adversarial_rows = result.observations[result.is_adversarial]
    expected = np.broadcast_to(target, adversarial_rows.shape)
    np.testing.assert_allclose(adversarial_rows, expected)


def test_against_sediment_defender_picks_value_between_median_and_target() -> None:
    # The white-box adversary's optimal placement against a trimmed-mean
    # defender lands somewhere between the benign median and the target. The
    # exact value depends on budget and trim fraction. The contract checked
    # here is just that placement is bounded by those two reference points
    rng = np.random.default_rng(seed=3)
    benign = rng.normal(size=(500, 4))
    target = np.full(4, 5.0)
    benign_median = np.median(benign, axis=0)

    result = inject_targeted(
        benign,
        target,
        budget_fraction=0.10,
        defender_trim_fraction=0.30,
        rng=np.random.default_rng(seed=4),
    )

    adversarial = result.observations[result.is_adversarial][0]
    assert np.all(adversarial >= benign_median - 1e-9)
    assert np.all(adversarial <= target + 1e-9)


def test_targeted_is_at_least_as_strong_as_pre_seed_against_sediment() -> None:
    # The grid-search white-box adversary considers the target itself as a
    # candidate placement, so it cannot do worse than pre_seed which always
    # injects at the target. This is the contract that motivates the attack
    benign_seed = 11
    poison_seed = 17
    feature_dim = 4
    target = np.full(feature_dim, 4.0)
    trim = 0.30

    for budget in (0.05, 0.10, 0.15, 0.18, 0.25):
        benign = _benign_window(samples=500, features=feature_dim, seed=benign_seed)
        pre_seeded = inject_pre_seed(
            benign, target, budget_fraction=budget, rng=np.random.default_rng(poison_seed)
        )
        targeted_attack = inject_targeted(
            benign,
            target,
            budget_fraction=budget,
            defender_trim_fraction=trim,
            rng=np.random.default_rng(poison_seed),
        )

        a = Sediment(trim_fraction=trim)
        a.fit(pre_seeded.observations)
        b = Sediment(trim_fraction=trim)
        b.fit(targeted_attack.observations)

        # Targeted should reduce the score by at least as much as pre_seed.
        # Allow a tiny tolerance because the trim count rounds to integers
        assert b.score(target) <= a.score(target) + 1e-6, (
            f"at budget={budget}, targeted={b.score(target)} vs "
            f"pre_seed={a.score(target)}"
        )


def test_targeted_picks_interior_placement_at_low_budget_against_sediment() -> None:
    # The grid-search white-box adversary picks a value strictly between
    # the benign median and the target at budgets below the trim/2
    # threshold. That is the regime where surviving-region placement
    # measurably differs from at-target placement even if the fitted
    # moments end up equivalent
    rng = np.random.default_rng(seed=42)
    benign = rng.normal(loc=0.0, scale=1.0, size=(500, 4))
    target = np.full(4, 4.0)

    result = inject_targeted(
        benign,
        target,
        budget_fraction=0.05,
        defender_trim_fraction=0.30,
        rng=np.random.default_rng(seed=99),
    )

    adversarial_value = result.observations[result.is_adversarial][0]
    benign_median = np.median(benign, axis=0)
    assert np.all(adversarial_value > benign_median)
    assert np.all(adversarial_value < target)


def test_full_takeover_at_budget_one() -> None:
    benign = _benign_window(samples=50)
    target = np.full(benign.shape[1], 3.0)

    result = inject_targeted(benign, target, budget_fraction=1.0)

    assert result.adversarial_count == 50


def test_random_permutation_uses_provided_rng() -> None:
    benign = _benign_window(samples=200)
    target = np.full(benign.shape[1], 5.0)

    first = inject_targeted(
        benign, target, budget_fraction=0.1, rng=np.random.default_rng(7)
    )
    second = inject_targeted(
        benign, target, budget_fraction=0.1, rng=np.random.default_rng(7)
    )
    np.testing.assert_array_equal(first.is_adversarial, second.is_adversarial)


def test_targeted_with_zero_trim_matches_target_injection() -> None:
    # When the defender does no trimming, the optimal placement is the
    # target itself. The result should match what NaiveBaseline sees under
    # any target-valued attack
    benign = _benign_window(samples=200)
    target = np.full(benign.shape[1], 5.0)

    result = inject_targeted(
        benign,
        target,
        budget_fraction=0.1,
        defender_trim_fraction=0.0,
        rng=np.random.default_rng(seed=2),
    )

    naive = NaiveBaseline()
    naive.fit(result.observations)
    target_score = naive.score(target)
    assert target_score < 100.0


def test_rejects_invalid_budget() -> None:
    benign = _benign_window()
    target = np.zeros(benign.shape[1])
    with pytest.raises(InvalidParameterError):
        inject_targeted(benign, target, budget_fraction=-0.1)
    with pytest.raises(InvalidParameterError):
        inject_targeted(benign, target, budget_fraction=1.5)


def test_rejects_invalid_defender_trim_fraction() -> None:
    benign = _benign_window()
    target = np.zeros(benign.shape[1])
    with pytest.raises(InvalidParameterError):
        inject_targeted(benign, target, budget_fraction=0.1, defender_trim_fraction=-0.1)
    with pytest.raises(InvalidParameterError):
        inject_targeted(benign, target, budget_fraction=0.1, defender_trim_fraction=1.0)


def test_rejects_dimension_mismatch() -> None:
    benign = _benign_window(features=4)
    target = np.zeros(3)
    with pytest.raises(DimensionMismatchError):
        inject_targeted(benign, target, budget_fraction=0.1)


def test_rejects_non_finite_inputs() -> None:
    benign = _benign_window()
    bad_benign = benign.copy()
    bad_benign[0, 0] = np.nan
    target = np.zeros(benign.shape[1])
    with pytest.raises(InvalidObservationError):
        inject_targeted(bad_benign, target, budget_fraction=0.1)
    bad_target = target.copy()
    bad_target[0] = np.inf
    with pytest.raises(InvalidObservationError):
        inject_targeted(benign, bad_target, budget_fraction=0.1)
