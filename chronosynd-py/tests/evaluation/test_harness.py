"""Unit tests for the poisoning-sweep harness"""

from __future__ import annotations

from functools import partial

import numpy as np
import pytest

from chronosynd_py.baseline.naive import NaiveBaseline
from chronosynd_py.baseline.sediment import Sediment
from chronosynd_py.core import InvalidParameterError
from chronosynd_py.evaluation.harness import SweepSample, run_poisoning_sweep
from chronosynd_py.evaluation.workloads import isotropic_gaussian, jittered_target


def _small_sweep(
    *,
    target_distance: float = 6.0,
    malicious_jitter: float = 0.1,
    **overrides: object,
) -> list[SweepSample]:
    target = np.full(4, target_distance)
    defaults: dict[str, object] = {
        "benign_source": isotropic_gaussian(feature_dim=4),
        "malicious_source": jittered_target(target, jitter_scale=malicious_jitter),
        "target_attack_behavior": target,
        "estimator_factories": {
            "naive": NaiveBaseline,
            "sediment": partial(Sediment, trim_fraction=0.3),
        },
        "budget_grid": (0.0, 0.15),
        "seeds": (1, 2),
        "learning_window_size": 200,
        "benign_test_size": 200,
        "malicious_test_size": 100,
        "target_fpr": 0.05,
    }
    defaults.update(overrides)
    return run_poisoning_sweep(**defaults)  # type: ignore[arg-type]


def test_produces_one_sample_per_grid_cell() -> None:
    samples = _small_sweep()
    # 2 estimators * 2 budgets * 2 seeds = 8 samples
    assert len(samples) == 8


def test_sample_names_and_budgets_match_grid() -> None:
    samples = _small_sweep()
    names = {s.estimator_name for s in samples}
    budgets = {s.budget_fraction for s in samples}
    seeds = {s.seed for s in samples}
    assert names == {"naive", "sediment"}
    assert budgets == {0.0, 0.15}
    assert seeds == {1, 2}


def _mean_detection_rate(
    samples: list[SweepSample], name: str, budget: float
) -> float:
    return float(
        np.mean(
            [
                s.detection_rate
                for s in samples
                if s.estimator_name == name and s.budget_fraction == budget
            ]
        )
    )


def test_detection_rates_are_in_valid_range() -> None:
    samples = _small_sweep(budget_grid=(0.0, 0.15, 0.30), seeds=tuple(range(5)))
    for sample in samples:
        assert 0.0 <= sample.detection_rate <= 1.0
        assert 0.0 <= sample.false_positive_rate <= 1.0


def test_sediment_detection_is_at_least_as_good_as_naive_under_heavy_poisoning() -> None:
    """Sediment must not be worse than naive under heavy poisoning when
    averaged across seeds, even if the synthetic config keeps both rates high
    """
    samples = _small_sweep(
        target_distance=3.0,
        malicious_jitter=0.05,
        budget_grid=(0.30,),
        seeds=tuple(range(10)),
    )
    naive_rate = _mean_detection_rate(samples, "naive", 0.30)
    sediment_rate = _mean_detection_rate(samples, "sediment", 0.30)
    assert sediment_rate >= naive_rate - 0.05


def test_clean_condition_fpr_is_close_to_target() -> None:
    """Measured FPR on unpoisoned learning windows must land near target_fpr"""
    samples = _small_sweep(budget_grid=(0.0,), seeds=tuple(range(10)), target_fpr=0.05)
    for sample in samples:
        assert sample.false_positive_rate == pytest.approx(0.05, abs=0.03)


def test_score_level_metrics_are_recorded() -> None:
    samples = _small_sweep(budget_grid=(0.0,), seeds=(1,))
    sample = samples[0]
    assert sample.target_score > 0.0
    assert sample.median_benign_score > 0.0
    assert sample.median_malicious_score > 0.0


def _mean_target_score_ratio(
    samples: list[SweepSample], name: str, budget: float
) -> float:
    rows = [
        s.target_score / s.median_benign_score
        for s in samples
        if s.estimator_name == name and s.budget_fraction == budget
    ]
    return float(np.mean(rows))


def test_sediment_holds_score_separation_under_poisoning_better_than_naive() -> None:
    """Score-level main result. Sediment keeps target_score / benign_score
    closer to its clean ratio than naive does, invariant to thresholding
    """
    samples = _small_sweep(
        target_distance=3.0,
        malicious_jitter=0.05,
        budget_grid=(0.0, 0.15),
        seeds=tuple(range(8)),
    )
    naive_ratio_drop = _mean_target_score_ratio(samples, "naive", 0.0) / max(
        _mean_target_score_ratio(samples, "naive", 0.15), 1e-9
    )
    sediment_ratio_drop = _mean_target_score_ratio(samples, "sediment", 0.0) / max(
        _mean_target_score_ratio(samples, "sediment", 0.15), 1e-9
    )
    assert naive_ratio_drop > sediment_ratio_drop


def test_is_reproducible_under_fixed_seeds() -> None:
    first = _small_sweep()
    second = _small_sweep()
    assert first == second


def test_rejects_empty_estimator_factories() -> None:
    with pytest.raises(InvalidParameterError):
        _small_sweep(estimator_factories={})


def test_rejects_empty_budget_grid() -> None:
    with pytest.raises(InvalidParameterError):
        _small_sweep(budget_grid=())


def test_rejects_empty_seeds() -> None:
    with pytest.raises(InvalidParameterError):
        _small_sweep(seeds=())


@pytest.mark.parametrize("bad_size", [0, -1])
def test_rejects_invalid_sample_sizes(bad_size: int) -> None:
    with pytest.raises(InvalidParameterError):
        _small_sweep(learning_window_size=bad_size)
    with pytest.raises(InvalidParameterError):
        _small_sweep(benign_test_size=bad_size)
    with pytest.raises(InvalidParameterError):
        _small_sweep(malicious_test_size=bad_size)
