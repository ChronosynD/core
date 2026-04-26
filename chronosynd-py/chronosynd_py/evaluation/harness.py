"""Poisoning-sweep harness for the paper's experiments. Runs the full grid
of (estimator, poisoning budget, seed) and produces one `SweepSample` per
cell that downstream experiment scripts flatten into CSVs"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from chronosynd_py.baseline.base import Baseline
from chronosynd_py.core import (
    InvalidParameterError,
    Observation,
    ObservationBatch,
    PoisonedTrace,
)
from chronosynd_py.evaluation.attacks.pre_seed import inject_pre_seed
from chronosynd_py.evaluation.metrics import compute_detection_metrics
from chronosynd_py.evaluation.workloads import WorkloadFactory

EstimatorFactory = Callable[[], Baseline]
AttackInjector = Callable[
    [ObservationBatch, Observation, float, np.random.Generator],
    PoisonedTrace,
]


@dataclass(frozen=True, slots=True)
class SweepSample:
    """One row of the poisoning sweep results table. A single (estimator,
    budget, seed) cell carrying both the ROC-style detection view and the
    threshold-invariant score-level view used for score-collapse figures"""

    estimator_name: str
    budget_fraction: float
    seed: int
    detection_rate: float
    false_positive_rate: float
    threshold: float
    target_score: float
    median_benign_score: float
    median_malicious_score: float


def _default_pre_seed_injector(
    benign: ObservationBatch,
    target: Observation,
    budget: float,
    rng: np.random.Generator,
) -> PoisonedTrace:
    return inject_pre_seed(benign, target, budget_fraction=budget, rng=rng)


def run_poisoning_sweep(
    *,
    benign_source: WorkloadFactory,
    malicious_source: WorkloadFactory,
    target_attack_behavior: Observation,
    estimator_factories: Mapping[str, EstimatorFactory],
    budget_grid: Sequence[float],
    seeds: Sequence[int],
    attack_injector: AttackInjector | None = None,
    learning_window_size: int = 500,
    benign_test_size: int = 500,
    malicious_test_size: int = 100,
    target_fpr: float = 0.05,
) -> list[SweepSample]:
    """Run the full (estimator, budget, seed) grid and return flat results.
    Each cell draws a seed-keyed benign window, poisons at the budget, fits
    a fresh estimator, and records detection metrics at the target FPR"""
    _validate_sweep_inputs(
        estimator_factories=estimator_factories,
        budget_grid=budget_grid,
        seeds=seeds,
        learning_window_size=learning_window_size,
        benign_test_size=benign_test_size,
        malicious_test_size=malicious_test_size,
        target_fpr=target_fpr,
    )

    injector: AttackInjector = (
        attack_injector if attack_injector is not None else _default_pre_seed_injector
    )

    samples: list[SweepSample] = []
    for estimator_name, make_estimator in estimator_factories.items():
        for budget_fraction in budget_grid:
            for seed in seeds:
                samples.append(
                    _run_single_cell(
                        estimator_name=estimator_name,
                        make_estimator=make_estimator,
                        budget_fraction=budget_fraction,
                        seed=seed,
                        benign_source=benign_source,
                        malicious_source=malicious_source,
                        target_attack_behavior=target_attack_behavior,
                        attack_injector=injector,
                        learning_window_size=learning_window_size,
                        benign_test_size=benign_test_size,
                        malicious_test_size=malicious_test_size,
                        target_fpr=target_fpr,
                    )
                )
    return samples


def _run_single_cell(
    *,
    estimator_name: str,
    make_estimator: EstimatorFactory,
    budget_fraction: float,
    seed: int,
    benign_source: WorkloadFactory,
    malicious_source: WorkloadFactory,
    target_attack_behavior: Observation,
    attack_injector: AttackInjector,
    learning_window_size: int,
    benign_test_size: int,
    malicious_test_size: int,
    target_fpr: float,
) -> SweepSample:
    child_seeds = np.random.SeedSequence(seed).spawn(4)
    learning_rng = np.random.default_rng(child_seeds[0])
    poison_rng = np.random.default_rng(child_seeds[1])
    benign_test_rng = np.random.default_rng(child_seeds[2])
    malicious_test_rng = np.random.default_rng(child_seeds[3])

    benign_window = benign_source(learning_rng, learning_window_size)
    poisoned = attack_injector(
        benign_window,
        target_attack_behavior,
        budget_fraction,
        poison_rng,
    )

    estimator = make_estimator()
    estimator.fit(poisoned.observations)

    benign_test = benign_source(benign_test_rng, benign_test_size)
    malicious_test = malicious_source(malicious_test_rng, malicious_test_size)

    benign_scores = estimator.score_batch(benign_test)
    malicious_scores = estimator.score_batch(malicious_test)
    target_score = estimator.score(target_attack_behavior)

    metrics = compute_detection_metrics(
        benign_scores, malicious_scores, target_fpr=target_fpr
    )

    return SweepSample(
        estimator_name=estimator_name,
        budget_fraction=budget_fraction,
        seed=seed,
        detection_rate=metrics.detection_rate,
        false_positive_rate=metrics.false_positive_rate,
        threshold=metrics.threshold,
        target_score=target_score,
        median_benign_score=float(np.median(benign_scores)),
        median_malicious_score=float(np.median(malicious_scores)),
    )


def _validate_sweep_inputs(
    *,
    estimator_factories: Mapping[str, EstimatorFactory],
    budget_grid: Sequence[float],
    seeds: Sequence[int],
    learning_window_size: int,
    benign_test_size: int,
    malicious_test_size: int,
    target_fpr: float,
) -> None:
    if len(estimator_factories) == 0:
        raise InvalidParameterError("estimator_factories must contain at least one entry")
    if len(budget_grid) == 0:
        raise InvalidParameterError("budget_grid must contain at least one budget")
    if len(seeds) == 0:
        raise InvalidParameterError("seeds must contain at least one seed")
    if learning_window_size < 1:
        raise InvalidParameterError(
            f"learning_window_size must be at least 1, got {learning_window_size}"
        )
    if benign_test_size < 1:
        raise InvalidParameterError(
            f"benign_test_size must be at least 1, got {benign_test_size}"
        )
    if malicious_test_size < 1:
        raise InvalidParameterError(
            f"malicious_test_size must be at least 1, got {malicious_test_size}"
        )
    if not (0.0 < target_fpr < 1.0):
        raise InvalidParameterError(
            f"target_fpr must be in (0.0, 1.0), got {target_fpr}"
        )
