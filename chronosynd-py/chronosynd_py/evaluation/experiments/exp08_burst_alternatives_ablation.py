"""Experiment 08, four-way ablation under burst poisoning. Mirrors exp07
but swaps the uniformly-shuffled `pre_seed` attack for the contiguous
`burst` injection so a sub-window consensus filter has a chance to fire"""

from __future__ import annotations

import csv
from functools import partial
from pathlib import Path

import numpy as np

from chronosynd_py.baseline.alternatives import (
    AnomalyWithinBaseline,
    ConsensusBaseline,
)
from chronosynd_py.baseline.naive import NaiveBaseline
from chronosynd_py.baseline.sediment import Sediment
from chronosynd_py.core import (
    Observation,
    ObservationBatch,
    PoisonedTrace,
)
from chronosynd_py.evaluation.attacks.burst import inject_burst
from chronosynd_py.evaluation.harness import (
    EstimatorFactory,
    SweepSample,
    run_poisoning_sweep,
)
from chronosynd_py.evaluation.workloads import isotropic_gaussian, jittered_target

FEATURE_DIM = 8
TARGET_DISTANCE = 4.0
SEDIMENT_TRIM_FRACTION = 0.3
ANOMALY_WITHIN_DROP_FRACTION = 0.1
CONSENSUS_SUB_WINDOWS = 4
CONSENSUS_THRESHOLD = 3.0
BUDGET_GRID: tuple[float, ...] = (0.0, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30)
SEEDS: tuple[int, ...] = tuple(range(10))
TARGET_FPR = 0.05
LEARNING_WINDOW = 500
BENIGN_TEST = 500
MALICIOUS_TEST = 200
MALICIOUS_JITTER = 0.1


def _results_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    results_dir = repo_root / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / "exp08_burst_alternatives_ablation.csv"


def _estimator_factories() -> dict[str, EstimatorFactory]:
    return {
        "naive": NaiveBaseline,
        "consensus": partial(
            ConsensusBaseline,
            n_sub_windows=CONSENSUS_SUB_WINDOWS,
            disagreement_threshold=CONSENSUS_THRESHOLD,
        ),
        "anomaly_within": partial(
            AnomalyWithinBaseline,
            drop_fraction=ANOMALY_WITHIN_DROP_FRACTION,
        ),
        "sediment": partial(Sediment, trim_fraction=SEDIMENT_TRIM_FRACTION),
    }


def _burst_injector(
    benign: ObservationBatch,
    target: Observation,
    budget: float,
    rng: np.random.Generator,
) -> PoisonedTrace:
    return inject_burst(benign, target, budget_fraction=budget, rng=rng)


def run() -> list[SweepSample]:
    target = np.full(FEATURE_DIM, TARGET_DISTANCE)
    benign = isotropic_gaussian(feature_dim=FEATURE_DIM)
    malicious = jittered_target(target, jitter_scale=MALICIOUS_JITTER)

    return run_poisoning_sweep(
        benign_source=benign,
        malicious_source=malicious,
        target_attack_behavior=target,
        estimator_factories=_estimator_factories(),
        budget_grid=BUDGET_GRID,
        seeds=SEEDS,
        attack_injector=_burst_injector,
        learning_window_size=LEARNING_WINDOW,
        benign_test_size=BENIGN_TEST,
        malicious_test_size=MALICIOUS_TEST,
        target_fpr=TARGET_FPR,
    )


def write_csv(samples: list[SweepSample], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "estimator_name",
                "budget_fraction",
                "seed",
                "detection_rate",
                "false_positive_rate",
                "threshold",
                "target_score",
                "median_benign_score",
                "median_malicious_score",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.estimator_name,
                    sample.budget_fraction,
                    sample.seed,
                    sample.detection_rate,
                    sample.false_positive_rate,
                    sample.threshold,
                    sample.target_score,
                    sample.median_benign_score,
                    sample.median_malicious_score,
                ]
            )


def main() -> None:
    samples = run()
    path = _results_path()
    write_csv(samples, path)
    print(f"exp08_burst_alternatives_ablation: wrote {len(samples)} rows to {path}")


if __name__ == "__main__":
    main()
