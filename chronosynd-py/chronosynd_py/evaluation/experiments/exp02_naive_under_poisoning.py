"""Experiment 02, naive baseline under poisoning. Measures how
NaiveBaseline's score of the target collapses as the adversary's budget
grows. The sharp drop motivates the need for a robust baseline"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from chronosynd_py.baseline.naive import NaiveBaseline
from chronosynd_py.evaluation.harness import (
    EstimatorFactory,
    SweepSample,
    run_poisoning_sweep,
)
from chronosynd_py.evaluation.workloads import isotropic_gaussian, jittered_target

FEATURE_DIM = 8
TARGET_DISTANCE = 4.0
BUDGET_GRID: tuple[float, ...] = (
    0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30,
)
SEEDS: tuple[int, ...] = tuple(range(15))
TARGET_FPR = 0.05
LEARNING_WINDOW = 500
BENIGN_TEST = 500
MALICIOUS_TEST = 200
MALICIOUS_JITTER = 0.2


def _results_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    results_dir = repo_root / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / "exp02_naive_under_poisoning.csv"


def _estimator_factories() -> dict[str, EstimatorFactory]:
    return {"naive": NaiveBaseline}


def run() -> list[SweepSample]:
    target = np.full(FEATURE_DIM, TARGET_DISTANCE)
    return run_poisoning_sweep(
        benign_source=isotropic_gaussian(feature_dim=FEATURE_DIM),
        malicious_source=jittered_target(target, jitter_scale=MALICIOUS_JITTER),
        target_attack_behavior=target,
        estimator_factories=_estimator_factories(),
        budget_grid=BUDGET_GRID,
        seeds=SEEDS,
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
                "target_score",
                "median_benign_score",
                "detection_rate",
                "false_positive_rate",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.estimator_name,
                    sample.budget_fraction,
                    sample.seed,
                    sample.target_score,
                    sample.median_benign_score,
                    sample.detection_rate,
                    sample.false_positive_rate,
                ]
            )


def main() -> None:
    samples = run()
    path = _results_path()
    write_csv(samples, path)
    print(f"exp02_naive_under_poisoning: wrote {len(samples)} rows to {path}")


if __name__ == "__main__":
    main()
