"""Experiment 05, false-positive cost of trim_fraction. Holds the window
unpoisoned and sweeps Sediment's trim against naive to measure clean-
condition FPR, the operational cost operators pay for robustness"""

from __future__ import annotations

import csv
from functools import partial
from pathlib import Path

import numpy as np

from chronosynd_py.baseline.naive import NaiveBaseline
from chronosynd_py.baseline.sediment import Sediment
from chronosynd_py.evaluation.harness import (
    EstimatorFactory,
    SweepSample,
    run_poisoning_sweep,
)
from chronosynd_py.evaluation.workloads import isotropic_gaussian, jittered_target

FEATURE_DIM = 8
TARGET_DISTANCE = 4.0
TRIM_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50)
SEEDS: tuple[int, ...] = tuple(range(20))
TARGET_FPR = 0.05
LEARNING_WINDOW = 500
BENIGN_TEST = 1000
MALICIOUS_TEST = 100
MALICIOUS_JITTER = 0.2


def _results_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    results_dir = repo_root / "evaluation" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return results_dir / "exp05_fp_cost.csv"


def _estimator_factories() -> dict[str, EstimatorFactory]:
    factories: dict[str, EstimatorFactory] = {"naive": NaiveBaseline}
    for trim in TRIM_FRACTIONS:
        name = f"sediment_trim{round(trim * 100):02d}"
        factories[name] = partial(Sediment, trim_fraction=trim)
    return factories


def run() -> list[SweepSample]:
    target = np.full(FEATURE_DIM, TARGET_DISTANCE)
    return run_poisoning_sweep(
        benign_source=isotropic_gaussian(feature_dim=FEATURE_DIM),
        malicious_source=jittered_target(target, jitter_scale=MALICIOUS_JITTER),
        target_attack_behavior=target,
        estimator_factories=_estimator_factories(),
        budget_grid=(0.0,),
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
                "false_positive_rate",
                "detection_rate",
                "threshold",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    sample.estimator_name,
                    sample.budget_fraction,
                    sample.seed,
                    sample.false_positive_rate,
                    sample.detection_rate,
                    sample.threshold,
                ]
            )


def main() -> None:
    samples = run()
    path = _results_path()
    write_csv(samples, path)
    print(f"exp05_fp_cost: wrote {len(samples)} rows to {path}")


if __name__ == "__main__":
    main()
