"""Experiment 06, multimodal benign distribution. Swaps the isotropic
benign source for a two-component Gaussian mixture and places the target
outside both modes. The bimodal-with-target-between case is out of scope"""

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
from chronosynd_py.evaluation.workloads import gaussian_mixture, jittered_target

FEATURE_DIM = 8
MODE_OFFSET = 3.0
MODE_SCALE = 0.3
TARGET_DISTANCE = 6.0  # target sits past the upper mode at +3, in zero-density tail
TRIM_FRACTIONS: tuple[float, ...] = (0.10, 0.30, 0.50)
BUDGET_GRID: tuple[float, ...] = (0.0, 0.05, 0.10, 0.15, 0.20)
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
    return results_dir / "exp06_multimodal_benign.csv"


def _benign_components() -> list[tuple[float, list[float], list[float]]]:
    """Symmetric two-mode mixture. Benign sits at +/- MODE_OFFSET per feature"""
    positive_mean = [MODE_OFFSET] * FEATURE_DIM
    negative_mean = [-MODE_OFFSET] * FEATURE_DIM
    scale = [MODE_SCALE] * FEATURE_DIM
    return [(1.0, positive_mean, scale), (1.0, negative_mean, scale)]


def _estimator_factories() -> dict[str, EstimatorFactory]:
    factories: dict[str, EstimatorFactory] = {"naive": NaiveBaseline}
    for trim in TRIM_FRACTIONS:
        name = f"sediment_trim{round(trim * 100):02d}"
        factories[name] = partial(Sediment, trim_fraction=trim)
    return factories


def run() -> list[SweepSample]:
    target = np.full(FEATURE_DIM, TARGET_DISTANCE)
    return run_poisoning_sweep(
        benign_source=gaussian_mixture(_benign_components()),
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
    print(f"exp06_multimodal_benign: wrote {len(samples)} rows to {path}")


if __name__ == "__main__":
    main()
