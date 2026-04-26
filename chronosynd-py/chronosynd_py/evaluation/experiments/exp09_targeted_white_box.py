"""Experiment 09, white-box adversary against the four-way ablation. The
targeted attack knows the defender's trim_fraction and grid-searches the
placement that minimizes the resulting target score. The strongest
threat-model evidence the paper can produce against Sediment"""

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
from chronosynd_py.evaluation.attacks.targeted import inject_targeted
from chronosynd_py.evaluation.harness import (
    AttackInjector,
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
    return results_dir / "exp09_targeted_white_box.csv"


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


def _targeted_injector_for(defender_trim_fraction: float) -> AttackInjector:
    """The white-box adversary's knowledge of the defender's trim is what
    makes this attack white-box. The harness pre-binds it per estimator"""
    def _inject(
        benign: ObservationBatch,
        target: Observation,
        budget: float,
        rng: np.random.Generator,
    ) -> PoisonedTrace:
        return inject_targeted(
            benign,
            target,
            budget_fraction=budget,
            defender_trim_fraction=defender_trim_fraction,
            rng=rng,
        )
    return _inject


def run() -> list[SweepSample]:
    """Per-estimator sweep. The white-box attack adapts to each defender's
    trim_fraction so the rows are honest even though the harness contract
    is one injector per call to `run_poisoning_sweep`"""
    target = np.full(FEATURE_DIM, TARGET_DISTANCE)
    benign = isotropic_gaussian(feature_dim=FEATURE_DIM)
    malicious = jittered_target(target, jitter_scale=MALICIOUS_JITTER)

    samples: list[SweepSample] = []
    factories = _estimator_factories()
    # Naive, consensus, and anomaly_within do not trim, so the white-box
    # adversary against them reduces to the at-target attack. Only
    # Sediment gets a different trim_fraction
    estimator_trim: dict[str, float] = {
        "naive": 0.0,
        "consensus": 0.0,
        "anomaly_within": 0.0,
        "sediment": SEDIMENT_TRIM_FRACTION,
    }
    for estimator_name, factory in factories.items():
        injector = _targeted_injector_for(estimator_trim[estimator_name])
        samples.extend(
            run_poisoning_sweep(
                benign_source=benign,
                malicious_source=malicious,
                target_attack_behavior=target,
                estimator_factories={estimator_name: factory},
                budget_grid=BUDGET_GRID,
                seeds=SEEDS,
                attack_injector=injector,
                learning_window_size=LEARNING_WINDOW,
                benign_test_size=BENIGN_TEST,
                malicious_test_size=MALICIOUS_TEST,
                target_fpr=TARGET_FPR,
            )
        )
    return samples


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
    print(f"exp09_targeted_white_box: wrote {len(samples)} rows to {path}")


if __name__ == "__main__":
    main()
