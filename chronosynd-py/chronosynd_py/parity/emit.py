"""Emit parity test vectors for the Rust port to validate against. Run as
a module to regenerate the JSON file the Rust integration test reads"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from chronosynd_py.baseline.naive import NaiveBaseline
from chronosynd_py.baseline.sediment import Sediment

SCHEMA_VERSION = 1
SAMPLE_GRID: tuple[tuple[int, int, int], ...] = (
    # (n_samples, n_features, seed)
    (50, 4, 1),
    (200, 8, 2),
    (500, 12, 3),
)
TRIM_FRACTIONS: tuple[float, ...] = (0.0, 0.1, 0.3, 0.5)
SCORE_INPUT_COUNT = 20


def _output_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root
        / "chronosynd-rs"
        / "crates"
        / "baseline"
        / "tests"
        / "parity_vectors.json"
    )


def _emit_naive(
    name: str,
    observations: np.ndarray,
    score_inputs: np.ndarray,
    *,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    baseline = NaiveBaseline(epsilon=epsilon)
    baseline.fit(observations)
    expected = [float(baseline.score(row)) for row in score_inputs]
    return {
        "name": name,
        "estimator": {"kind": "naive", "epsilon": epsilon},
        "fit_observations": observations.tolist(),
        "score_inputs": score_inputs.tolist(),
        "expected_scores": expected,
    }


def _emit_sediment(
    name: str,
    observations: np.ndarray,
    score_inputs: np.ndarray,
    *,
    trim_fraction: float,
    epsilon: float = 1e-6,
) -> dict[str, Any]:
    baseline = Sediment(trim_fraction=trim_fraction, epsilon=epsilon)
    baseline.fit(observations)
    expected = [float(baseline.score(row)) for row in score_inputs]
    return {
        "name": name,
        "estimator": {
            "kind": "sediment",
            "trim_fraction": trim_fraction,
            "epsilon": epsilon,
        },
        "fit_observations": observations.tolist(),
        "score_inputs": score_inputs.tolist(),
        "expected_scores": expected,
    }


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for n_samples, n_features, seed in SAMPLE_GRID:
        rng = np.random.default_rng(seed=seed)
        observations = rng.normal(size=(n_samples, n_features))
        score_inputs = rng.normal(size=(SCORE_INPUT_COUNT, n_features))

        cases.append(
            _emit_naive(
                f"naive_n{n_samples}_d{n_features}",
                observations,
                score_inputs,
            )
        )
        for trim in TRIM_FRACTIONS:
            trim_label = f"{round(trim * 100):02d}"
            cases.append(
                _emit_sediment(
                    f"sediment_trim{trim_label}_n{n_samples}_d{n_features}",
                    observations,
                    score_inputs,
                    trim_fraction=trim,
                )
            )
    return cases


def main() -> None:
    cases = build_cases()
    output = {"schema_version": SCHEMA_VERSION, "cases": cases}
    path = _output_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"parity emit: wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
