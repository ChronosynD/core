"""Unit tests for plot aggregation helpers. Verifies the CSV-to-per-
estimator series aggregation with means and standard errors, plus the
color-lookup helpers used by the per-figure scripts"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from chronosynd_py.evaluation.plots import (
    aggregate_csv,
    color_for,
)


@pytest.fixture(name="sweep_csv")
def sweep_csv_fixture(tmp_path: Path) -> Path:
    """Write a tiny sweep CSV with two estimators and three budgets"""
    rows = [
        {"estimator_name": "naive", "budget_fraction": 0.0, "seed": 0,
         "target_score": 100.0, "median_benign_score": 8.0, "detection_rate": 1.0},
        {"estimator_name": "naive", "budget_fraction": 0.0, "seed": 1,
         "target_score": 102.0, "median_benign_score": 8.5, "detection_rate": 1.0},
        {"estimator_name": "naive", "budget_fraction": 0.1, "seed": 0,
         "target_score": 50.0, "median_benign_score": 8.0, "detection_rate": 0.6},
        {"estimator_name": "naive", "budget_fraction": 0.1, "seed": 1,
         "target_score": 52.0, "median_benign_score": 8.2, "detection_rate": 0.7},
        {"estimator_name": "sediment_trim30", "budget_fraction": 0.0, "seed": 0,
         "target_score": 99.0, "median_benign_score": 8.1, "detection_rate": 1.0},
        {"estimator_name": "sediment_trim30", "budget_fraction": 0.0, "seed": 1,
         "target_score": 101.0, "median_benign_score": 8.0, "detection_rate": 1.0},
        {"estimator_name": "sediment_trim30", "budget_fraction": 0.1, "seed": 0,
         "target_score": 90.0, "median_benign_score": 7.5, "detection_rate": 0.95},
        {"estimator_name": "sediment_trim30", "budget_fraction": 0.1, "seed": 1,
         "target_score": 92.0, "median_benign_score": 7.6, "detection_rate": 0.96},
    ]
    csv_path = tmp_path / "sweep.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "estimator_name",
                "budget_fraction",
                "seed",
                "target_score",
                "median_benign_score",
                "detection_rate",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return csv_path


def test_aggregate_returns_one_series_per_estimator(sweep_csv: Path) -> None:
    series = aggregate_csv(sweep_csv, metric="target_score")
    assert {s.estimator_name for s in series} == {"naive", "sediment_trim30"}


def test_aggregate_groups_by_budget_within_estimator(sweep_csv: Path) -> None:
    series = aggregate_csv(sweep_csv, metric="target_score")
    naive = next(s for s in series if s.estimator_name == "naive")
    assert naive.budgets == [0.0, 0.1]


def test_aggregate_computes_mean_correctly(sweep_csv: Path) -> None:
    series = aggregate_csv(sweep_csv, metric="target_score")
    naive = next(s for s in series if s.estimator_name == "naive")
    assert naive.mean == pytest.approx([101.0, 51.0])


def test_aggregate_computes_stderr_for_multiple_seeds(sweep_csv: Path) -> None:
    series = aggregate_csv(sweep_csv, metric="target_score")
    naive = next(s for s in series if s.estimator_name == "naive")
    # std of [100, 102] with ddof=1 is sqrt(2), stderr = sqrt(2)/sqrt(2) = 1
    assert naive.stderr == pytest.approx([1.0, 1.0], rel=1e-9)


def test_aggregate_accepts_callable_metric(sweep_csv: Path) -> None:
    def ratio(row: dict[str, str]) -> float:
        return float(row["target_score"]) / float(row["median_benign_score"])

    series = aggregate_csv(sweep_csv, metric=ratio)
    naive = next(s for s in series if s.estimator_name == "naive")
    # Mean ratio at budget 0.0, ((100/8) + (102/8.5)) / 2 = (12.5 + 12.0) / 2 = 12.25
    assert naive.mean[0] == pytest.approx(12.25, rel=1e-6)


def test_aggregate_orders_budgets_within_each_series(sweep_csv: Path) -> None:
    series = aggregate_csv(sweep_csv, metric="target_score")
    for s in series:
        assert s.budgets == sorted(s.budgets)


def test_aggregate_orders_series_alphabetically(sweep_csv: Path) -> None:
    series = aggregate_csv(sweep_csv, metric="target_score")
    names = [s.estimator_name for s in series]
    assert names == sorted(names)


def test_color_for_known_estimator() -> None:
    assert color_for("naive") == "#d62728"
    assert color_for("sediment") == "#1f77b4"


def test_color_for_sediment_variant_uses_sediment_color() -> None:
    assert color_for("sediment_trim30") == color_for("sediment")
    assert color_for("sediment_trim50") == color_for("sediment")


def test_color_for_unknown_estimator_falls_back_to_gray() -> None:
    assert color_for("mystery_baseline") == "#7f7f7f"
