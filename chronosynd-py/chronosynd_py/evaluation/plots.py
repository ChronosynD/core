"""Plot helpers for paper figures. Owns the canonical color per estimator,
column-width PDF sizing, and the seed-aggregation that turns a sweep CSV
into series with mean and standard error"""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# Column widths in inches, sized for typical IEEE/ACM two-column papers
SINGLE_COLUMN_WIDTH = 3.5
DOUBLE_COLUMN_WIDTH = 7.0
DEFAULT_HEIGHT = 2.5

# Canonical color palette per estimator family
_ESTIMATOR_COLORS: dict[str, str] = {
    "naive": "#d62728",
    "sediment": "#1f77b4",
    "consensus": "#2ca02c",
    "anomaly_within": "#9467bd",
}
_FALLBACK_COLOR = "#7f7f7f"


@dataclass(frozen=True, slots=True)
class AggregatedSeries:
    """Per-budget mean and standard error of a metric for one estimator"""

    estimator_name: str
    budgets: list[float]
    mean: list[float]
    stderr: list[float]


MetricExtractor = Callable[[dict[str, str]], float]


def aggregate_csv(
    csv_path: Path,
    *,
    metric: str | MetricExtractor,
) -> list[AggregatedSeries]:
    """Group sweep rows by `(estimator_name, budget_fraction)` and
    aggregate. `metric` is either a CSV column name or a callable that
    derives a per-row value, used for ratios without CSV preprocessing"""
    extract = _build_extractor(metric)
    rows = _read_rows(csv_path)
    grouped = _group_by_estimator_and_budget(rows, extract)
    return _summarize(grouped)


def color_for(estimator_name: str) -> str:
    """Look up the canonical color for `estimator_name`. Variants share
    the Sediment color so they are visually grouped on combined plots"""
    if estimator_name in _ESTIMATOR_COLORS:
        return _ESTIMATOR_COLORS[estimator_name]
    for prefix, color in _ESTIMATOR_COLORS.items():
        if estimator_name.startswith(prefix):
            return color
    return _FALLBACK_COLOR


_PDF_METADATA: dict[str, str] = {
    "Title": "ChronosynD paper figure",
    "Author": "The ChronosynD Authors",
    "Subject": "Sediment, poisoning-resistant behavioral baselines for HIDS",
    "Keywords": "intrusion-detection; baseline-poisoning; trimmed-mean; CC-BY-4.0",
    "Creator": "chronosynd_py.evaluation.figures",
    "Producer": "matplotlib via chronosynd-py, https://github.com/ChronosynD/core",
}


def save_figure(figure: Figure, output_path: Path) -> None:
    """Write a figure as PDF with embedded CC-BY-4.0 metadata. The
    output directory is created if needed. The metadata makes the
    license traceable through the binary"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        metadata=_PDF_METADATA,
    )


def _build_extractor(metric: str | MetricExtractor) -> MetricExtractor:
    if callable(metric):
        return metric
    column = metric

    def _from_column(row: dict[str, str]) -> float:
        return float(row[column])

    return _from_column


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _group_by_estimator_and_budget(
    rows: list[dict[str, str]],
    extract: MetricExtractor,
) -> dict[tuple[str, float], list[float]]:
    grouped: dict[tuple[str, float], list[float]] = defaultdict(list)
    for row in rows:
        key = (row["estimator_name"], float(row["budget_fraction"]))
        grouped[key].append(extract(row))
    return grouped


def _summarize(
    grouped: dict[tuple[str, float], list[float]],
) -> list[AggregatedSeries]:
    by_estimator: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
    for (estimator_name, budget), values in grouped.items():
        # Drop non-finite samples. score_ratio yields inf when an estimator
        # drops every feature and median(benign) collapses to zero
        finite = [v for v in values if np.isfinite(v)]
        if not finite:
            mean = float("nan")
            stderr = 0.0
        else:
            mean = float(np.mean(finite))
            stderr = (
                float(np.std(finite, ddof=1) / np.sqrt(len(finite)))
                if len(finite) > 1
                else 0.0
            )
        by_estimator[estimator_name].append((budget, mean, stderr))

    series: list[AggregatedSeries] = []
    for estimator_name in sorted(by_estimator):
        sorted_points = sorted(by_estimator[estimator_name], key=lambda point: point[0])
        series.append(
            AggregatedSeries(
                estimator_name=estimator_name,
                budgets=[point[0] for point in sorted_points],
                mean=[point[1] for point in sorted_points],
                stderr=[point[2] for point in sorted_points],
            )
        )
    return series
