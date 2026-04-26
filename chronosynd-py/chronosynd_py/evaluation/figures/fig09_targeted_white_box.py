"""Figure 09, four-way ablation under white-box targeted poisoning. The
strongest threat-model evidence the paper produces. The white-box
adversary knows Sediment's trim_fraction and picks the optimal placement"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

from chronosynd_py.evaluation.plots import (
    DEFAULT_HEIGHT,
    SINGLE_COLUMN_WIDTH,
    AggregatedSeries,
    aggregate_csv,
    color_for,
    save_figure,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _csv_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "evaluation" / "results" / "exp09_targeted_white_box.csv"


def _output_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "paper" / "figures"


def _score_ratio(row: dict[str, str]) -> float:
    target = float(row["target_score"])
    benign = float(row["median_benign_score"])
    return target / benign if benign > 0.0 else float("inf")


def _plot(
    series_list: list[AggregatedSeries],
    *,
    title: str,
    ylabel: str,
    use_log_y: bool = False,
) -> Figure:
    figure, axes = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, DEFAULT_HEIGHT))
    for series in series_list:
        axes.errorbar(
            series.budgets,
            series.mean,
            yerr=series.stderr,
            label=series.estimator_name,
            color=color_for(series.estimator_name),
            marker="o",
            markersize=4,
            linewidth=1.4,
            capsize=2,
        )
    axes.set_xlabel(r"Poisoning budget $\beta$")
    axes.set_ylabel(ylabel)
    axes.set_title(title)
    if use_log_y:
        axes.set_yscale("log")
    axes.legend(frameon=False, fontsize=8)
    axes.grid(visible=True, alpha=0.3)
    figure.tight_layout()
    return figure


def main() -> None:
    output_dir = _output_dir()
    csv_path = _csv_path()

    target_series = aggregate_csv(csv_path, metric="target_score")
    target_fig = _plot(
        target_series,
        title="Target score under white-box targeted poisoning",
        ylabel="Target score",
        use_log_y=True,
    )
    save_figure(target_fig, output_dir / "fig09_target_score.pdf")
    plt.close(target_fig)

    ratio_series = aggregate_csv(csv_path, metric=_score_ratio)
    ratio_fig = _plot(
        ratio_series,
        title="Score ratio under white-box targeted poisoning",
        ylabel="target / median(benign)",
    )
    save_figure(ratio_fig, output_dir / "fig09_score_ratio.pdf")
    plt.close(ratio_fig)

    detect_series = aggregate_csv(csv_path, metric="detection_rate")
    detect_fig = _plot(
        detect_series,
        title="Detection rate at 5 percent FPR, white-box targeted attack",
        ylabel="Detection rate",
    )
    save_figure(detect_fig, output_dir / "fig09_detection_rate.pdf")
    plt.close(detect_fig)

    print(f"fig09_targeted_white_box: wrote 3 PDFs to {output_dir}")


if __name__ == "__main__":
    main()
