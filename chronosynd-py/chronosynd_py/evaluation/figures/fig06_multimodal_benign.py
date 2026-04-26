"""Figure 06, multimodal benign distribution. Two panels derived from
exp06's CSV showing target score and target-to-benign score ratio across
budgets. The bimodal scenario where Sediment's trimming pays off most"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from chronosynd_py.evaluation.plots import (
    DEFAULT_HEIGHT,
    SINGLE_COLUMN_WIDTH,
    aggregate_csv,
    color_for,
    save_figure,
)


def _csv_path() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "evaluation" / "results" / "exp06_multimodal_benign.csv"


def _output_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "paper" / "figures"


def _line_style_for(name: str) -> str:
    return "-" if name == "naive" else "--"


def _score_ratio(row: dict[str, str]) -> float:
    target = float(row["target_score"])
    benign = float(row["median_benign_score"])
    return target / benign if benign > 0.0 else float("inf")


def main() -> None:
    output_dir = _output_dir()

    target_series = aggregate_csv(_csv_path(), metric="target_score")
    target_fig, axes = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, DEFAULT_HEIGHT))
    for series in target_series:
        axes.errorbar(
            series.budgets,
            series.mean,
            yerr=series.stderr,
            label=series.estimator_name,
            color=color_for(series.estimator_name),
            linestyle=_line_style_for(series.estimator_name),
            marker="o",
            markersize=3,
            linewidth=1.2,
            capsize=2,
            alpha=0.9,
        )
    axes.set_xlabel(r"Poisoning budget $\beta$")
    axes.set_ylabel("Target score (log)")
    axes.set_yscale("log")
    axes.set_title("Multimodal benign, target score")
    axes.legend(frameon=False, fontsize=7, ncol=2)
    axes.grid(visible=True, alpha=0.3)
    target_fig.tight_layout()
    save_figure(target_fig, output_dir / "fig06_target_score.pdf")
    plt.close(target_fig)

    ratio_series = aggregate_csv(_csv_path(), metric=_score_ratio)
    ratio_fig, axes = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, DEFAULT_HEIGHT))
    for series in ratio_series:
        axes.errorbar(
            series.budgets,
            series.mean,
            yerr=series.stderr,
            label=series.estimator_name,
            color=color_for(series.estimator_name),
            linestyle=_line_style_for(series.estimator_name),
            marker="o",
            markersize=3,
            linewidth=1.2,
            capsize=2,
            alpha=0.9,
        )
    axes.set_xlabel(r"Poisoning budget $\beta$")
    axes.set_ylabel("target / median(benign)")
    axes.set_title("Multimodal benign, score ratio")
    axes.legend(frameon=False, fontsize=7, ncol=2)
    axes.grid(visible=True, alpha=0.3)
    ratio_fig.tight_layout()
    save_figure(ratio_fig, output_dir / "fig06_score_ratio.pdf")
    plt.close(ratio_fig)

    print(f"fig06_multimodal_benign: wrote 2 PDFs to {output_dir}")


if __name__ == "__main__":
    main()
