"""Figure 01, clean-baseline false-positive rate per estimator. Bar chart
under unpoisoned windows with standard-error bars across seeds. The
"this is what the baseline costs you on quiet days" panel"""

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
    return repo_root / "evaluation" / "results" / "exp01_clean_baseline_fpr.csv"


def _output_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "paper" / "figures"


def main() -> None:
    series = aggregate_csv(_csv_path(), metric="false_positive_rate")

    figure, axes = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, DEFAULT_HEIGHT))
    names = [s.estimator_name for s in series]
    means = [s.mean[0] for s in series]
    stderrs = [s.stderr[0] for s in series]
    colors = [color_for(name) for name in names]

    positions = list(range(len(names)))
    axes.bar(
        positions,
        means,
        yerr=stderrs,
        color=colors,
        capsize=3,
        edgecolor="black",
        linewidth=0.6,
    )
    axes.set_xticks(positions)
    axes.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    axes.set_ylabel("False-positive rate")
    axes.set_title("Clean-condition FPR per estimator")
    axes.axhline(0.05, color="gray", linestyle="--", linewidth=0.8, label="target FPR")
    axes.legend(frameon=False, fontsize=8)
    axes.grid(visible=True, alpha=0.3, axis="y")
    figure.tight_layout()

    output_dir = _output_dir()
    save_figure(figure, output_dir / "fig01_clean_baseline_fpr.pdf")
    plt.close(figure)
    print(f"fig01_clean_baseline_fpr: wrote PDF to {output_dir}")


if __name__ == "__main__":
    main()
