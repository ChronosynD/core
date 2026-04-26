"""Figure 03, Sediment design space against poisoning. One line per
estimator, with naive as the prior-work reference and Sediment variants
showing how trim trades clean-condition cost against poisoning resistance"""

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
    return repo_root / "evaluation" / "results" / "exp03_robust_under_poisoning.csv"


def _output_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    return repo_root / "paper" / "figures"


def _line_style_for(name: str) -> str:
    return "-" if name == "naive" else "--"


def main() -> None:
    series = aggregate_csv(_csv_path(), metric="target_score")
    figure, axes = plt.subplots(figsize=(SINGLE_COLUMN_WIDTH, DEFAULT_HEIGHT))

    for s in series:
        axes.errorbar(
            s.budgets,
            s.mean,
            yerr=s.stderr,
            label=s.estimator_name,
            color=color_for(s.estimator_name),
            linestyle=_line_style_for(s.estimator_name),
            marker="o",
            markersize=3,
            linewidth=1.2,
            capsize=2,
            alpha=0.9,
        )

    axes.set_xlabel(r"Poisoning budget $\beta$")
    axes.set_ylabel("Target score (log)")
    axes.set_yscale("log")
    axes.set_title("Sediment vs naive across trim and budget")
    axes.legend(frameon=False, fontsize=7, ncol=2)
    axes.grid(visible=True, alpha=0.3)
    figure.tight_layout()

    output_dir = _output_dir()
    save_figure(figure, output_dir / "fig03_robust_under_poisoning.pdf")
    plt.close(figure)
    print(f"fig03_robust_under_poisoning: wrote PDF to {output_dir}")


if __name__ == "__main__":
    main()
