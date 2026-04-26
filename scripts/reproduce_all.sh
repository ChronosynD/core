#!/usr/bin/env bash
# Regenerate every paper figure from scratch by chaining experiments and figure scripts

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root/chronosynd-py"

experiments=(
  "exp01_clean_baseline_fpr"
  "exp02_naive_under_poisoning"
  "exp03_robust_under_poisoning"
  "exp04_budget_sweep"
  "exp05_fp_cost"
  "exp06_multimodal_benign"
  "exp07_alternatives_ablation"
  "exp08_burst_alternatives_ablation"
  "exp09_targeted_white_box"
)

figures=(
  "fig01_clean_baseline_fpr"
  "fig02_naive_under_poisoning"
  "fig03_robust_under_poisoning"
  "fig04_budget_sweep"
  "fig05_fp_cost"
  "fig06_multimodal_benign"
  "fig07_alternatives_ablation"
  "fig08_burst_alternatives_ablation"
  "fig09_targeted_white_box"
)

for name in "${experiments[@]}"; do
  echo "==> regenerating CSV for $name"
  uv run python -m "chronosynd_py.evaluation.experiments.$name"
done

for name in "${figures[@]}"; do
  echo "==> regenerating PDF for $name"
  uv run python -m "chronosynd_py.evaluation.figures.$name"
done

echo "==> Done, see $repo_root/paper/figures/"
