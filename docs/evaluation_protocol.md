<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright 2026 The ChronosynD Authors, see LICENSE-CC-BY-4.0 -->

# Evaluation protocol

How a single experiment is structured. Holding this protocol constant across experiments is what makes the paper's results comparable.

## Per-experiment pipeline

1. **Workload selection.** Pick a benign source from `chronosynd_py.evaluation.workloads` and a malicious source. Both are deterministic given an `np.random.Generator`.
2. **Poisoning configuration.** Pick an attack from `chronosynd_py.evaluation.attacks` and a budget grid in `[0.0, 1.0]`.
3. **Estimator factories.** Pick the estimators to compare. Every factory is a zero-argument callable that returns a fresh `Baseline` instance, so the harness constructs a clean estimator per cell of the sweep.
4. **Sweep execution.** Hand the configuration to `chronosynd_py.evaluation.harness.run_poisoning_sweep`. For each `(estimator, budget, seed)` cell the harness draws independent RNG streams via `np.random.SeedSequence(seed).spawn(4)`, generates a benign learning window, applies the poisoning attack, fits a fresh estimator instance, draws benign and malicious test sets, scores both with the fitted baseline, and records one `SweepSample` row.
5. **Metrics.** Each row carries a detection rate at the configured target FPR, the false-positive rate actually observed, the chosen threshold, the score of the target observation, and the median benign and malicious test scores.
6. **Persistence.** The experiment script writes a flat CSV to `evaluation/results/<experiment>.csv`. Each row is identified by `(estimator_name, budget_fraction, seed)`, so the CSV is the canonical artifact for the paper figure.
7. **Plotting.** A matching figure script under `chronosynd_py.evaluation.figures.<figXX>` reads the CSV, aggregates by `(estimator, budget)` with mean and standard error across seeds, and writes a PDF into `paper/figures/`.

## Reporting conventions

- Detection rate and false-positive rate are reported at the operator-realistic target FPR (default 5 percent). Threshold selection is per-estimator, the `1 - target_fpr` quantile of benign test scores.
- Confidence comes from error bars across at least ten seeds per configuration. Most experiments use ten; a few use twenty for figures where a tight FPR estimate matters.
- Poisoning budgets are reported as a fraction of the benign learning-window size, not as an absolute count, so results stay portable across workloads.
- Score-level metrics (target score, score ratio) are reported alongside detection rates because detection rate under quantile thresholding is scale-invariant and obscures variance-inflation effects that the score-level view captures cleanly.

## What not to do

- Don't use post-learning test traces during baseline construction.
- Don't tune thresholds per-experiment on the test trace. The threshold-selection rule must be declared before the test trace is touched and held constant across the comparison.
- Don't compare estimators using different feature extractors. Feature extraction is held constant so only the baseline estimator varies.
- Don't compare estimators using different RNG streams. The harness uses `SeedSequence.spawn(4)` to give each `(estimator, budget, seed)` cell four independent streams, and the same `seed` produces the same data across estimators.

## Reproducibility

[`bash scripts/reproduce_all.sh`](../scripts/reproduce_all.sh) regenerates every CSV and every PDF from scratch. The full pipeline takes well under two minutes.

## Real-data evaluation

The synthetic Gaussian workloads carry the controlled experiments. The real-data path complements them with end-to-end demonstrations on captured behavior. The protocol is symmetric to the synthetic one with the workload sources swapped for recorded JSONL traces.

1. **Capture.** Run [`scripts/capture_baseline.sh DURATION OUTPUT_PATH`](../scripts/capture_baseline.sh), which invokes the collector under `--bpf --record`. The collector serializes every observed event to a JSONL file as it scores. The capture is bounded by `timeout` so a runaway run cannot block the protocol.
2. **Fit.** Run `chronosynd fit-from-trace <process_key> --input <recording>`. It filters the recording to events whose `comm` matches the key, replays them through the canonical `SyscallNgramExtractor`, and persists a Sediment baseline to the configured store.
3. **Attack run.** Capture a second trace while an attack payload from [`evaluation/attack_payloads/`](../evaluation/attack_payloads/) runs alongside the target process. The attack payloads are behavioral fixtures, not exploitation, so they are safe to run on a development host.
4. **Score.** Replay the attack trace through `fit-from-trace` to materialize observations, or run the live daemon against the same workload to score in real time. Drift scores above the per-process threshold print as `[ALERT]` lines on stdout.

[`scripts/real_world_demo.sh`](../scripts/real_world_demo.sh) runs steps one through four in sequence as a smoke test of the full pipeline on a Linux box with a release-mode collector built using `cargo build --release --features bpf`.

The real-data and synthetic paths share the same feature extractor, scorer, and storage layer, so results from one are directly comparable to results from the other. The split is purely about how the observation rows are produced.
