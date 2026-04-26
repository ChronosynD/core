<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright 2026 The ChronosynD Authors, see LICENSE-CC-BY-4.0 -->

# Paper outline

Working title: **Sediment: Poisoning-Resistant Behavioral Baselines for Host Intrusion Detection**

The outline is tracked in git so that experiments stay tied to claims. If an experiment does not support a claim in this outline, we either change the claim or drop the experiment.

## 1. Introduction

- Behavioral HIDS detect process drift from a learned baseline.
- Implicit assumption across prior work: the learning window is adversary-free.
- Real-world attacker dwell time is nonzero by definition. M-Trends 2025 puts the global median at 11 days for 2024 incidents. Any detector deployed onto an already-compromised host fits a baseline shaped by the adversary.
- Contribution: formalize the baseline-poisoning threat model, show naive baselines collapse under it, propose **Sediment**, evaluate.

## 2. Background and related work

- Behavioral HIDS lineage: Forrest "sense of self", sequence models, provenance graphs.
- Mimicry attacks: Wagner & Soto 2002, Goyal et al. NDSS 2023.
- Robust statistics primer: trimmed means, Huber estimators, median-of-means.
- Concept drift versus adversarial drift, distinct problems often conflated in deployed tools.

## 3. Threat model

- Adversary present on the host before or during the learning window with budget `β ∈ [0, 1]`, defined as `|A| / N`.
- Goal: cause a specific post-learning malicious behavior `M` to score below threshold `θ`.
- Observability variants: black-box, grey-box, white-box. Default for the headline experiments is white-box.
- See [`../docs/THREAT_MODEL.md`](../docs/THREAT_MODEL.md) for the formal statement.

## 4. ChronosynD system

- Pipeline overview: BPF source, feature extractor, scorer, tamper-evident store.
- Cross-language reference and production split. Python is authoritative for research, Rust is authoritative for deployment, CI enforces bit-equivalence on both the estimator and the n-gram extractor.
- Operator surface, the `chronosynd` CLI: baseline inspection, maintenance windows, store verification, fitting from CSV or JSONL traces.

## 5. Sediment

- Algorithm: per-feature symmetric trimmed mean and trimmed standard deviation, with a closed-form Gaussian-truncation bias correction so the trimmed std unbiasedly estimates the underlying sigma on clean data.
- Single design knob `trim_fraction`. Setting it to roughly twice the worst expected adversary budget gives the resistance guarantee.
- Complexity: `O(n log n)` per feature for sorting, `O(d)` memory for the fitted moments per process.
- Discussion of the bias correction's calibration to the unimodal Gaussian null and what that implies for non-Gaussian feature distributions.

## 6. Evaluation

- Workloads: `isotropic_gaussian`, `heterogeneous_gaussian`, `gaussian_mixture`, configured per experiment.
- Attacks: `inject_pre_seed` (uniform shuffled), `inject_burst` (contiguous block), `inject_targeted` (white-box grid search).
- Estimators compared: naive mean-and-stddev, Sediment at trim fractions 0.1 / 0.2 / 0.3 / 0.5, Consensus, Anomaly-Within.
- Nine controlled experiments plus one real-data validation:
  - **Exp01**, clean-baseline FPR, sanity check that thresholds land at the configured target FPR for every estimator.
  - **Exp02**, naive under poisoning, the motivation figure showing the score collapse on isotropic benign.
  - **Exp03**, Sediment design space, target score across the budget × trim_fraction grid, the main empirical result.
  - **Exp04**, budget sweep with naive vs Sediment overlaid, three views of the same data.
  - **Exp05**, FP cost as a function of trim_fraction under unpoisoned conditions.
  - **Exp06**, multimodal benign distribution, target outside the modes, exposes a structural limit of independent-Gaussian baselines.
  - **Exp07**, four-way ablation under uniform poisoning.
  - **Exp08**, four-way ablation under burst poisoning.
  - **Exp09**, four-way ablation under white-box targeted poisoning.
  - **Real-data**, fitted Sediment baseline on captured Linux bash behavior, end-to-end attack-payload run, 126× median score separation between alert and clean windows, zero false positives during the attack capture period.
- Reporting: ten seeds per cell with standard-error bars, target_fpr = 0.05 throughout, all controlled experiments regenerable via `bash scripts/reproduce_all.sh`. Real-data run reproducible via `bash scripts/real_world_demo.sh` on a Linux box with the BPF-enabled collector built.

## 7. Discussion

- When Sediment helps cleanly: budgets up to roughly `trim_fraction / 2` on broadly-Gaussian features.
- When the bias correction over-trims: bimodal benign distributions where the trimmed middle has lower variance than the full mixture, leading to inflated absolute scores even though discriminability ratios stay reasonable.
- Interaction with concept drift and the maintenance-window mechanism, which is operator-driven and orthogonal to the poisoning threat.
- Residual risk from an adaptive adversary that models Sediment's correction explicitly and crafts an injection that survives the trim.

## 8. Limitations

- Assumes per-feature independence under a Gaussian null. Multimodal feature distributions need a different baseline family.
- Controlled poisoning evaluation runs on synthetic Gaussian workloads. The single-host real-data run validates the artifact end-to-end but is not a substitute for multi-host controlled evaluation across diverse workloads.
- The current threat-model formalization assumes the adversary cannot manipulate kernel-side identity (PID, UID, cgroup). True for unprivileged adversaries, starts to fray under kernel-level compromise.
- The default BPF probe captures every syscall and classifies the kind in-kernel, which works on stock Linux 6.x. Older kernels without `raw_syscalls/sys_enter` would need a different attach point.