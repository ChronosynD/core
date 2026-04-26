<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright 2026 The ChronosynD Authors, see LICENSE-CC-BY-4.0 -->

# Baseline poisoning threat model

The formal definition of the adversary ChronosynD's research contribution defends against. Everything in `evaluation/attacks/` is a concrete instance of this model.

## Setting

ChronosynD observes a stream of behavioral events from a monitored process on a single host. Over a **learning window** of length *T* it constructs a baseline *B* by applying a baseline estimator *E* to the observed events. After *T* it enters the **scoring phase**: each new observation is scored against *B*, and observations with score above threshold *θ* generate alerts.

## Adversary

We assume a **pre-existing adversary**: the attacker is present on the host at or before time 0, the start of the learning window. This reflects the real-world fact that attacker dwell times frequently exceed realistic learning windows. 

The adversary has two phases:

- **Learning phase (0 ≤ *t* < *T*).** The adversary injects a budgeted set of adversary-chosen observations *A* into the event stream. Observations are *plausible*: they respect the same syntactic constraints as legitimate observations but are chosen strategically.
- **Scoring phase (*t* ≥ *T*).** The adversary executes a specific malicious behavior *M*, a pre-chosen sequence of observations representing the attack the adversary ultimately wants to perform.

### Adversary budget

The adversary's budget is parameterized by *β* ∈ [0, 1], the **poisoning fraction**:

- |*A*| ≤ *β* · *N*, where *N* is the total number of legitimate observations in the learning window.

Small *β* (≤ 0.05) represents opportunistic pre-existing presence. Larger *β* (up to ~0.25) represents sustained, patient compromise. *β* > 0.5 is out of scope: under majority-adversary conditions no baseline-learning detector can recover.

### Adversary goal

Cause *M* to be scored below *θ* by the baseline *B* = *E*(legitimate ∪ *A*):

  **score**(*M*; *E*(legitimate ∪ *A*)) < *θ*

without triggering alerts during the learning phase. Learning-phase observations are not scored, so this is free unless a *learning-phase vigilance* mechanism is in play.

### Adversary knowledge

Three variants:

| Variant | Adversary knows |
|---|---|
| Black-box | That a detector exists |
| Grey-box | The detector's feature extractor and estimator family |
| White-box | Exact estimator *E*, threshold *θ*, and learning-window length *T* |

**Default for headline experiments: white-box.** White-box is the conservative claim. Grey-box and black-box results are reported but are not the headline.

### What the adversary cannot do

- Modify legitimate observations already emitted by the kernel. The event stream up to time *t* is append-only from the adversary's perspective.
- Tamper with the baseline store. The storage layer's hash chain detects corruption at startup.
- Forge kernel-level identity (PID, UID, cgroup). The kernel guarantees this.
- Observe or modify the detector's internal state at runtime.

## Defender goal

Design an estimator *E* such that

  **score**(*M*; *E*(legitimate ∪ *A*)) ≥ *θ*    for all *A* with |*A*| ≤ *β* · *N*

under some target poisoning budget *β*, while keeping the **clean-condition false-positive rate** comparable to the naive baseline:

  **FPR**(*E*(legitimate); benign) ≈ **FPR**(*E*<sub>naive</sub>(legitimate); benign)

## Out of scope

- **Concept drift.** Legitimate behavioral evolution from software updates, maintenance windows, and seasonal patterns. Handled separately by the maintenance-window mechanism in the storage layer.
- **Network-level attacks.** ChronosynD operates at the process-behavior layer.
- **Majority adversary.** *β* > 0.5. No estimator can recover meaningful signal under this condition.
- **Baseline-store tamper as part of the attack.** Out of scope because the storage layer's hash chain provides tamper-evidence. Included here for completeness.

## Evaluation instantiation

Concrete adversary strategies in `evaluation/attacks/`:

- `pre_seed.py`. The adversary injects observations that shift the baseline toward *M*, shuffled uniformly across the learning window.
- `burst.py`. The adversary concentrates the injection budget in a contiguous slice of the learning window.
- `targeted.py`. The white-box optimal adversary: grid-searches the per-feature placement that minimizes **score**(*M*) after a fresh fit by an estimator with known *trim_fraction*.