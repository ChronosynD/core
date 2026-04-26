<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright 2026 The ChronosynD Authors, see LICENSE-CC-BY-4.0 -->

# Workloads

Benign workload fixtures that produce learning-window traces. Two paths land here: one synthetic, one captured.

## Synthetic workloads

The synthetic Gaussian factories live in code at [`chronosynd_py.evaluation.workloads`](../../chronosynd-py/chronosynd_py/evaluation/workloads.py) and drive every paper experiment. No fixture files are needed. Three factories are exposed:

- `isotropic_gaussian(feature_dim, loc, scale)` for the canonical experiments
- `heterogeneous_gaussian(mean, scale)` for per-feature heterogeneity
- `gaussian_mixture(components)` for multimodal benign distributions

## Captured real-data traces

Real captures land in this directory as JSONL files produced by the collector's `--record` mode. The protocol:

```bash
# Linux only, requires a release-mode collector built with --features bpf
sudo ../../scripts/capture_baseline.sh 600 ./nginx_clean.jsonl
```

Each file is a stream of newline-delimited `WireEvent` records. The wire format is in [`chronosynd-rs/crates/collector/src/wire.rs`](../../chronosynd-rs/crates/collector/src/wire.rs). Captures are not committed because they are host-specific and large. The protocol for regenerating them lives in [`docs/evaluation_protocol.md`](../../docs/evaluation_protocol.md).

To turn a capture into a fitted baseline:

```bash
chronosynd --store /tmp/chronosynd.db \
    fit-from-trace nginx --input ./nginx_clean.jsonl \
    --estimator sediment --trim-fraction 0.3
```

## Planned captured workloads

The capture script and `fit-from-trace` path are general. The workloads below are the ones the paper plans to demonstrate on:

- `nginx_worker`, a running nginx under a synthetic request load
- `cron_backup`, an rsync-based backup cron job
- `sshd_session`, interactive sshd session traces
- `python_web`, a Python web service under request load

Determinism on real captures is bounded. The same workload on the same host should produce statistically equivalent feature distributions, but byte-identical traces are not the goal. Determinism for the paper's controlled comparisons comes from the synthetic path.
