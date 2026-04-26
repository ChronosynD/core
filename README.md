# ChronosynD

**Behavioral host intrusion detection that survives baseline poisoning.**

Most behavioral HIDS in the literature assume the learning window is free of adversary activity. Real attacker dwell time is not zero (Mandiant M-Trends 2025 puts the 2024 global median at 11 days), so a detector that turns on while the adversary is already on the host fits a baseline shaped by the attack. ChronosynD's contribution, **Sediment**, is a poisoning-resistant baseline estimator that stays robust where the standard naive Gaussian baseline collapses.

Validated on captured Linux bash behavior under three poisoning attacks: **126x median score separation, zero false positives during attack capture.**

**Read the paper:** [`paper.pdf`](paper.pdf)

## Why

Skilled attackers rarely drop their own malware. They take over legitimate processes, things like `bash`, `python`, a long-running web server, or a cron job, and use those to do harmful work. Signature-based and technique-based detectors have nothing to match against, because the binary is the legitimate one and each individual action is within its capability set.

The bet is that even when the binary and the operations look normal, the *combination* a process produces under attacker control is statistically distinct from what the same process does in steady state on the same host. ChronosynD captures the steady-state distribution and alerts on drift away from it.

How it sits relative to other layers of defense:

- **Antivirus** matches binaries against known-bad signatures. ChronosynD has no signatures.
- **EDR** matches sequences of operations against known-bad techniques. ChronosynD has no technique library.
- **Network anomaly detection** watches the wire. ChronosynD watches the process below it.

## Research contribution: Sediment

Behavioral HIDS in the literature share an implicit assumption: the learning window during which the baseline is fitted is free of adversary activity. Real-world attacker dwell time is nonzero by definition. Mandiant's M-Trends 2025 report puts the global median at 11 days for 2024 incidents, with longer tails for sophisticated actors. A detector that turns on while the adversary is already on the host fits a baseline shaped by the adversary, and after that the planted behavior *is* the baseline's notion of normal.

We call this **baseline poisoning**. ChronosynD's research contribution is **Sediment**, a poisoning-resistant baseline estimator that treats the learning window as partially adversarial. Sediment uses a symmetric trimmed mean and a bias-corrected trimmed standard deviation per feature, parameterized by a single `trim_fraction` knob. Under the threat model in [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md), setting `trim_fraction` to roughly twice the worst expected adversary budget gives Sediment a defensible robustness guarantee.

## Architecture

Two language trees:

- [`chronosynd-rs/`](chronosynd-rs/), seven Rust crates that make up the production runtime: kernel-side BPF programs, a userspace collector, a feature extractor, the Sediment estimator, a real-time scoring engine, a tamper-evident SQLite-backed baseline store, and an operator CLI.
- [`chronosynd-py/`](chronosynd-py/), the Python research reference: the canonical Sediment implementation, the naive baseline used as the prior-work reference, the poisoning-attack harness, and the experiment scripts that produce paper figures.

Both the Sediment algorithm and the syscall n-gram extractor exist in both languages. CI enforces bit-equivalent outputs through two parity checks wired into [`scripts/check_parity.sh`](scripts/check_parity.sh): 300 cross-implementation comparisons for the baseline estimator within 1e-9 floating-point tolerance, and per-window feature-vector equality for the n-gram extractor across four event-stream cases. The Python side is authoritative during research iteration, the Rust side is authoritative for deployment, and neither is allowed to drift from the other.

See [`docs/architecture.md`](docs/architecture.md) for the per-crate breakdown.

## Status

Both the research path and the system path run end to end on Linux. The daemon attaches BPF, drains real events, scores against fitted baselines, and can record JSONL traces that the CLI replays back into a baseline.

| Piece | State |
|---|---|
| Sediment algorithm (Python reference) | Implemented and tested |
| Sediment algorithm (Rust port) | Implemented, parity-checked against Python |
| Naive baseline (both languages) | Implemented for the comparison |
| Alternative baselines (Consensus, AnomalyWithin) | Implemented in Python for the four-way ablation |
| Poisoning attacks (`pre_seed` uniform, `burst` contiguous, `targeted` white-box) | Implemented with reproducible seeds |
| Evaluation harness, nine experiments, eighteen paper PDFs | Reproducible from a command |
| Tamper-evident SQLite baseline store | Implemented with hash-chain verification |
| Real-time scoring engine | Implemented with cache warming from storage |
| Operator CLI (`chronosynd`) | Implemented, includes `fit-from-trace` for real captures |
| Collector daemon (`chronosynd-collector`) | Implemented end to end, supports `--record` for JSONL captures |
| BPF source code | Compiled into the collector via `--features bpf` on Linux |
| Real-data capture and replay pipeline | Capture script, attack payloads, and `fit-from-trace` CLI path |
| Real-data validation on captured Linux behavior | Validated on bash under WSL2: 126× median score separation, zero false positives during attack capture |

The numbers behind those rows: Python at 202 tests passing under `mypy --strict` and `ruff` with security and exception-handling rules enabled, Rust at 95 tests passing across seven crates with `forbid(unsafe_code)` on every crate that does not need it, parity covering both the baseline estimator (15 cases × 20 score inputs = 300 comparisons within 1e-9) and the syscall n-gram extractor (4 event-stream cases asserting per-window feature-vector equality).

## Platform

The runtime targets **Linux only**. The kernel-side collector uses eBPF and there is no equivalent on Windows or macOS with comparable syscall-level visibility. The Python research code, the Rust baseline crate, the storage layer, the scoring engine, and the CLI all run cross-platform; the BPF source is the only Linux-only component.

## Development

```bash
# Python research side: install, lint, type-check, run tests
cd chronosynd-py
uv sync --extra dev --extra viz
uv run ruff check chronosynd_py tests
uv run mypy
uv run pytest

# Rust runtime: workspace test
cd chronosynd-rs
cargo test --workspace

# Reproduce every paper figure end to end
bash scripts/reproduce_all.sh

# Verify Python and Rust agree on the shared parity vectors
bash scripts/check_parity.sh
```

## Real-data workflow

Synthetic Gaussians drive the paper's experiments, but the runtime also fits baselines from real captured behavior on Linux. The flow is record-then-replay: the collector serializes every observed event to a JSONL file, then the CLI replays that file through the same feature extractor used at runtime to fit a baseline.

```bash
# Build a release-mode collector with the BPF feature enabled
cd chronosynd-rs && cargo build --release --features bpf

# Capture 10 minutes of clean nginx behavior to a JSONL event trace
sudo scripts/capture_baseline.sh 600 /tmp/nginx_clean.jsonl

# Fit a Sediment baseline from the trace and persist it
chronosynd-rs/target/release/chronosynd \
    --store /tmp/chronosynd.db \
    fit-from-trace nginx --input /tmp/nginx_clean.jsonl --estimator sediment

# Capture clean, fit, run an attack payload, score the drift
sudo scripts/real_world_demo.sh nginx 600 evaluation/attack_payloads/etc_recon.sh
```

See [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) for the full real-data evaluation protocol and [`evaluation/attack_payloads/README.md`](evaluation/attack_payloads/README.md) for the bundled behavioral fixtures.

## Citation

[`CITATION.cff`](CITATION.cff)

## License

ChronosynD is dual-licensed. Source code and written materials are covered by different licenses.

| What | License | File |
|---|---|---|
| Source code (everything compiled, executed, or imported as a dependency) | Apache License 2.0 | [`LICENSE`](LICENSE) |
| Written materials including the paper, formal documentation, threat model, evaluation protocol, design notes, figures and PDFs, READMEs, and **algorithm descriptions** (the mathematical formulation of Sediment, the bias-correction derivation, the threat-model formalization) | Creative Commons Attribution 4.0 International (CC-BY-4.0) | [`LICENSE-CC-BY-4.0`](LICENSE-CC-BY-4.0) |

If you redistribute this work, keep [`NOTICE`](NOTICE) and [`LICENSE`](LICENSE), and `LICENSE-CC-BY-4.0` too if you redistribute any prose, figures, or algorithm descriptions. For attribution conventions and the formal academic citation, see [`CITATION.cff`](CITATION.cff).
