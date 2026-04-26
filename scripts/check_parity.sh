#!/usr/bin/env bash
# Verify Python and Rust implementations agree on shared test vectors,
# covers both the baseline estimator and the syscall n-gram extractor

set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"

echo "==> regenerating baseline parity vectors from the Python reference"
(cd "$repo_root/chronosynd-py" && uv run python -m chronosynd_py.parity.emit)

echo "==> regenerating feature parity vectors from the Python reference"
(cd "$repo_root/chronosynd-py" && uv run python -m chronosynd_py.parity.emit_features)

echo "==> running the Rust baseline parity test"
(cd "$repo_root/chronosynd-rs" && cargo test -p chronosynd-baseline --test parity_tests --quiet)

echo "==> running the Rust feature-extractor parity test"
(cd "$repo_root/chronosynd-rs" && cargo test -p chronosynd-features --test parity_tests --quiet)

echo "==> Done, Python and Rust agree on all parity vectors"
