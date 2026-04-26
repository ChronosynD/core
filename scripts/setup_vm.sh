#!/usr/bin/env bash
# Bootstrap a Linux dev environment for ChronosynD on Ubuntu 24.04 or WSL2.
# Installs the BPF toolchain, Rust, uv, the Python deps, and builds the
# release binaries with the `bpf` feature enabled. Run from the repo root.

set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
    echo "setup_vm.sh targets Linux, current platform is $(uname -s)" >&2
    exit 1
fi

if ! command -v sudo >/dev/null 2>&1; then
    echo "sudo not found, this script needs root via sudo" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"

echo "==> apt: kernel headers, BPF toolchain, build tools"
sudo apt update
sudo apt install -y \
    build-essential \
    clang \
    libelf-dev \
    linux-headers-generic \
    linux-tools-common \
    linux-tools-generic \
    pkg-config \
    git \
    curl \
    ca-certificates

if ! clang -target bpf -E -x c /dev/null > /dev/null 2>&1; then
    echo "clang lacks the BPF target, the BPF crate will not build" >&2
    exit 3
fi

echo "==> Rust toolchain"
if ! command -v cargo >/dev/null 2>&1; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --default-toolchain stable
fi
[[ -f "$HOME/.cargo/env" ]] && source "$HOME/.cargo/env"
rustc --version
cargo --version

echo "==> uv (Python package manager)"
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "==> Python deps for the research side"
(cd chronosynd-py && uv sync --extra dev --extra viz)

if [[ ! -f "chronosynd-rs/crates/bpf/bpf/vmlinux.h" ]]; then
    echo "==> Generating vmlinux.h from running kernel BTF"
    if ! command -v bpftool >/dev/null 2>&1; then
        echo "bpftool not found in PATH, install via 'sudo apt install bpftrace' \
or build from linux-tools-\$(uname -r), then rerun" >&2
        exit 4
    fi
    sudo bpftool btf dump file /sys/kernel/btf/vmlinux format c \
        > chronosynd-rs/crates/bpf/bpf/vmlinux.h
fi

echo "==> Rust workspace build (release, with BPF feature)"
(cd chronosynd-rs && cargo build --release --features bpf)

echo
echo "==> Done. Verified binaries:"
ls -la chronosynd-rs/target/release/chronosynd \
       chronosynd-rs/target/release/chronosynd-collector

cat <<EOF

Next steps:
  - Run tests:       (cd chronosynd-py && uv run pytest -q) && (cd chronosynd-rs && cargo test --workspace)
  - Reproduce paper: bash scripts/reproduce_all.sh
  - Real-data demo:  sudo bash scripts/real_world_demo.sh <process_key> <duration_seconds> <attack_payload>
EOF
