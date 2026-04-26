#!/usr/bin/env bash
# capture_baseline.sh, run the collector in record mode for a bounded
# duration and write a JSONL trace, the trace can then be replayed with
# `chronosynd fit-from-trace` to produce a real-data baseline

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 DURATION_SECONDS OUTPUT_PATH [STORE_PATH]" >&2
    exit 1
fi

duration="$1"
output="$2"
store="${3:-$(mktemp -t chronosynd-capture-XXXXXX.db)}"

if ! [[ "$duration" =~ ^[0-9]+$ ]]; then
    echo "DURATION_SECONDS must be a non-negative integer, got '$duration'" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
collector_bin="$repo_root/chronosynd-rs/target/release/chronosynd-collector"

if [[ ! -x "$collector_bin" ]]; then
    echo "collector binary not found at $collector_bin" >&2
    echo "build it first with:" >&2
    echo "    cd chronosynd-rs && cargo build --release --features bpf" >&2
    exit 3
fi

mkdir -p "$(dirname "$output")"

echo "==> capturing for ${duration}s to $output"
echo "==> using ephemeral store $store"
set +e
sudo timeout "${duration}s" "$collector_bin" \
    --bpf \
    --record "$output" \
    --store "$store"
status=$?
set -e

if [[ $status -ne 0 && $status -ne 124 ]]; then
    echo "capture failed with status $status" >&2
    exit "$status"
fi

if [[ ! -s "$output" ]]; then
    echo "capture produced no events, did the workload actually run?" >&2
    exit 4
fi

event_count=$(wc -l < "$output")
echo "==> captured ${event_count} events to $output"
echo "==> next: chronosynd fit-from-trace <process_key> --input $output"
