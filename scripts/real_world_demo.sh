#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 PROCESS_KEY CLEAN_DURATION_SECONDS ATTACK_PAYLOAD_PATH" >&2
    exit 1
fi

process_key="$1"
clean_duration="$2"
attack_payload="$3"

if [[ ! -x "$attack_payload" ]]; then
    echo "attack payload not executable: $attack_payload" >&2
    exit 2
fi

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
work_dir="$(mktemp -d -t chronosynd-demo-XXXXXX)"
trap 'rm -rf "$work_dir"' EXIT

clean_trace="$work_dir/clean.jsonl"
attack_trace="$work_dir/attack.jsonl"
store="$work_dir/store.db"
collector_bin="$repo_root/chronosynd-rs/target/release/chronosynd-collector"
cli_bin="$repo_root/chronosynd-rs/target/release/chronosynd"

for bin in "$collector_bin" "$cli_bin"; do
    if [[ ! -x "$bin" ]]; then
        echo "missing $bin, build with: cd chronosynd-rs && cargo build --release --features bpf" >&2
        exit 3
    fi
done

echo "==> 1/4 capturing clean trace for ${process_key} (${clean_duration}s)"
"$repo_root/scripts/capture_baseline.sh" "$clean_duration" "$clean_trace" "$store"

echo "==> 2/4 fitting Sediment baseline from clean trace"
"$cli_bin" --store "$store" fit-from-trace "$process_key" \
    --input "$clean_trace" \
    --estimator sediment \
    --trim-fraction 0.3

echo "==> 3/4 capturing trace while attack payload runs"
attack_duration=30
sudo timeout "${attack_duration}s" "$collector_bin" \
    --bpf \
    --record "$attack_trace" \
    --store "$store" \
    &
collector_pid=$!
sleep 2
"$attack_payload" || true
wait "$collector_pid" || true

echo "==> 4/4 replaying attack trace, drift scores follow"
"$cli_bin" --store "$store" baseline show "$process_key"
echo
echo "trace files retained in $work_dir until script exits, copy now if needed"
ls -la "$work_dir"
