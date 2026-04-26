#!/usr/bin/env bash
# spawn_storm.sh, simulates the syscall pattern of a process spawning many
# short-lived helper commands, common in post-exploitation reconnaissance
# scripts that pipe through awk, sed, grep, and similar utilities

set -euo pipefail

count="${1:-100}"

if ! [[ "$count" =~ ^[0-9]+$ ]]; then
    echo "count must be a non-negative integer, got '$count'" >&2
    exit 1
fi

for ((i = 0; i < count; i++)); do
    echo "$i" | awk '{print $1}' >/dev/null
    /usr/bin/true
    /usr/bin/false || true
done
