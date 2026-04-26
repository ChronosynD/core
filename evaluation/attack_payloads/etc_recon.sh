#!/usr/bin/env bash
# etc_recon.sh, simulates an attacker harvesting host configuration by
# opening and reading many files under /etc, the resulting syscall mix is
# heavy on openat and read calls, distinct from a typical web server load

set -euo pipefail

count="${1:-200}"
target_dir="${2:-/etc}"

if ! [[ "$count" =~ ^[0-9]+$ ]]; then
    echo "count must be a non-negative integer, got '$count'" >&2
    exit 1
fi

find "$target_dir" -maxdepth 3 -type f -readable 2>/dev/null \
    | head -n "$count" \
    | while read -r path; do
        cat -- "$path" >/dev/null 2>&1 || true
    done
