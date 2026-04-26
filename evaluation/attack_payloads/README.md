<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright 2026 The ChronosynD Authors, see LICENSE-CC-BY-4.0 -->

# Attack payloads

Behavioral fixtures that drive the real-data demonstration of the detector. Each payload is a small shell script whose runtime syscall pattern is distinct from a steady-state web server, so a baseline fitted on the clean process drifts when the payload runs alongside it.

These are simulators, not exploitation. Every script reads files the invoking user already has access to or runs harmless coreutils. None of them touch the network or write outside their own scratch space. They exist so the paper can show real drift signal under plausible attacker behavior without packaging actual offensive code.

## Bundled payloads

| Script | Pattern simulated | What it does |
|---|---|---|
| [`etc_recon.sh`](etc_recon.sh) | Configuration harvesting, openat-heavy and read-heavy | Reads up to N regular files under `/etc` or any directory passed as the second argument, suppressing permission errors |
| [`spawn_storm.sh`](spawn_storm.sh) | Post-exploitation script-kiddie patterns, exec-heavy and process-heavy | Spawns N short-lived helper processes in a loop |

Each script takes the iteration count as its first argument and defaults to a sensible value, so a quick demo run is `./etc_recon.sh` with no arguments.

## How they fit into evaluation

The end-to-end demo is in [`scripts/real_world_demo.sh`](../../scripts/real_world_demo.sh):

1. Capture a clean trace of the target process for some duration.
2. Fit a Sediment baseline from the clean trace.
3. Capture a second trace while one of these payloads runs.
4. Replay the attack trace and report how far the drift score moves.

A baseline fit on real nginx behavior should produce low scores on a second clean nginx capture and high scores on a capture taken while `etc_recon.sh` or `spawn_storm.sh` runs in the same process tree. The drift signal is large by design because these are behavioral simulators, not nginx-shaped attackers. On a real attacker the drift would be smaller. The paper's argument is about the algorithm's robustness under poisoning, not about catching subtle real-world adversaries with these specific scripts.

## Adding a payload

A new payload should be a single shell or Python script with no installed dependencies beyond what a typical Linux box has, accept a `count` argument as the first positional, and document the syscall pattern it represents in a leading comment block. Add a row to the table above when committing.
