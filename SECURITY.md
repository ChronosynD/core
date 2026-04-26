<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright 2026 The ChronosynD Authors, see LICENSE-CC-BY-4.0 -->

# Security policy

ChronosynD is a research artifact, not a deployed product, but it builds on a privileged BPF runtime and persists tamper-evident state. 

## Reporting a vulnerability

**Preferred path:** GitHub's private vulnerability reporting at https://github.com/ChronosynD/core/security/advisories/new. 

Do not file public GitHub issues for security vulnerabilities.

## Scope

**In scope:**

- Memory safety, panics, or undefined behavior in the Rust runtime (`chronosynd-rs/`).
- Logic errors in the Sediment algorithm or the parity tests that mask incorrect output.
- Bypasses of the SHA-256 hash chain in the storage layer.
- Privilege escalation through the BPF probe or its userspace loader.
- Information disclosure from the BPF ring buffer or recorded JSONL traces.
- Issues in the Python research reference (`chronosynd-py/`) that produce wrong scientific output.

**Out of scope:**

- Vulnerabilities in upstream dependencies. Report those to the dependency's maintainer; we'll bump the version when a fix is released.
- Theoretical attacks already documented in the paper's threat model (`docs/THREAT_MODEL.md`) or limitations section.
- Social-engineering and physical-access attacks against operators.
- Issues in tooling-time dependencies (latex, uv, cargo) that don't touch the runtime.
- Denial-of-service through legitimate but expensive operations (e.g., a huge JSONL recording that fills the disk).

## Coordinated disclosure

For unfixed issues, we ask reporters not to discuss the vulnerability publicly until either (a) a patched release is published, or (b) 90 days have passed from acknowledgment, whichever comes first. After that the reporter is free to disclose.

We will credit the reporter by name (or pseudonym, on request) in the published advisory unless they ask not to be.

## Intellectual property

The text of a vulnerability report (description, reproduction steps, observed behavior, suggested mitigations as prose) carries no IP claim from this project. The reporter retains whatever rights they have in their own writing.

If a report includes code that the reporter intends the project to incorporate (proposed patches, exploit proofs of concept, regression test cases, harness modifications, or any other creative work in code form), that code is treated as a contribution to the project. By submitting it through any channel reachable from this security policy, you irrevocably assign copyright in that code to the project lead under the same terms as a pull request, documented in [`CONTRIBUTING.md`](CONTRIBUTING.md). Submit code only if you have the legal right to assign it.

A report that contains no code is not a contribution and the assignment does not attach. If you wish to keep proof-of-concept code under your own copyright while still reporting the vulnerability, say so in the report and describe the issue in prose only.
