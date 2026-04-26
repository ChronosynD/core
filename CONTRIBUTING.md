<!-- SPDX-License-Identifier: CC-BY-4.0 -->
<!-- Copyright 2026 The ChronosynD Authors, see LICENSE-CC-BY-4.0 -->

# Contributing to ChronosynD

Thanks for your interest. Before opening a pull request, please read the IP terms below in full. They are non-negotiable. Submitting a PR is your acknowledgment that you have read and accepted them.

## Copyright assignment

**Every contribution to ChronosynD is assigned to the project lead, as a condition of merging.** This is a one-way assignment, not a license grant. By submitting a contribution (code, prose, figures, documentation, scripts, configuration, or any other material) through any mechanism (pull request, patch, email, issue comment, gist, voice transcript, or any other channel), you irrevocably and unconditionally:

1. Assign all right, title, and interest in your contribution to the project lead, including without limitation the entire copyright, all moral rights to the maximum extent permitted by applicable law, all related neighboring rights, and any rights to the underlying ideas, algorithms, and design choices contained in the contribution.

2. Waive any and all claims you may otherwise have against the project lead or the ChronosynD project arising out of your contribution, including but not limited to attribution claims beyond the credit conventions described below, royalty claims, or claims to control downstream use.

3. Warrant that you have the legal right to make this assignment. If your contribution incorporates work originally created by your employer, by another individual, or under a contract that vests rights in a third party, you must obtain and provide written confirmation that the third party also assigns those rights to the project lead before the contribution can be merged. Submitting a contribution you do not have the right to assign is a material breach of these terms.

4. Acknowledge that the project lead may, at sole discretion, relicense ChronosynD or any portion of it under different license terms in the future, including proprietary terms, and that you have no veto over such relicensing.

This assignment takes effect at the moment you submit the contribution, regardless of whether the contribution is ultimately merged.

The project's outbound license remains the dual Apache-2.0 / CC-BY-4.0 split documented in [`README.md`](README.md) and [`NOTICE`](NOTICE). The assignment above does not change what users of ChronosynD receive; it only consolidates the right to set those terms in a single party.

## Attribution

Contributors are credited in the project's git history and, for substantive contributions, in a forthcoming `CONTRIBUTORS.md`. Credit is a courtesy, not a contract right. Removal of credit is at the project lead's sole discretion. The CC-BY-4.0 attribution requirements that apply to downstream users of the written materials do not apply to the project lead's internal handling of contributor credit.

## What counts as a contribution

Anything that ends up in the repository's tracked files. This includes:

- Source code changes in `chronosynd-py/`, `chronosynd-rs/`, `scripts/`, `evaluation/attack_payloads/`.
- Prose changes in `paper/`, `docs/`, README files, `NOTICE`, `CITATION.cff`.
- New experiments, attack payloads, baseline estimators, feature extractors.
- Test cases, fixtures, parity vectors.
- Build configuration, CI workflows, lint rules.
- Bug reports that include reproduction code or proposed fixes (the report itself is a contribution).

Filing an issue without proposed code is not a contribution under these terms; ordinary issue triage applies.

## Quality gates

Every PR must:

- Pass the Python gates: `cd chronosynd-py && uv run ruff check && uv run mypy && uv run pytest`.
- Pass the Rust gates: `cd chronosynd-rs && cargo build --workspace && cargo test --workspace && cargo clippy --workspace --all-targets -- -D warnings`.
- Pass the cross-language parity check: `bash scripts/check_parity.sh`.
- For paper changes, the build must still succeed: `cd paper && make`.
- Hold the [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) and [`docs/evaluation_protocol.md`](docs/evaluation_protocol.md) contracts. If a PR breaks one of these on purpose, the threat model or protocol document must be updated in the same PR with a clear rationale.

PRs that do not pass the gates will not be reviewed.

## Out of scope

- Removing the dual-license structure. The Apache-2.0 / CC-BY-4.0 split is a deliberate decision documented in `README.md` and `NOTICE`.
- Removing the cross-language parity tests. The Python reference and Rust port must stay bit-equivalent.
- Adding telemetry, network calls, or third-party API integrations to the runtime. ChronosynD is local-only by construction.

If you want to discuss a change in any of these areas, file an issue first.

## Reporting security vulnerabilities

See [`SECURITY.md`](SECURITY.md). Do not file public issues for vulnerabilities.

## How to submit a contribution

1. Fork the repository at https://github.com/ChronosynD/core.
2. Make your changes on a branch.
3. Run every quality gate listed above and confirm all pass.
4. Open a pull request with a description that explains both the WHAT (what changed) and the WHY (why it matters).
5. Respond to review comments. Reviews are direct; treat them as such.

By opening the pull request, you confirm that you have read this document and that the copyright assignment in the section above takes effect.
