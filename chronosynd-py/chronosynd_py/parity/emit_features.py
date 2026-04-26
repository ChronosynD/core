"""Emit feature-extractor parity test vectors for the Rust port to
validate against. Mirrors `chronosynd_py.parity.emit` but for the
SyscallNgramExtractor side of the cross-implementation contract"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from chronosynd_py.features.syscall_ngram import SyscallNgramExtractor
from chronosynd_py.features.vocab import default_syscall_vocab

SCHEMA_VERSION = 1

# (case_name, window_size, vocab_kind, event_count, seed)
_CASES: tuple[tuple[str, int, str, int, int], ...] = (
    ("default_w16_n200", 16, "default", 200, 1),
    ("default_w8_n128", 8, "default", 128, 2),
    ("default_w32_n320", 32, "default", 320, 3),
    ("small_w4_n40", 4, "small", 40, 4),
)

_PROCESS_KEYS: tuple[str, ...] = ("nginx", "sshd", "cron")


def _vocab_for(kind: str) -> list[int]:
    if kind == "default":
        return default_syscall_vocab()
    if kind == "small":
        return [0, 1, 2, 3]
    raise ValueError(f"unknown vocab kind {kind}")


def _build_event_stream(
    *, vocab: list[int], event_count: int, seed: int
) -> list[tuple[str, int]]:
    """Deterministic event stream mixing in-vocab and out-of-vocab syscalls
    across multiple process keys. The seed pins the order across emit runs"""
    rng = np.random.default_rng(seed=seed)
    # 80% in-vocab, 20% out-of-vocab so the "other" bucket exercises too
    pool = [*vocab, 9_999, 10_001, 10_002, 10_003, 10_004]
    weights = np.array([0.8 / len(vocab)] * len(vocab) + [0.2 / 5] * 5)
    weights /= weights.sum()
    syscalls = rng.choice(pool, size=event_count, replace=True, p=weights)
    process_choices = rng.choice(len(_PROCESS_KEYS), size=event_count)
    return [
        (_PROCESS_KEYS[int(p_idx)], int(syscall))
        for p_idx, syscall in zip(process_choices, syscalls, strict=True)
    ]


def _emit_case(
    *, name: str, window_size: int, vocab: list[int], events: list[tuple[str, int]]
) -> dict[str, Any]:
    extractor = SyscallNgramExtractor(vocab=vocab, window_size=window_size)
    emissions: list[dict[str, Any]] = []
    for index, (process_key, syscall_nr) in enumerate(events):
        result = extractor.accumulate(process_key, syscall_nr)
        if result is not None:
            emissions.append(
                {
                    "event_index": index,
                    "process_key": result.process_key,
                    "feature_vector": result.feature_vector,
                }
            )
    return {
        "name": name,
        "window_size": window_size,
        "vocab": vocab,
        "events": [{"process_key": pk, "syscall_nr": nr} for pk, nr in events],
        "emissions": emissions,
    }


def _output_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return (
        repo_root
        / "chronosynd-rs"
        / "crates"
        / "features"
        / "tests"
        / "parity_vectors.json"
    )


def build_cases() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for name, window_size, vocab_kind, event_count, seed in _CASES:
        vocab = _vocab_for(vocab_kind)
        events = _build_event_stream(
            vocab=vocab, event_count=event_count, seed=seed
        )
        cases.append(
            _emit_case(
                name=name, window_size=window_size, vocab=vocab, events=events
            )
        )
    return cases


def main() -> None:
    cases = build_cases()
    output = {"schema_version": SCHEMA_VERSION, "cases": cases}
    path = _output_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(f"feature parity emit: wrote {len(cases)} cases to {path}")


if __name__ == "__main__":
    main()
