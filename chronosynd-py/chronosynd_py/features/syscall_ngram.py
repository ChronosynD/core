"""Disjoint-window 1-gram syscall histogram extractor, parity-graded
mirror of `chronosynd-rs/crates/features/src/syscall_ngram.rs`. Both
sides assert byte-identical feature vectors on the shared test vectors"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from chronosynd_py.core import (
    FloatArray,
    InvalidParameterError,
)


@dataclass(frozen=True, slots=True)
class EmittedFeatures:
    """A feature vector emitted when a per-process window closes.
    `process_key` identifies the process and `feature_vector` is the dense
    numeric representation the scoring engine consumes"""

    process_key: str
    feature_vector: list[float]


class SyscallNgramExtractor:
    """Disjoint-window 1-gram histogram extractor over a fixed syscall
    vocabulary. The emitted feature vector has dimension `len(vocab) + 1`,
    with the trailing slot counting out-of-vocab syscalls"""

    def __init__(self, vocab: Sequence[int], window_size: int) -> None:
        """Build with an explicit syscall vocabulary and disjoint window
        size. Duplicates in `vocab` raise so the feature contract stays
        unambiguous"""
        if window_size < 1:
            raise InvalidParameterError(
                f"window_size must be at least 1, got {window_size}"
            )
        vocab_index: dict[int, int] = {}
        for idx, syscall_nr in enumerate(vocab):
            if syscall_nr in vocab_index:
                raise InvalidParameterError(
                    f"duplicate vocabulary entry {syscall_nr} at index {idx}"
                )
            vocab_index[syscall_nr] = idx
        self._vocab_index = vocab_index
        self._feature_dim = len(vocab) + 1
        self._window_size = window_size
        self._per_process: dict[str, list[float]] = {}
        self._pending: dict[str, int] = {}

    @property
    def feature_dim(self) -> int:
        """Dimension of every emitted feature vector, constant across calls"""
        return self._feature_dim

    @property
    def other_bucket_index(self) -> int:
        """Index in the feature vector reserved for syscalls outside vocab"""
        return self._feature_dim - 1

    @property
    def tracked_process_count(self) -> int:
        """Number of processes the extractor is currently accumulating for"""
        return len(self._per_process)

    def accumulate(
        self, process_key: str, syscall_nr: int
    ) -> EmittedFeatures | None:
        """Feed one observed syscall. Returns an `EmittedFeatures` when the
        per-process window closes, `None` while the window is still open"""
        counts = self._per_process.get(process_key)
        if counts is None:
            counts = [0.0] * self._feature_dim
            self._per_process[process_key] = counts

        bucket = self._vocab_index.get(syscall_nr, self._feature_dim - 1)
        counts[bucket] += 1.0

        pending = self._pending.get(process_key, 0) + 1
        self._pending[process_key] = pending

        if pending < self._window_size:
            return None

        denom = float(self._window_size)
        normalized = [count / denom for count in counts]
        emitted = EmittedFeatures(
            process_key=process_key, feature_vector=normalized
        )

        self._per_process[process_key] = [0.0] * self._feature_dim
        self._pending[process_key] = 0
        return emitted

    def reset_process(self, process_key: str) -> None:
        """Discard any in-progress window for a process. The next event for
        that key starts a fresh window"""
        self._per_process.pop(process_key, None)
        self._pending.pop(process_key, None)


def feature_vector_to_array(features: EmittedFeatures) -> FloatArray:
    """Convert an `EmittedFeatures` into a 1-D numpy array. Kept here so
    callers that mix the harness with the extractor do not have to know
    the storage shape `feature_vector` uses internally"""
    return np.asarray(features.feature_vector, dtype=np.float64)
