"""Unit tests for the Python SyscallNgramExtractor"""

from __future__ import annotations

import pytest

from chronosynd_py.core import InvalidParameterError
from chronosynd_py.features.syscall_ngram import SyscallNgramExtractor
from chronosynd_py.features.vocab import default_syscall_vocab


def test_feature_dim_is_vocab_plus_one_for_other_bucket() -> None:
    extractor = SyscallNgramExtractor(vocab=[1, 2, 3], window_size=4)
    assert extractor.feature_dim == 4
    assert extractor.other_bucket_index == 3


def test_window_emits_only_when_full() -> None:
    extractor = SyscallNgramExtractor(vocab=[1, 2], window_size=3)
    assert extractor.accumulate("p", 1) is None
    assert extractor.accumulate("p", 2) is None
    emitted = extractor.accumulate("p", 1)
    assert emitted is not None
    assert emitted.process_key == "p"
    assert emitted.feature_vector == [2 / 3, 1 / 3, 0.0]


def test_out_of_vocab_lands_in_other_bucket() -> None:
    extractor = SyscallNgramExtractor(vocab=[1, 2], window_size=2)
    extractor.accumulate("p", 1)
    emitted = extractor.accumulate("p", 9999)  # not in vocab
    assert emitted is not None
    assert emitted.feature_vector == [0.5, 0.0, 0.5]


def test_per_process_windows_are_disjoint() -> None:
    extractor = SyscallNgramExtractor(vocab=[1, 2], window_size=2)
    # interleave events from two processes, each closes its own window
    assert extractor.accumulate("a", 1) is None
    assert extractor.accumulate("b", 2) is None
    a = extractor.accumulate("a", 2)
    b = extractor.accumulate("b", 1)
    assert a is not None
    assert a.process_key == "a"
    assert a.feature_vector == [0.5, 0.5, 0.0]
    assert b is not None
    assert b.process_key == "b"
    assert b.feature_vector == [0.5, 0.5, 0.0]


def test_window_resets_after_emission() -> None:
    extractor = SyscallNgramExtractor(vocab=[1, 2], window_size=2)
    extractor.accumulate("p", 1)
    extractor.accumulate("p", 1)
    second = extractor.accumulate("p", 2)
    assert second is None
    third = extractor.accumulate("p", 2)
    assert third is not None
    assert third.feature_vector == [0.0, 1.0, 0.0]


def test_reset_process_drops_pending_window() -> None:
    extractor = SyscallNgramExtractor(vocab=[1, 2], window_size=3)
    extractor.accumulate("p", 1)
    extractor.accumulate("p", 1)
    extractor.reset_process("p")
    # next event starts fresh, single read does not yet emit
    assert extractor.accumulate("p", 1) is None


def test_tracked_process_count_grows_with_distinct_keys() -> None:
    extractor = SyscallNgramExtractor(vocab=[1], window_size=4)
    extractor.accumulate("a", 1)
    extractor.accumulate("b", 1)
    extractor.accumulate("c", 1)
    assert extractor.tracked_process_count == 3


def test_default_vocab_has_expected_size() -> None:
    vocab = default_syscall_vocab()
    assert len(vocab) == 26
    assert vocab[0] == 0  # read
    assert vocab[-1] == 257  # openat


def test_constructor_rejects_zero_window() -> None:
    with pytest.raises(InvalidParameterError):
        SyscallNgramExtractor(vocab=[1, 2], window_size=0)


def test_constructor_rejects_duplicate_vocab() -> None:
    with pytest.raises(InvalidParameterError):
        SyscallNgramExtractor(vocab=[1, 2, 1], window_size=4)
