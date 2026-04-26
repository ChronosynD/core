"""PyO3 bindings into the Rust feature extractors, built when
`chronosynd-features` enables the `python` Cargo feature. This module
re-exports from the resulting `chronosynd_features` extension"""

from __future__ import annotations


def extract_syscall_ngrams(*_args: object, **_kwargs: object) -> object:
    """Compute syscall n-gram frequency distributions"""
    raise NotImplementedError
