"""Offline drift scoring over pre-recorded traces. A thin wrapper around
`Baseline.score_batch` that grows to handle trace preprocessing like window
reshaping and per-process grouping without churning callers"""

from __future__ import annotations

from chronosynd_py.baseline.base import Baseline
from chronosynd_py.core import FloatArray, ObservationBatch


def score_trace(baseline: Baseline, trace: ObservationBatch) -> FloatArray:
    """Score every observation in `trace` against the fitted `baseline`"""
    return baseline.score_batch(trace)
