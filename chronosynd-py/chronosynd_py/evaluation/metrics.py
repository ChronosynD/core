"""Detection metrics computed from per-observation drift scores. Given
benign and malicious score sets from the same fitted baseline, pick a
threshold at the target FPR and report the resulting detection rate"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chronosynd_py.core import (
    OBSERVATION_RANK,
    DimensionMismatchError,
    EmptyLearningWindowError,
    FloatArray,
    InvalidParameterError,
)


@dataclass(frozen=True, slots=True)
class DetectionMetrics:
    """Summary at a single threshold derived from benign scores. The cut is
    chosen to target `target_fpr`. `false_positive_rate` lands close but not
    exactly equal because thresholds are discrete on finite samples"""

    threshold: float
    detection_rate: float
    false_positive_rate: float
    benign_sample_count: int
    malicious_sample_count: int


def compute_detection_metrics(
    benign_scores: FloatArray,
    malicious_scores: FloatArray,
    *,
    target_fpr: float = 0.05,
) -> DetectionMetrics:
    """Pick a threshold from benign scores at `target_fpr` and measure TPR
    on malicious. The threshold is the `1 - target_fpr` quantile of
    `benign_scores`, and a strictly greater score raises an alert"""
    if benign_scores.ndim != OBSERVATION_RANK:
        raise DimensionMismatchError(
            "benign_scores must be 1-D (n_samples,), "
            f"got ndim={benign_scores.ndim} shape={benign_scores.shape}"
        )
    if malicious_scores.ndim != OBSERVATION_RANK:
        raise DimensionMismatchError(
            "malicious_scores must be 1-D (n_samples,), "
            f"got ndim={malicious_scores.ndim} shape={malicious_scores.shape}"
        )
    if benign_scores.size == 0:
        raise EmptyLearningWindowError(
            "cannot choose a threshold from zero benign scores"
        )
    if not (0.0 < target_fpr < 1.0) or not np.isfinite(target_fpr):
        raise InvalidParameterError(
            f"target_fpr must be in (0.0, 1.0), got {target_fpr}"
        )

    threshold = float(np.quantile(benign_scores, 1.0 - target_fpr))
    false_positive_rate = float(np.mean(benign_scores > threshold))
    detection_rate = (
        float(np.mean(malicious_scores > threshold))
        if malicious_scores.size > 0
        else 0.0
    )

    return DetectionMetrics(
        threshold=threshold,
        detection_rate=detection_rate,
        false_positive_rate=false_positive_rate,
        benign_sample_count=int(benign_scores.size),
        malicious_sample_count=int(malicious_scores.size),
    )
