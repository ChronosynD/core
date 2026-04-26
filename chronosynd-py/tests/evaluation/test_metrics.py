"""Unit tests for compute_detection_metrics"""

from __future__ import annotations

import numpy as np
import pytest

from chronosynd_py.core import (
    DimensionMismatchError,
    EmptyLearningWindowError,
    InvalidParameterError,
)
from chronosynd_py.evaluation.metrics import compute_detection_metrics


def test_perfectly_separated_scores_hit_full_detection() -> None:
    # Benign in [0, 1], malicious in [5, 6], any reasonable threshold splits them
    benign = np.linspace(0.0, 1.0, 100)
    malicious = np.linspace(5.0, 6.0, 100)

    metrics = compute_detection_metrics(benign, malicious, target_fpr=0.05)

    assert metrics.detection_rate == 1.0
    assert metrics.false_positive_rate <= 0.05


def test_identical_distributions_hit_detection_at_target_fpr() -> None:
    # Same distribution for benign and malicious, TPR should be close to FPR
    rng = np.random.default_rng(seed=42)
    benign = rng.normal(size=1000)
    malicious = rng.normal(size=1000)

    metrics = compute_detection_metrics(benign, malicious, target_fpr=0.1)

    assert metrics.false_positive_rate == pytest.approx(0.1, abs=0.01)
    assert metrics.detection_rate == pytest.approx(0.1, abs=0.05)


def test_threshold_is_the_benign_quantile() -> None:
    benign = np.arange(1.0, 101.0)
    malicious = np.array([200.0])

    metrics = compute_detection_metrics(benign, malicious, target_fpr=0.1)

    # 90th percentile of 1..100 is 90.1
    assert metrics.threshold == pytest.approx(90.1, rel=1e-9)


def test_rejects_2d_benign_scores() -> None:
    with pytest.raises(DimensionMismatchError):
        compute_detection_metrics(np.zeros((5, 2)), np.zeros(5))


def test_rejects_2d_malicious_scores() -> None:
    with pytest.raises(DimensionMismatchError):
        compute_detection_metrics(np.zeros(5), np.zeros((5, 2)))


def test_rejects_empty_benign() -> None:
    with pytest.raises(EmptyLearningWindowError):
        compute_detection_metrics(np.zeros(0), np.zeros(10))


@pytest.mark.parametrize("bad_fpr", [0.0, 1.0, -0.1, 1.1, float("nan")])
def test_rejects_invalid_target_fpr(bad_fpr: float) -> None:
    with pytest.raises(InvalidParameterError):
        compute_detection_metrics(np.zeros(10), np.zeros(10), target_fpr=bad_fpr)


def test_empty_malicious_scores_zero_detection() -> None:
    metrics = compute_detection_metrics(
        np.arange(100.0, dtype=np.float64), np.zeros(0), target_fpr=0.05
    )
    assert metrics.detection_rate == 0.0
    assert metrics.malicious_sample_count == 0


def test_sample_counts_reported_correctly() -> None:
    benign = np.zeros(123)
    malicious = np.ones(45)
    metrics = compute_detection_metrics(benign, malicious)
    assert metrics.benign_sample_count == 123
    assert metrics.malicious_sample_count == 45
