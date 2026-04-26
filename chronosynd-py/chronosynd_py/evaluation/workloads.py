"""Synthetic workload factories for the evaluation harness. Real
deployments will feed pre-recorded BPF traces, but these deterministic
distributions let the research side iterate on estimators today"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from chronosynd_py.core import (
    OBSERVATION_RANK,
    DimensionMismatchError,
    InvalidParameterError,
    Observation,
    ObservationBatch,
)

WorkloadFactory = Callable[[np.random.Generator, int], ObservationBatch]


def isotropic_gaussian(
    feature_dim: int,
    *,
    loc: float = 0.0,
    scale: float = 1.0,
) -> WorkloadFactory:
    """Factory for a workload drawn from N(loc, scale) in each of `feature_dim` features"""
    if feature_dim < 1:
        raise InvalidParameterError(
            f"feature_dim must be at least 1, got {feature_dim}"
        )
    if not np.isfinite(loc):
        raise InvalidParameterError(f"loc must be finite, got {loc}")
    if not np.isfinite(scale) or scale <= 0.0:
        raise InvalidParameterError(
            f"scale must be finite and positive, got {scale}"
        )

    def _source(rng: np.random.Generator, sample_count: int) -> ObservationBatch:
        if sample_count < 0:
            raise InvalidParameterError(
                f"sample_count must be non-negative, got {sample_count}"
            )
        return rng.normal(loc=loc, scale=scale, size=(sample_count, feature_dim))

    return _source


def jittered_target(
    target: Observation,
    *,
    jitter_scale: float = 0.1,
) -> WorkloadFactory:
    """Factory for observations placed near `target` with isotropic Gaussian jitter"""
    if target.ndim != OBSERVATION_RANK:
        raise DimensionMismatchError(
            "target must be 1-D (n_features,), "
            f"got ndim={target.ndim} shape={target.shape}"
        )
    if not np.all(np.isfinite(target)):
        raise InvalidParameterError("target must contain only finite values")
    if not np.isfinite(jitter_scale) or jitter_scale < 0.0:
        raise InvalidParameterError(
            f"jitter_scale must be finite and non-negative, got {jitter_scale}"
        )

    feature_dim = target.shape[0]

    def _source(rng: np.random.Generator, sample_count: int) -> ObservationBatch:
        if sample_count < 0:
            raise InvalidParameterError(
                f"sample_count must be non-negative, got {sample_count}"
            )
        if jitter_scale == 0.0:
            return np.tile(target, (sample_count, 1))
        noise = rng.normal(scale=jitter_scale, size=(sample_count, feature_dim))
        return target[np.newaxis, :] + noise

    return _source


def heterogeneous_gaussian(
    mean: Sequence[float] | np.ndarray,
    scale: Sequence[float] | np.ndarray,
) -> WorkloadFactory:
    """Factory for a per-feature Gaussian with heterogeneous means and
    scales. Mirrors `isotropic_gaussian` but each feature has its own
    location and spread, useful when features live on different scales"""
    mean_array = _validate_vector(np.asarray(mean, dtype=np.float64), "mean")
    scale_array = _validate_vector(np.asarray(scale, dtype=np.float64), "scale")
    if mean_array.shape != scale_array.shape:
        raise DimensionMismatchError(
            f"mean shape {mean_array.shape} does not match scale shape {scale_array.shape}"
        )
    if np.any(scale_array <= 0.0):
        raise InvalidParameterError("every entry of scale must be strictly positive")

    feature_dim = mean_array.shape[0]

    def _source(rng: np.random.Generator, sample_count: int) -> ObservationBatch:
        if sample_count < 0:
            raise InvalidParameterError(
                f"sample_count must be non-negative, got {sample_count}"
            )
        # rng.normal broadcasts loc and scale across the (sample, feature) shape
        return rng.normal(loc=mean_array, scale=scale_array, size=(sample_count, feature_dim))

    return _source


def gaussian_mixture(
    components: Sequence[tuple[float, Sequence[float] | np.ndarray, Sequence[float] | np.ndarray]],
) -> WorkloadFactory:
    """Factory for a multimodal Gaussian mixture. `components` is a sequence
    of `(weight, mean, scale)` triples with normalized weights and a shared
    feature dimension. Used for benign distributions with natural modes"""
    if len(components) == 0:
        raise InvalidParameterError("gaussian_mixture requires at least one component")

    weights = np.asarray([entry[0] for entry in components], dtype=np.float64)
    if np.any(weights <= 0.0):
        raise InvalidParameterError("every component weight must be strictly positive")
    if not np.all(np.isfinite(weights)):
        raise InvalidParameterError("component weights must be finite")
    weights /= weights.sum()

    means = [
        _validate_vector(np.asarray(entry[1], dtype=np.float64), "mean")
        for entry in components
    ]
    scales = [
        _validate_vector(np.asarray(entry[2], dtype=np.float64), "scale")
        for entry in components
    ]
    feature_dim = means[0].shape[0]
    for idx, (mean_array, scale_array) in enumerate(zip(means, scales, strict=True)):
        if mean_array.shape[0] != feature_dim:
            raise DimensionMismatchError(
                f"component {idx} mean has dim {mean_array.shape[0]}, expected {feature_dim}"
            )
        if scale_array.shape[0] != feature_dim:
            raise DimensionMismatchError(
                f"component {idx} scale has dim {scale_array.shape[0]}, expected {feature_dim}"
            )
        if np.any(scale_array <= 0.0):
            raise InvalidParameterError(
                f"component {idx} scale entries must be strictly positive"
            )

    def _source(rng: np.random.Generator, sample_count: int) -> ObservationBatch:
        if sample_count < 0:
            raise InvalidParameterError(
                f"sample_count must be non-negative, got {sample_count}"
            )
        if sample_count == 0:
            return np.zeros((0, feature_dim), dtype=np.float64)
        assignments = rng.choice(len(components), size=sample_count, p=weights)
        out = np.empty((sample_count, feature_dim), dtype=np.float64)
        for component_idx in range(len(components)):
            mask = assignments == component_idx
            count = int(mask.sum())
            if count == 0:
                continue
            out[mask] = rng.normal(
                loc=means[component_idx],
                scale=scales[component_idx],
                size=(count, feature_dim),
            )
        return out

    return _source


def _validate_vector(array: np.ndarray, name: str) -> np.ndarray:
    if array.ndim != OBSERVATION_RANK:
        raise DimensionMismatchError(
            f"{name} must be 1-D, got ndim={array.ndim} shape={array.shape}"
        )
    if not np.all(np.isfinite(array)):
        raise InvalidParameterError(f"{name} must contain only finite values")
    if array.shape[0] == 0:
        raise InvalidParameterError(f"{name} must have at least one entry")
    return array
