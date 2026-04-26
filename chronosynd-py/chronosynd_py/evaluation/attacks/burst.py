"""Burst poisoning attack. The adversary concentrates the entire injection
budget in a contiguous slice of the learning window so a sub-window
consensus filter has a chance to detect the disagreement"""

from __future__ import annotations

import numpy as np

from chronosynd_py.core import (
    BATCH_RANK,
    OBSERVATION_RANK,
    DimensionMismatchError,
    InvalidObservationError,
    InvalidParameterError,
    Observation,
    ObservationBatch,
    PoisonedTrace,
)

_MIN_BUDGET = 0.0
_MAX_BUDGET = 1.0


def inject_burst(
    benign_trace: ObservationBatch,
    target_attack_behavior: Observation,
    budget_fraction: float,
    *,
    burst_position: float | None = None,
    jitter_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> PoisonedTrace:
    """Replace a contiguous slice of `benign_trace` with `injection_count`
    copies of `target_attack_behavior` (with optional Gaussian jitter).
    `burst_position` in [0, 1] picks where the slice starts. The default is
    a uniformly random position drawn from `rng`"""
    _validate_inputs(
        benign_trace=benign_trace,
        target_attack_behavior=target_attack_behavior,
        budget_fraction=budget_fraction,
        burst_position=burst_position,
        jitter_scale=jitter_scale,
    )

    generator = rng if rng is not None else np.random.default_rng()
    benign_count = benign_trace.shape[0]
    feature_dim = benign_trace.shape[1]
    injection_count = round(benign_count * budget_fraction)

    if injection_count == 0:
        return PoisonedTrace(
            observations=benign_trace.copy(),
            is_adversarial=np.zeros(benign_count, dtype=np.bool_),
        )

    if injection_count >= benign_count:
        # full takeover, no benign rows survive
        injections = _build_injections(
            target_attack_behavior=target_attack_behavior,
            injection_count=benign_count,
            feature_dim=feature_dim,
            jitter_scale=jitter_scale,
            generator=generator,
        )
        return PoisonedTrace(
            observations=injections,
            is_adversarial=np.ones(benign_count, dtype=np.bool_),
        )

    max_start = benign_count - injection_count
    if burst_position is None:
        start_index = int(generator.integers(0, max_start + 1))
    else:
        start_index = round(burst_position * max_start)

    injections = _build_injections(
        target_attack_behavior=target_attack_behavior,
        injection_count=injection_count,
        feature_dim=feature_dim,
        jitter_scale=jitter_scale,
        generator=generator,
    )

    observations = benign_trace.copy()
    observations[start_index : start_index + injection_count] = injections
    labels = np.zeros(benign_count, dtype=np.bool_)
    labels[start_index : start_index + injection_count] = True

    return PoisonedTrace(
        observations=observations,
        is_adversarial=labels,
    )


def _validate_inputs(
    *,
    benign_trace: ObservationBatch,
    target_attack_behavior: Observation,
    budget_fraction: float,
    burst_position: float | None,
    jitter_scale: float,
) -> None:
    if benign_trace.ndim != BATCH_RANK:
        raise DimensionMismatchError(
            "benign_trace must be 2-D (n_samples, n_features), "
            f"got ndim={benign_trace.ndim} shape={benign_trace.shape}"
        )
    if target_attack_behavior.ndim != OBSERVATION_RANK:
        raise DimensionMismatchError(
            "target_attack_behavior must be 1-D (n_features,), "
            f"got ndim={target_attack_behavior.ndim} shape={target_attack_behavior.shape}"
        )
    if target_attack_behavior.shape[0] != benign_trace.shape[1]:
        raise DimensionMismatchError(
            f"target feature dim {target_attack_behavior.shape[0]} "
            f"does not match benign trace feature dim {benign_trace.shape[1]}"
        )
    if not (_MIN_BUDGET <= budget_fraction <= _MAX_BUDGET):
        raise InvalidParameterError(
            f"budget_fraction must be in [{_MIN_BUDGET}, {_MAX_BUDGET}], got {budget_fraction}"
        )
    if burst_position is not None and not (0.0 <= burst_position <= 1.0):
        raise InvalidParameterError(
            f"burst_position must be in [0.0, 1.0] or None, got {burst_position}"
        )
    if not np.isfinite(jitter_scale) or jitter_scale < 0.0:
        raise InvalidParameterError(
            f"jitter_scale must be finite and non-negative, got {jitter_scale}"
        )
    if not np.all(np.isfinite(benign_trace)):
        raise InvalidObservationError("benign_trace contains non-finite values, NaN or inf")
    if not np.all(np.isfinite(target_attack_behavior)):
        raise InvalidObservationError(
            "target_attack_behavior contains non-finite values, NaN or inf"
        )


def _build_injections(
    *,
    target_attack_behavior: Observation,
    injection_count: int,
    feature_dim: int,
    jitter_scale: float,
    generator: np.random.Generator,
) -> ObservationBatch:
    if jitter_scale > 0.0:
        noise = generator.normal(scale=jitter_scale, size=(injection_count, feature_dim))
        return target_attack_behavior[np.newaxis, :] + noise
    return np.tile(target_attack_behavior, (injection_count, 1))
