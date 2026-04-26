"""Targeted poisoning. The worst-case white-box adversary with full
knowledge of the detector's feature layout and baseline algorithm solves
for the minimum-budget injection that drops detection below threshold"""

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
_MIN_TRIM = 0.0
_MAX_TRIM = 1.0


_PLACEMENT_GRID_SIZE = 33


def inject_targeted(
    benign_trace: ObservationBatch,
    target_attack_behavior: Observation,
    budget_fraction: float,
    *,
    defender_trim_fraction: float = 0.0,
    rng: np.random.Generator | None = None,
) -> PoisonedTrace:
    """White-box poisoning against an `IndependentGaussianBaseline` with
    known `defender_trim_fraction`. Picks the per-feature injection value
    that minimizes the defender's score on `target_attack_behavior` after
    a fresh symmetric-trim fit on the poisoned window"""
    _validate_inputs(
        benign_trace=benign_trace,
        target_attack_behavior=target_attack_behavior,
        budget_fraction=budget_fraction,
        defender_trim_fraction=defender_trim_fraction,
    )

    generator = rng if rng is not None else np.random.default_rng()
    benign_count = benign_trace.shape[0]
    injection_count = round(benign_count * budget_fraction)

    if injection_count == 0:
        return PoisonedTrace(
            observations=benign_trace.copy(),
            is_adversarial=np.zeros(benign_count, dtype=np.bool_),
        )

    injection_vector = _optimal_injection_vector(
        benign_trace=benign_trace,
        target=target_attack_behavior,
        injection_count=injection_count,
        defender_trim_fraction=defender_trim_fraction,
    )

    injections = np.tile(injection_vector, (injection_count, 1))
    combined = np.vstack([benign_trace, injections])
    labels = np.concatenate(
        [
            np.zeros(benign_count, dtype=np.bool_),
            np.ones(injection_count, dtype=np.bool_),
        ]
    )
    permutation = generator.permutation(combined.shape[0])

    return PoisonedTrace(
        observations=combined[permutation],
        is_adversarial=labels[permutation],
    )


def _optimal_injection_vector(
    *,
    benign_trace: ObservationBatch,
    target: Observation,
    injection_count: int,
    defender_trim_fraction: float,
) -> Observation:
    """Per-feature injection value that minimizes the defender's score on
    `target` after fitting on the poisoned window. Evaluated by grid
    search over candidate placements between the benign median and target"""
    if defender_trim_fraction <= 0.0:
        # Untrimmed defender, the unconstrained optimum is the target
        # itself. Every injection moves the empirical mean toward target
        # by 1/N and inflates std without limit
        return target.astype(np.float64, copy=True)

    feature_dim = benign_trace.shape[1]
    out = np.empty(feature_dim, dtype=np.float64)
    for feature_idx in range(feature_dim):
        out[feature_idx] = _optimal_injection_value_per_feature(
            benign_feature=benign_trace[:, feature_idx],
            target_value=float(target[feature_idx]),
            injection_count=injection_count,
            defender_trim_fraction=defender_trim_fraction,
        )
    return out


def _optimal_injection_value_per_feature(
    *,
    benign_feature: np.ndarray,
    target_value: float,
    injection_count: int,
    defender_trim_fraction: float,
) -> float:
    """Grid-search the per-feature placement value that minimizes the
    squared standardized residual at `target_value` after a fresh trim
    fit. Scope is one feature because Sediment is per-feature independent"""
    benign_median = float(np.median(benign_feature))
    grid = _placement_grid(target_value=target_value, benign_median=benign_median)

    best_residual_sq = np.inf
    best_value = target_value
    for candidate in grid:
        residual_sq = _trimmed_residual_sq(
            benign_feature=benign_feature,
            injection_value=float(candidate),
            injection_count=injection_count,
            defender_trim_fraction=defender_trim_fraction,
            target_value=target_value,
        )
        if residual_sq < best_residual_sq:
            best_residual_sq = residual_sq
            best_value = float(candidate)
    return best_value


def _placement_grid(*, target_value: float, benign_median: float) -> np.ndarray:
    """Candidate placement values dense between median and target, plus
    the target itself as the extreme-trimmed case"""
    if target_value == benign_median:
        return np.array([target_value], dtype=np.float64)
    interior = np.linspace(benign_median, target_value, num=_PLACEMENT_GRID_SIZE)
    return np.unique(np.concatenate([interior, np.array([target_value])]))


def _trimmed_residual_sq(
    *,
    benign_feature: np.ndarray,
    injection_value: float,
    injection_count: int,
    defender_trim_fraction: float,
    target_value: float,
) -> float:
    """Squared standardized residual at `target_value` after a symmetric
    trim fit on the combined window. Mirrors `Sediment._fit_moments` so
    the adversary's optimization tracks the defender's actual computation"""
    injections = np.full(injection_count, injection_value, dtype=np.float64)
    combined = np.concatenate([benign_feature, injections])
    n_samples = combined.size
    per_tail = int(defender_trim_fraction / 2.0 * n_samples + 0.5)
    survivor_count = n_samples - 2 * per_tail
    if survivor_count < 1:
        return float("inf")
    sorted_combined = np.sort(combined)
    trimmed = sorted_combined[per_tail : n_samples - per_tail]
    mean = float(trimmed.mean())
    ddof = 1 if survivor_count > 1 else 0
    std = float(trimmed.std(ddof=ddof))
    # Adding a small epsilon mirrors the defender's epsilon floor and
    # keeps the optimization finite when std collapses to zero
    denom = std + 1e-12
    return ((target_value - mean) / denom) ** 2


def _validate_inputs(
    *,
    benign_trace: ObservationBatch,
    target_attack_behavior: Observation,
    budget_fraction: float,
    defender_trim_fraction: float,
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
    if not (_MIN_TRIM <= defender_trim_fraction < _MAX_TRIM):
        raise InvalidParameterError(
            "defender_trim_fraction must be in "
            f"[{_MIN_TRIM}, {_MAX_TRIM}), got {defender_trim_fraction}"
        )
    if not np.all(np.isfinite(benign_trace)):
        raise InvalidObservationError("benign_trace contains non-finite values, NaN or inf")
    if not np.all(np.isfinite(target_attack_behavior)):
        raise InvalidObservationError(
            "target_attack_behavior contains non-finite values, NaN or inf"
        )
