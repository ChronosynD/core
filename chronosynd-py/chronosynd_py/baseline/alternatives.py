"""Alternative poisoning-resistant estimators considered alongside
Sediment. Kept as siblings so the paper can ablate Sediment against
consensus-only, anomaly-within-only, and naive"""

from __future__ import annotations

import numpy as np

from chronosynd_py.baseline._shared import IndependentGaussianBaseline
from chronosynd_py.baseline.base import Baseline
from chronosynd_py.core import (
    BATCH_RANK,
    OBSERVATION_RANK,
    BaselineNotFittedError,
    BoolArray,
    DimensionMismatchError,
    EmptyLearningWindowError,
    FloatArray,
    InvalidObservationError,
    InvalidParameterError,
    Observation,
    ObservationBatch,
)

_MIN_SUB_WINDOWS = 2
_MIN_DROP = 0.0
_MAX_DROP = 1.0


class ConsensusBaseline(Baseline):
    """Sub-window consensus estimator. Splits the learning window into
    `n_sub_windows` equal slices and only scores against features whose
    per-sub-window means agree within `disagreement_threshold` pooled stds"""

    def __init__(
        self,
        *,
        n_sub_windows: int = 4,
        disagreement_threshold: float = 3.0,
        epsilon: float = 1e-6,
    ) -> None:
        """`n_sub_windows` is the number of equal-sized splits. A feature is
        kept when its sub-window means span no more than
        `disagreement_threshold` pooled standard deviations"""
        if n_sub_windows < _MIN_SUB_WINDOWS:
            raise InvalidParameterError(
                f"n_sub_windows must be at least {_MIN_SUB_WINDOWS}, got {n_sub_windows}"
            )
        if not np.isfinite(disagreement_threshold) or disagreement_threshold <= 0.0:
            raise InvalidParameterError(
                "disagreement_threshold must be a positive finite float, "
                f"got {disagreement_threshold!r}"
            )
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise InvalidParameterError(
                f"epsilon must be a positive finite float, got {epsilon!r}"
            )
        self._n_sub_windows = n_sub_windows
        self._disagreement_threshold = disagreement_threshold
        self._epsilon = epsilon
        self._mean: Observation | None = None
        self._std: Observation | None = None
        self._consensus_mask: BoolArray | None = None

    @property
    def n_sub_windows(self) -> int:
        """Number of equal-sized sub-windows the learning window is split into"""
        return self._n_sub_windows

    @property
    def consensus_mask(self) -> BoolArray:
        """Boolean mask over features kept after the consensus filter"""
        if self._consensus_mask is None:
            raise BaselineNotFittedError("call fit before reading consensus_mask")
        return self._consensus_mask

    def fit(self, observations: ObservationBatch) -> None:
        if observations.ndim != BATCH_RANK:
            raise DimensionMismatchError(
                "expected a 2-D batch with shape (n_samples, n_features), "
                f"got ndim={observations.ndim} shape={observations.shape}"
            )
        n_samples = observations.shape[0]
        if n_samples < self._n_sub_windows:
            raise EmptyLearningWindowError(
                f"need at least {self._n_sub_windows} samples for "
                f"n_sub_windows={self._n_sub_windows}, got {n_samples}"
            )
        if not np.all(np.isfinite(observations)):
            raise InvalidObservationError(
                "observations contain non-finite values, NaN or inf"
            )

        slices = np.array_split(observations, self._n_sub_windows, axis=0)
        sub_means = np.stack([s.mean(axis=0) for s in slices], axis=0)
        sub_stds = np.stack(
            [s.std(axis=0, ddof=1 if s.shape[0] > 1 else 0) for s in slices],
            axis=0,
        )

        pooled_std = np.sqrt(np.mean(sub_stds ** 2, axis=0))
        per_feature_spread = sub_means.max(axis=0) - sub_means.min(axis=0)
        # a constant feature has pooled_std == 0, scaling its spread by epsilon
        # flags any disagreement at all, which is what we want
        normalized_spread = per_feature_spread / (pooled_std + self._epsilon)
        consensus_mask = normalized_spread <= self._disagreement_threshold

        self._mean = observations.mean(axis=0)
        self._std = observations.std(
            axis=0, ddof=1 if n_samples > 1 else 0
        )
        self._consensus_mask = consensus_mask

    def score(self, observation: Observation) -> float:
        mean, std, mask = self._require_fitted()
        if observation.ndim != OBSERVATION_RANK:
            raise DimensionMismatchError(
                "expected a 1-D observation with shape (n_features,), "
                f"got ndim={observation.ndim} shape={observation.shape}"
            )
        if observation.shape != mean.shape:
            raise DimensionMismatchError(
                f"observation shape {observation.shape} does not match "
                f"fitted feature dimension {mean.shape}"
            )
        if not np.all(np.isfinite(observation)):
            raise InvalidObservationError(
                "observation contains non-finite values, NaN or inf"
            )
        if not np.any(mask):
            return 0.0
        standardized = (observation[mask] - mean[mask]) / (std[mask] + self._epsilon)
        return float(np.sum(standardized ** 2))

    def score_batch(self, observations: ObservationBatch) -> FloatArray:
        mean, std, mask = self._require_fitted()
        if observations.ndim != BATCH_RANK:
            raise DimensionMismatchError(
                "expected a 2-D batch with shape (n_samples, n_features), "
                f"got ndim={observations.ndim} shape={observations.shape}"
            )
        if observations.shape[1] != mean.shape[0]:
            raise DimensionMismatchError(
                f"batch feature dim {observations.shape[1]} does not match "
                f"fitted feature dimension {mean.shape[0]}"
            )
        if not np.all(np.isfinite(observations)):
            raise InvalidObservationError(
                "observations contain non-finite values, NaN or inf"
            )
        if not np.any(mask):
            return np.zeros(observations.shape[0], dtype=np.float64)
        standardized = (observations[:, mask] - mean[mask]) / (std[mask] + self._epsilon)
        return np.asarray(np.sum(standardized ** 2, axis=1, dtype=np.float64), dtype=np.float64)

    def _require_fitted(self) -> tuple[Observation, Observation, BoolArray]:
        if self._mean is None or self._std is None or self._consensus_mask is None:
            raise BaselineNotFittedError("call fit before score")
        return self._mean, self._std, self._consensus_mask


class AnomalyWithinBaseline(IndependentGaussianBaseline):
    """Self-consistency filter. Fits a naive Gaussian on the full window,
    drops the top `drop_fraction` of within-window outliers, and refits over
    what survives. An alternative cut to Sediment's symmetric per-feature trim"""

    def __init__(self, *, drop_fraction: float = 0.1, epsilon: float = 1e-6) -> None:
        """`drop_fraction` is the fraction of highest-scoring observations
        excluded from the second-pass fit. Scoring runs against an initial
        naive fit of the full window"""
        super().__init__(epsilon=epsilon)
        if not np.isfinite(drop_fraction) or not (_MIN_DROP <= drop_fraction < _MAX_DROP):
            raise InvalidParameterError(
                f"drop_fraction must be finite and in [{_MIN_DROP}, {_MAX_DROP}), "
                f"got {drop_fraction}"
            )
        self._drop_fraction = drop_fraction

    @property
    def drop_fraction(self) -> float:
        """Fraction of highest-scoring observations dropped before the refit"""
        return self._drop_fraction

    def _fit_moments(
        self, observations: ObservationBatch
    ) -> tuple[Observation, Observation]:
        n_samples = observations.shape[0]

        if self._drop_fraction == 0.0:
            ddof = 1 if n_samples > 1 else 0
            return observations.mean(axis=0), observations.std(axis=0, ddof=ddof)

        initial_mean = observations.mean(axis=0)
        ddof_initial = 1 if n_samples > 1 else 0
        initial_std = observations.std(axis=0, ddof=ddof_initial)

        standardized = (observations - initial_mean) / (initial_std + self._epsilon)
        per_observation_score = np.sum(standardized ** 2, axis=1)

        drop_count = int(self._drop_fraction * n_samples + 0.5)
        survivor_count = n_samples - drop_count
        if survivor_count < 1:
            raise EmptyLearningWindowError(
                f"drop_fraction={self._drop_fraction} leaves 0 samples for "
                f"n_samples={n_samples}, collect more data or drop less"
            )

        sorted_indices = np.argsort(per_observation_score, kind="stable")
        survivor_indices = sorted_indices[:survivor_count]
        survivors = observations[survivor_indices]

        ddof_final = 1 if survivor_count > 1 else 0
        return survivors.mean(axis=0), survivors.std(axis=0, ddof=ddof_final)
