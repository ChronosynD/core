"""Shared plumbing for baselines that model each feature as an independent
Gaussian. NaiveBaseline and Sediment both score with squared standardized
residuals and only differ in how `_fit_moments` is computed"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np

from chronosynd_py.baseline.base import Baseline
from chronosynd_py.core import (
    BATCH_RANK,
    OBSERVATION_RANK,
    BaselineNotFittedError,
    DimensionMismatchError,
    EmptyLearningWindowError,
    FloatArray,
    InvalidObservationError,
    InvalidParameterError,
    Observation,
    ObservationBatch,
)


class IndependentGaussianBaseline(Baseline):
    """Baseline that assumes each feature is an independent Gaussian. The
    score is the Mahalanobis distance under a diagonal covariance.
    Subclasses override `_fit_moments` to supply per-feature moments"""

    def __init__(self, *, epsilon: float = 1e-6) -> None:
        """`epsilon` is added to each feature's standard deviation before
        normalization so a zero-variance feature does not blow up the score"""
        if not np.isfinite(epsilon) or epsilon <= 0.0:
            raise InvalidParameterError(
                f"epsilon must be a positive finite float, got {epsilon!r}"
            )
        self._epsilon = epsilon
        self._mean: Observation | None = None
        self._std: Observation | None = None

    def fit(self, observations: ObservationBatch) -> None:
        if observations.ndim != BATCH_RANK:
            raise DimensionMismatchError(
                "expected a 2-D batch with shape (n_samples, n_features), "
                f"got ndim={observations.ndim} shape={observations.shape}"
            )
        if observations.shape[0] == 0:
            raise EmptyLearningWindowError(
                "cannot fit on zero observations, need at least one sample"
            )
        if not np.all(np.isfinite(observations)):
            raise InvalidObservationError(
                "observations contain non-finite values, NaN or inf"
            )
        self._mean, self._std = self._fit_moments(observations)

    def score(self, observation: Observation) -> float:
        if self._mean is None or self._std is None:
            raise BaselineNotFittedError("call fit before score")
        if observation.ndim != OBSERVATION_RANK:
            raise DimensionMismatchError(
                "expected a 1-D observation with shape (n_features,), "
                f"got ndim={observation.ndim} shape={observation.shape}"
            )
        if observation.shape != self._mean.shape:
            raise DimensionMismatchError(
                f"observation shape {observation.shape} does not match "
                f"fitted feature dimension {self._mean.shape}"
            )
        if not np.all(np.isfinite(observation)):
            raise InvalidObservationError(
                "observation contains non-finite values, NaN or inf"
            )

        standardized = (observation - self._mean) / (self._std + self._epsilon)
        return float(np.sum(standardized ** 2))

    def score_batch(self, observations: ObservationBatch) -> FloatArray:
        if self._mean is None or self._std is None:
            raise BaselineNotFittedError("call fit before score_batch")
        if observations.ndim != BATCH_RANK:
            raise DimensionMismatchError(
                "expected a 2-D batch with shape (n_samples, n_features), "
                f"got ndim={observations.ndim} shape={observations.shape}"
            )
        if observations.shape[1] != self._mean.shape[0]:
            raise DimensionMismatchError(
                f"batch feature dim {observations.shape[1]} does not match "
                f"fitted feature dimension {self._mean.shape[0]}"
            )
        if not np.all(np.isfinite(observations)):
            raise InvalidObservationError(
                "observations contain non-finite values, NaN or inf"
            )
        standardized = (observations - self._mean) / (self._std + self._epsilon)
        return np.asarray(
            np.sum(standardized ** 2, axis=1, dtype=np.float64),
            dtype=np.float64,
        )

    @abstractmethod
    def _fit_moments(
        self, observations: ObservationBatch
    ) -> tuple[Observation, Observation]:
        """Compute per-feature mean and standard deviation for a validated
        batch. Subclasses pick the construction strategy. The shared
        validation in `fit` has already checked shape and finiteness"""
