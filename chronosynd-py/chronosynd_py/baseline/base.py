"""Baseline estimator interface"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from chronosynd_py.core import (
    BATCH_RANK,
    DimensionMismatchError,
    FloatArray,
    Observation,
    ObservationBatch,
)


class Baseline(ABC):
    """A fitted baseline that scores new observations for drift. Subclasses
    pick a construction strategy. `score` returns a non-negative value
    where higher means more anomalous"""

    @abstractmethod
    def fit(self, observations: ObservationBatch) -> None:
        """Fit the baseline on a learning-window batch with shape
        `(n_samples, n_features)` where `n_features` is the fixed
        dimensionality of the feature vector"""

    @abstractmethod
    def score(self, observation: Observation) -> float:
        """Score one observation against the fitted baseline. `observation`
        has shape `(n_features,)`. The return is a non-negative float
        where larger means more anomalous"""

    def score_batch(self, observations: ObservationBatch) -> FloatArray:
        """Score each row of a batch. The default loops over `score` and
        subclasses may override with a vectorized version. The contract is
        `score_batch(X)[i] == score(X[i])` for every row `i` of `X`"""
        if observations.ndim != BATCH_RANK:
            raise DimensionMismatchError(
                "expected a 2-D batch with shape (n_samples, n_features), "
                f"got ndim={observations.ndim} shape={observations.shape}"
            )
        return np.array(
            [self.score(observations[i]) for i in range(observations.shape[0])],
            dtype=np.float64,
        )
