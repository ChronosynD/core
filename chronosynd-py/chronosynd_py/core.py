"""Shared primitives, types and exceptions used across baseline estimators,
feature extractors, and the evaluation harness"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# A single observation, shape `(n_features,)`, one dense feature vector
Observation = npt.NDArray[np.float64]

# A batch of observations, shape `(n_samples, n_features)`, one row per sample
ObservationBatch = npt.NDArray[np.float64]

# Boolean mask, shape `(n_samples,)`, used to label adversarial rows in a trace
BoolArray = npt.NDArray[np.bool_]

# 1-D array of float scores, shape `(n_samples,)`
FloatArray = npt.NDArray[np.float64]

# Canonical ranks of the array shapes above, used in runtime validation
BATCH_RANK = 2
OBSERVATION_RANK = 1


class ChronosynDError(Exception):
    """Root of the ChronosynD error hierarchy so callers can catch-all cleanly"""


class BaselineNotFittedError(ChronosynDError, RuntimeError):
    """Raised when a baseline is scored before `fit` has been called"""


class DimensionMismatchError(ChronosynDError, ValueError):
    """Raised when an observation's shape does not match the fitted baseline"""


class EmptyLearningWindowError(ChronosynDError, ValueError):
    """Raised when `fit` is called with zero observations"""


class InvalidParameterError(ChronosynDError, ValueError):
    """Raised when an estimator is constructed with an out-of-range parameter"""


class InvalidObservationError(ChronosynDError, ValueError):
    """Raised when observation values are malformed (NaN or inf)"""


@dataclass(frozen=True, slots=True)
class PoisonedTrace:
    """A learning-window batch with ground-truth labels for adversarial
    rows. Evaluation-time poisoning attacks produce these so the harness
    can tell adversary rows apart and measure baseline behavior under attack"""

    observations: ObservationBatch
    is_adversarial: BoolArray

    def __post_init__(self) -> None:
        if self.observations.ndim != BATCH_RANK:
            raise DimensionMismatchError(
                "observations must be 2-D (n_samples, n_features), "
                f"got shape {self.observations.shape}"
            )
        if self.is_adversarial.ndim != OBSERVATION_RANK:
            raise DimensionMismatchError(
                f"is_adversarial must be 1-D, got shape {self.is_adversarial.shape}"
            )
        if self.is_adversarial.shape[0] != self.observations.shape[0]:
            raise DimensionMismatchError(
                f"label length {self.is_adversarial.shape[0]} "
                f"does not match observation count {self.observations.shape[0]}"
            )
        if not np.issubdtype(self.is_adversarial.dtype, np.bool_):
            raise DimensionMismatchError(
                f"is_adversarial must be a boolean array, got dtype {self.is_adversarial.dtype}"
            )

    @property
    def sample_count(self) -> int:
        """Total number of observations in the trace, benign plus adversarial"""
        return int(self.observations.shape[0])

    @property
    def adversarial_count(self) -> int:
        """Number of observations labeled as adversary-injected"""
        return int(np.count_nonzero(self.is_adversarial))
