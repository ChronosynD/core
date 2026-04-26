"""Sediment, the poisoning-resistant baseline estimator with a symmetric
trimmed mean and bias-corrected trimmed std per feature. Set
`trim_fraction >= 2β` to resist a budget-`β` adversary"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm

from chronosynd_py.baseline._shared import IndependentGaussianBaseline
from chronosynd_py.core import (
    EmptyLearningWindowError,
    InvalidParameterError,
    Observation,
    ObservationBatch,
)

_MIN_TRIM = 0.0
_MAX_TRIM = 1.0


class Sediment(IndependentGaussianBaseline):
    """Trimmed-mean estimator that is robust under learning-window poisoning"""

    def __init__(self, *, trim_fraction: float = 0.1, epsilon: float = 1e-6) -> None:
        """`trim_fraction` is the total fraction dropped per feature, split
        evenly between low and high tails. `epsilon` keeps scoring finite
        against zero-variance features"""
        super().__init__(epsilon=epsilon)
        if not np.isfinite(trim_fraction) or not (_MIN_TRIM <= trim_fraction < _MAX_TRIM):
            raise InvalidParameterError(
                f"trim_fraction must be finite and in [{_MIN_TRIM}, {_MAX_TRIM}), "
                f"got {trim_fraction}"
            )
        self._trim_fraction = trim_fraction
        self._std_correction = _gaussian_trim_std_correction(trim_fraction)

    @property
    def trim_fraction(self) -> float:
        """Fraction of samples dropped from the tails of each feature"""
        return self._trim_fraction

    def _fit_moments(
        self, observations: ObservationBatch
    ) -> tuple[Observation, Observation]:
        n_samples = observations.shape[0]

        if self._trim_fraction == 0.0:
            ddof = 1 if n_samples > 1 else 0
            return observations.mean(axis=0), observations.std(axis=0, ddof=ddof)

        # round-half-up rather than banker's so this matches Rust's `.round()`
        per_tail = int(self._trim_fraction / 2.0 * n_samples + 0.5)
        survivor_count = n_samples - 2 * per_tail
        if survivor_count < 1:
            raise EmptyLearningWindowError(
                f"trim_fraction={self._trim_fraction} leaves 0 samples for "
                f"n_samples={n_samples}, collect more data or trim less"
            )

        sorted_per_feature = np.sort(observations, axis=0)
        trimmed = sorted_per_feature[per_tail : n_samples - per_tail]
        ddof = 1 if survivor_count > 1 else 0
        return (
            trimmed.mean(axis=0),
            trimmed.std(axis=0, ddof=ddof) * self._std_correction,
        )


def _gaussian_trim_std_correction(trim_fraction: float) -> float:
    """Multiplicative factor that unbiases a trimmed std against a Gaussian
    null. For `N(0, s)` symmetrically trimmed the sample variance under-
    estimates `s^2`, and this factor recovers an unbiased estimate of `s`"""
    if trim_fraction <= 0.0:
        return 1.0
    cutoff = float(norm.ppf(1.0 - trim_fraction / 2.0))
    phi_cutoff = float(norm.pdf(cutoff))
    phi_bulk = 1.0 - trim_fraction
    truncated_variance = 1.0 - 2.0 * cutoff * phi_cutoff / phi_bulk
    return 1.0 / float(np.sqrt(truncated_variance))
