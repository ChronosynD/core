"""Naive baseline, one Gaussian per feature under an independence assumption.
This is the prior-work reference Sediment is compared against, and is known
to be highly vulnerable to learning-window poisoning"""

from __future__ import annotations

from chronosynd_py.baseline._shared import IndependentGaussianBaseline
from chronosynd_py.core import Observation, ObservationBatch


class NaiveBaseline(IndependentGaussianBaseline):
    """Mean and standard deviation baseline with features treated as
    independent. The drift score is the sum of squared standardized residuals
    and equals the Mahalanobis distance under diagonal covariance"""

    def _fit_moments(
        self, observations: ObservationBatch
    ) -> tuple[Observation, Observation]:
        ddof = 1 if observations.shape[0] > 1 else 0
        return observations.mean(axis=0), observations.std(axis=0, ddof=ddof)
