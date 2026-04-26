"""Patience attack. The adversary smears injections thinly across the
learning window so a change-point detector watching the stream does not
see a burst of anomalous behavior concentrated in one region"""

from __future__ import annotations

import numpy as np

from chronosynd_py.core import Observation, ObservationBatch, PoisonedTrace


def inject_patience(
    benign_trace: ObservationBatch,
    target_attack_behavior: Observation,
    budget_fraction: float,
    *,
    smear_factor: float = 1.0,
    rng: np.random.Generator | None = None,
) -> PoisonedTrace:
    """Inject the poisoning budget spread thinly across the trace"""
    raise NotImplementedError
