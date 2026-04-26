"""Baseline-poisoning attacks, concrete instantiations of the adversary in
`docs/THREAT_MODEL.md`. Each produces a `PoisonedTrace` that mixes benign
and adversary rows with known ground-truth labels"""

from chronosynd_py.evaluation.attacks.burst import inject_burst
from chronosynd_py.evaluation.attacks.pre_seed import inject_pre_seed
from chronosynd_py.evaluation.attacks.targeted import inject_targeted

__all__ = ["inject_burst", "inject_pre_seed", "inject_targeted"]
