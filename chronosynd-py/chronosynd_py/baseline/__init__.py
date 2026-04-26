"""Baseline estimators on the research side. The production port lives in
the `chronosynd-baseline` Rust crate and CI cross-checks both sides on a
fixed set of test vectors"""

from chronosynd_py.baseline.base import Baseline

__all__ = ["Baseline"]
