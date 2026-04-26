"""Tests for vectorized `score_batch` against the row-by-row `score`. The
contract is `score_batch(X)[i] == score(X[i])`. This parity test guards
against regressions when either scoring path is optimized independently"""

from __future__ import annotations

import numpy as np
import pytest

from chronosynd_py.baseline.naive import NaiveBaseline
from chronosynd_py.baseline.sediment import Sediment
from chronosynd_py.core import (
    BaselineNotFittedError,
    DimensionMismatchError,
    InvalidObservationError,
)


@pytest.fixture(name="fitted_naive")
def fitted_naive_fixture() -> NaiveBaseline:
    rng = np.random.default_rng(seed=42)
    baseline = NaiveBaseline()
    baseline.fit(rng.normal(size=(200, 5)))
    return baseline


@pytest.fixture(name="fitted_sediment")
def fitted_sediment_fixture() -> Sediment:
    rng = np.random.default_rng(seed=42)
    baseline = Sediment(trim_fraction=0.2)
    baseline.fit(rng.normal(size=(200, 5)))
    return baseline


def _test_batch(seed: int, dim: int, count: int) -> np.ndarray:
    return np.random.default_rng(seed=seed).normal(size=(count, dim))


def test_naive_score_batch_matches_row_by_row(fitted_naive: NaiveBaseline) -> None:
    batch = _test_batch(seed=7, dim=5, count=50)
    vectorized = fitted_naive.score_batch(batch)
    row_by_row = np.array([fitted_naive.score(batch[i]) for i in range(batch.shape[0])])
    np.testing.assert_allclose(vectorized, row_by_row, rtol=1e-12, atol=1e-12)


def test_sediment_score_batch_matches_row_by_row(fitted_sediment: Sediment) -> None:
    batch = _test_batch(seed=7, dim=5, count=50)
    vectorized = fitted_sediment.score_batch(batch)
    row_by_row = np.array(
        [fitted_sediment.score(batch[i]) for i in range(batch.shape[0])]
    )
    np.testing.assert_allclose(vectorized, row_by_row, rtol=1e-12, atol=1e-12)


def test_score_batch_before_fit_raises() -> None:
    with pytest.raises(BaselineNotFittedError):
        NaiveBaseline().score_batch(np.zeros((10, 3)))


def test_score_batch_rejects_1d_input(fitted_naive: NaiveBaseline) -> None:
    with pytest.raises(DimensionMismatchError):
        fitted_naive.score_batch(np.zeros(5))


def test_score_batch_rejects_wrong_feature_dim(fitted_naive: NaiveBaseline) -> None:
    with pytest.raises(DimensionMismatchError):
        fitted_naive.score_batch(np.zeros((10, 3)))


def test_score_batch_rejects_non_finite_values(fitted_naive: NaiveBaseline) -> None:
    bad = np.zeros((3, 5))
    bad[1, 2] = np.nan
    with pytest.raises(InvalidObservationError):
        fitted_naive.score_batch(bad)


def test_score_batch_returns_float64_array(fitted_naive: NaiveBaseline) -> None:
    batch = _test_batch(seed=1, dim=5, count=8)
    scores = fitted_naive.score_batch(batch)
    assert scores.dtype == np.float64
    assert scores.shape == (8,)


def test_empty_batch_returns_empty_array(fitted_naive: NaiveBaseline) -> None:
    empty = np.zeros((0, 5))
    scores = fitted_naive.score_batch(empty)
    assert scores.shape == (0,)
