"""Unit tests for the synthetic workload factories"""

from __future__ import annotations

import numpy as np
import pytest

from chronosynd_py.core import DimensionMismatchError, InvalidParameterError
from chronosynd_py.evaluation.workloads import (
    gaussian_mixture,
    heterogeneous_gaussian,
    isotropic_gaussian,
    jittered_target,
)


class TestIsotropicGaussian:
    def test_produces_expected_shape(self) -> None:
        source = isotropic_gaussian(feature_dim=5)
        batch = source(np.random.default_rng(seed=1), 50)
        assert batch.shape == (50, 5)

    def test_respects_loc_and_scale(self) -> None:
        source = isotropic_gaussian(feature_dim=3, loc=10.0, scale=2.0)
        batch = source(np.random.default_rng(seed=42), 5000)
        assert batch.mean() == pytest.approx(10.0, abs=0.1)
        assert batch.std() == pytest.approx(2.0, abs=0.1)

    def test_reproducible_under_seeded_rng(self) -> None:
        source = isotropic_gaussian(feature_dim=4)
        first = source(np.random.default_rng(seed=7), 100)
        second = source(np.random.default_rng(seed=7), 100)
        np.testing.assert_array_equal(first, second)

    @pytest.mark.parametrize("bad_dim", [0, -1, -5])
    def test_rejects_invalid_feature_dim(self, bad_dim: int) -> None:
        with pytest.raises(InvalidParameterError):
            isotropic_gaussian(feature_dim=bad_dim)

    @pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("inf"), float("nan")])
    def test_rejects_invalid_scale(self, bad_scale: float) -> None:
        with pytest.raises(InvalidParameterError):
            isotropic_gaussian(feature_dim=3, scale=bad_scale)

    def test_rejects_non_finite_loc(self) -> None:
        with pytest.raises(InvalidParameterError):
            isotropic_gaussian(feature_dim=3, loc=float("nan"))


class TestJitteredTarget:
    def test_zero_jitter_produces_exact_copies(self) -> None:
        target = np.array([1.0, 2.0, 3.0])
        source = jittered_target(target, jitter_scale=0.0)
        batch = source(np.random.default_rng(seed=1), 10)
        for row in batch:
            np.testing.assert_array_equal(row, target)

    def test_positive_jitter_produces_near_target(self) -> None:
        target = np.array([5.0, 5.0, 5.0])
        source = jittered_target(target, jitter_scale=0.05)
        batch = source(np.random.default_rng(seed=1), 200)

        distances = np.linalg.norm(batch - target, axis=1)
        assert np.all(distances > 0.0)
        assert np.all(distances < 0.5)

    def test_rejects_2d_target(self) -> None:
        with pytest.raises(DimensionMismatchError):
            jittered_target(np.zeros((1, 3)))

    def test_rejects_non_finite_target(self) -> None:
        with pytest.raises(InvalidParameterError):
            jittered_target(np.array([1.0, np.nan, 3.0]))

    @pytest.mark.parametrize("bad_jitter", [-0.1, -1.0, float("inf"), float("nan")])
    def test_rejects_invalid_jitter_scale(self, bad_jitter: float) -> None:
        with pytest.raises(InvalidParameterError):
            jittered_target(np.zeros(3), jitter_scale=bad_jitter)


class TestHeterogeneousGaussian:
    def test_per_feature_mean_and_scale_are_respected(self) -> None:
        mean = np.array([0.0, 10.0, -5.0])
        scale = np.array([0.5, 2.0, 1.0])
        source = heterogeneous_gaussian(mean, scale)
        batch = source(np.random.default_rng(seed=42), 5000)

        np.testing.assert_allclose(batch.mean(axis=0), mean, atol=0.1)
        np.testing.assert_allclose(batch.std(axis=0), scale, atol=0.1)

    def test_reproducible_under_seeded_rng(self) -> None:
        source = heterogeneous_gaussian([0.0, 1.0], [1.0, 0.5])
        first = source(np.random.default_rng(seed=7), 100)
        second = source(np.random.default_rng(seed=7), 100)
        np.testing.assert_array_equal(first, second)

    def test_rejects_dimension_mismatch_between_mean_and_scale(self) -> None:
        with pytest.raises(DimensionMismatchError):
            heterogeneous_gaussian([0.0, 1.0, 2.0], [1.0, 1.0])

    @pytest.mark.parametrize("bad_scale", [[1.0, 0.0], [1.0, -0.5]])
    def test_rejects_non_positive_scale_entries(self, bad_scale: list[float]) -> None:
        with pytest.raises(InvalidParameterError):
            heterogeneous_gaussian([0.0, 0.0], bad_scale)


class TestGaussianMixture:
    def test_two_component_mixture_produces_bimodal_data(self) -> None:
        components = [
            (1.0, [-3.0, -3.0], [0.2, 0.2]),
            (1.0, [+3.0, +3.0], [0.2, 0.2]),
        ]
        source = gaussian_mixture(components)
        batch = source(np.random.default_rng(seed=42), 5000)

        # The mixture should split roughly 50/50 between the modes
        near_negative = np.sum((batch[:, 0] < -1.0) & (batch[:, 1] < -1.0))
        near_positive = np.sum((batch[:, 0] > 1.0) & (batch[:, 1] > 1.0))
        assert near_negative > 2000
        assert near_positive > 2000
        # The aggregate mean lands near the origin between the modes
        np.testing.assert_allclose(batch.mean(axis=0), [0.0, 0.0], atol=0.2)

    def test_weights_are_normalized(self) -> None:
        components = [
            (3.0, [0.0], [1.0]),
            (1.0, [10.0], [0.1]),
        ]
        source = gaussian_mixture(components)
        batch = source(np.random.default_rng(seed=42), 4000)
        # 3:1 weights -> ~3000 from mode 0 around 0, ~1000 from mode 1 around 10
        from_mode_zero = np.sum(np.abs(batch[:, 0]) < 5.0)
        assert 2700 < from_mode_zero < 3300

    def test_rejects_empty_components(self) -> None:
        with pytest.raises(InvalidParameterError):
            gaussian_mixture([])

    def test_rejects_non_positive_weight(self) -> None:
        with pytest.raises(InvalidParameterError):
            gaussian_mixture([(0.0, [0.0], [1.0])])

    def test_rejects_dimension_mismatch_across_components(self) -> None:
        with pytest.raises(DimensionMismatchError):
            gaussian_mixture(
                [
                    (1.0, [0.0, 0.0], [1.0, 1.0]),
                    (1.0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
                ]
            )

    def test_zero_sample_count_returns_empty_batch_with_correct_dim(self) -> None:
        source = gaussian_mixture([(1.0, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0])])
        batch = source(np.random.default_rng(seed=1), 0)
        assert batch.shape == (0, 3)
