"""Tests for the pure geometric helpers feeding feature extraction."""

from __future__ import annotations

import math

import numpy as np

from safewatch.features import geometry
from tests.unit._feature_helpers import person_keypoints


class TestPointAndMultipoint:
    def test_point_gates_on_confidence(self) -> None:
        kp = ((1.0, 2.0, 0.9), (3.0, 4.0, 0.1))
        assert geometry.point(kp, 0, gate=0.45) == (1.0, 2.0)
        assert geometry.point(kp, 1, gate=0.45) is None

    def test_point_rejects_non_finite_and_out_of_range(self) -> None:
        kp = ((float("nan"), 2.0, 0.9), (1.0, float("inf"), 0.9))
        assert geometry.point(kp, 0, gate=0.45) is None
        assert geometry.point(kp, 1, gate=0.45) is None
        assert geometry.point(kp, 5, gate=0.45) is None

    def test_multipoint_omits_missing(self) -> None:
        kp = ((1.0, 2.0, 0.9), (3.0, 4.0, 0.1), (5.0, 6.0, 0.9))
        assert geometry.multipoint(kp, (0, 1, 2), gate=0.45) == (
            (1.0, 2.0),
            (5.0, 6.0),
        )
        assert geometry.multipoint(kp, (1,), gate=0.45) == ()


class TestMidpointCentroid:
    def test_midpoint_averages(self) -> None:
        assert geometry.midpoint((0.0, 0.0), (10.0, 20.0)) == (5.0, 10.0)

    def test_midpoint_none_when_either_missing(self) -> None:
        assert geometry.midpoint(None, (1.0, 2.0)) is None
        assert geometry.midpoint((1.0, 2.0), None) is None

    def test_centroid_mean(self) -> None:
        assert geometry.centroid(((0.0, 0.0), (2.0, 4.0), (4.0, 8.0))) == (2.0, 4.0)

    def test_centroid_empty(self) -> None:
        assert geometry.centroid(()) is None


class TestTorsoHeight:
    def test_vertical_person_has_h120(self) -> None:
        assert geometry.torso_height(person_keypoints(400.0, 300.0), 0.45) == 120.0

    def test_missing_hip_returns_none(self) -> None:
        kp = person_keypoints(400.0, 300.0, conf=0.1)
        assert geometry.torso_height(kp, 0.45) is None


class TestCenterOfMass:
    def test_centroid_of_valid_keypoints(self) -> None:
        com = geometry.center_of_mass(person_keypoints(100.0, 200.0), 0.45)
        assert com is not None
        assert com[0] == 100.0
        assert com[1] > 200.0  # below the shoulders, above the ankles

    def test_falls_back_to_hip_midpoint(self) -> None:
        kp = person_keypoints(100.0, 200.0, conf=0.1)
        kp = kp[:5] + ((105.0, 320.0, 0.9), (95.0, 320.0, 0.9)) + kp[7:]
        com = geometry.center_of_mass(kp, 0.45)
        assert com == (100.0, 320.0)

    def test_returns_none_without_enough_points(self) -> None:
        kp = ((0.0, 0.0, 0.1),) * 17
        assert geometry.center_of_mass(kp, 0.45) is None


class TestVectorHelpers:
    def test_distance(self) -> None:
        assert geometry.distance((0.0, 0.0), (3.0, 4.0)) == 5.0

    def test_unit_vector_normalized(self) -> None:
        unit = geometry.unit_vector((0.0, 0.0), (3.0, 4.0))
        assert unit is not None
        assert abs(unit[0] - 0.6) < 1e-9
        assert abs(unit[1] - 0.8) < 1e-9

    def test_unit_vector_zero_length_is_none(self) -> None:
        assert geometry.unit_vector((1.0, 1.0), (1.0, 1.0)) is None

    def test_interior_angle_right_angle(self) -> None:
        angle = geometry.interior_angle((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        assert abs(angle - math.pi / 2) < 1e-9

    def test_interior_angle_degenerate_apex(self) -> None:
        assert geometry.interior_angle((0.0, 0.0), (0.0, 0.0), (1.0, 0.0)) == 0.0

    def test_spine_lean_upright_is_zero(self) -> None:
        assert geometry.spine_lean_radians((0.0, 0.0), (0.0, 120.0)) == 0.0

    def test_spine_lean_horizontal_is_half_pi(self) -> None:
        assert (
            abs(geometry.spine_lean_radians((0.0, 0.0), (120.0, 0.0)) - math.pi / 2)
            < 1e-9
        )


class TestDivisionAndStats:
    def test_safe_divide_floors_denominator(self) -> None:
        assert geometry.safe_divide(5.0, 0.0) == 5.0 / 1e-9
        assert geometry.safe_divide(5.0, 2.0) == 2.5

    def test_finite_mean_skips_nan(self) -> None:
        assert geometry.finite_mean([1.0, float("nan"), 3.0]) == 2.0
        assert math.isnan(geometry.finite_mean([]))
        assert math.isnan(geometry.finite_mean([float("nan")]))

    def test_central_difference_linear_ramp(self) -> None:
        series = [0.0, 1.0, 2.0, 3.0, 4.0]
        diff = geometry.central_difference(series, dt=1.0)
        for value in diff[1:-1]:
            assert abs(value - 1.0) < 1e-9
        assert math.isnan(diff[0]) and math.isnan(diff[-1])

    def test_central_difference_two_frames_forward(self) -> None:
        diff = geometry.central_difference([10.0, 13.0], dt=1.0)
        assert diff[0] == 3.0
        assert math.isnan(diff[1])

    def test_central_difference_below_two_all_nan(self) -> None:
        assert all(math.isnan(v) for v in geometry.central_difference([5.0], dt=1.0))


class TestGaussianSmoothing:
    def test_kernel_is_odd_symmetric_unit_sum(self) -> None:
        kernel = geometry.gaussian_kernel_1d(5, 2.0)
        assert kernel.shape == (5,)
        assert np.allclose(kernel, kernel[::-1])
        assert abs(float(kernel.sum()) - 1.0) < 1e-9

    def test_kernel_forces_odd_width(self) -> None:
        kernel = geometry.gaussian_kernel_1d(4, 2.0)
        assert kernel.shape[0] % 2 == 1

    def test_smooth_constant_series_stays_constant(self) -> None:
        values = [3.0] * 20
        smoothed = geometry.causal_gaussian_smooth(values, 2.0)
        assert len(smoothed) == len(values)
        for value in smoothed:
            assert abs(value - 3.0) < 1e-9

    def test_smooth_preserves_nan_gaps(self) -> None:
        values = [1.0, float("nan"), 3.0, 4.0]
        smoothed = geometry.causal_gaussian_smooth(values, 2.0)
        assert math.isnan(smoothed[1])
        assert abs(smoothed[0] - 1.0) < 1e-9

    def test_smooth_short_series_returns_current(self) -> None:
        values = [5.0]
        assert geometry.causal_gaussian_smooth(values, 2.0) == [5.0]
