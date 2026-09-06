"""Tests for the Phase 4 behavioral feature engine.

Verifies the frozen geometry/normalization for A.3, B.1, C.1-C.3, D.6, the
NaN gating rules, the pair bookkeeping, and the nan_policy application.
"""

from __future__ import annotations

import math
import statistics

import pytest

from safewatch.core.schemas.pose import PoseRecord
from safewatch.features import geometry
from safewatch.features.behavioral_features import (
    EGO_FEATURE_NAMES,
    PAIR_FEATURE_NAMES,
    BehavioralFeatureError,
    FeatureEngine,
    _pearson,
)
from tests.unit._feature_helpers import H, person_keypoints, person_pose


class TestEmissionShape:
    def test_record_names_and_pair_linkage(self) -> None:
        engine = FeatureEngine(smoothing_sigma=0.0)
        records = engine.process(
            [person_pose(1, 400.0, 300.0), person_pose(2, 800.0, 300.0)],
            frame_index=0,
        )
        ego = [r for r in records if r.pair_track_id is None]
        pairs = [r for r in records if r.pair_track_id is not None]
        assert len(ego) == 2
        assert len(pairs) == 2
        assert all(r.names == EGO_FEATURE_NAMES for r in ego)
        assert all(r.names == PAIR_FEATURE_NAMES for r in pairs)
        pair_1 = next(r for r in pairs if r.track_id == 1)
        pair_2 = next(r for r in pairs if r.track_id == 2)
        assert pair_1.pair_track_id == 2 and pair_2.pair_track_id == 1
        assert pair_1.values == pair_2.values

    def test_single_person_emits_ego_only(self) -> None:
        engine = FeatureEngine()
        records = engine.process([person_pose(1, 400.0, 300.0)], frame_index=0)
        assert [r.pair_track_id for r in records] == [None]

    def test_empty_input_emits_nothing(self) -> None:
        engine = FeatureEngine()
        assert engine.process([], frame_index=0) == []

    def test_record_timestamp_and_frame(self) -> None:
        engine = FeatureEngine()
        records = engine.process([person_pose(1, 400.0, 300.0)], frame_index=7)
        assert records[0].frame_index == 7
        assert records[0].timestamp == 0.0


class TestEgoFeatures:
    def test_upright_torso_lean_is_zero(self) -> None:
        engine = FeatureEngine(smoothing_sigma=0.0)
        records = engine.process([person_pose(1, 400.0, 300.0)], frame_index=0)
        values = dict(zip(records[0].names, records[0].values))
        assert abs(values["pose.torso_lean"] - 0.0) < 1e-9

    def test_leaning_person_lean_matches_geometry(self) -> None:
        x, y = 400.0, 300.0
        kp = person_keypoints(x, y)
        shifted = kp[:5] + ((x + 20.0, y, 0.9),) + kp[6:]
        pose = PoseRecord(track_id=1, timestamp=0.0, keypoints=shifted, confidence=0.9)
        records = FeatureEngine(smoothing_sigma=0.0).process([pose], frame_index=0)
        values = dict(zip(records[0].names, records[0].values))
        expected = geometry.spine_lean_radians(
            ((x + 20.0 + x + 5.0) / 2.0, y), (x, y + H)
        )
        assert abs(values["pose.torso_lean"] - expected) < 1e-9
        assert values["pose.torso_lean"] > 0.0

    def test_arm_angular_velocity_follows_wrist_sweep(self) -> None:
        fps = 30.0
        engine = FeatureEngine(smoothing_sigma=0.0, fps=fps)
        series_left: list[float] = []
        frame_records = []
        for frame in range(5):
            dx = 6.0 + frame * 2.0
            pose = person_pose(1, 400.0, 300.0, wrist_dx=(dx, 6.0))
            records = engine.process([pose], frame_index=frame)
            frame_records.append(records[0])
            kp = pose.keypoints
            series_left.append(geometry.interior_angle(kp[7][:2], kp[5][:2], kp[9][:2]))
        for frame in range(5):
            values = dict(zip(frame_records[frame].names, frame_records[frame].values))
            if frame == 0:
                assert math.isnan(values["motion.arm_angular_vel.L"])
            else:
                expected = (series_left[frame] - series_left[frame - 1]) * fps
                assert abs(values["motion.arm_angular_vel.L"] - expected) < 1e-6

    def test_missing_keypoints_nan_ego(self) -> None:
        engine = FeatureEngine(smoothing_sigma=0.0)
        records = engine.process(
            [person_pose(1, 400.0, 300.0, conf=0.1)], frame_index=0
        )
        values = dict(zip(records[0].names, records[0].values))
        assert all(math.isnan(value) for value in values.values())


class TestPairFeatures:
    def test_distance_is_com_distance_over_mean_h(self) -> None:
        engine = FeatureEngine(smoothing_sigma=0.0)
        records = engine.process(
            [person_pose(1, 400.0, 300.0), person_pose(2, 640.0, 300.0)],
            frame_index=0,
        )
        pair = next(r for r in records if r.pair_track_id is not None)
        values = dict(zip(pair.names, pair.values))
        assert abs(values["inter.distance"] - 240.0 / H) < 1e-9

    def test_approach_rate_constant_toward(self) -> None:
        engine = FeatureEngine(smoothing_sigma=0.0, fps=30.0)
        distances: list[float] = []
        for frame in range(4):
            x2 = 1600.0 - 60.0 * frame
            records = engine.process(
                [person_pose(1, 400.0, 300.0), person_pose(2, x2, 300.0)],
                frame_index=frame,
            )
            pair = next(r for r in records if r.pair_track_id is not None)
            values = dict(zip(pair.names, pair.values))
            distances.append(values["inter.distance"])
            if frame == 0:
                assert math.isnan(values["inter.approach_rate"])
            else:
                expected = -(distances[frame] - distances[frame - 1]) / (1.0 / 30.0)
                assert abs(values["inter.approach_rate"] - expected) < 1e-6
                assert values["inter.approach_rate"] > 0.0

    def test_approach_rate_sign_flips_when_parting(self) -> None:
        engine = FeatureEngine(smoothing_sigma=0.0, fps=30.0)
        for frame in range(4):
            x2 = 700.0 + 60.0 * frame
            records = engine.process(
                [person_pose(1, 400.0, 300.0), person_pose(2, x2, 300.0)],
                frame_index=frame,
            )
            pair = next(r for r in records if r.pair_track_id is not None)
            values = dict(zip(pair.names, pair.values))
            if frame == 0:
                assert math.isnan(values["inter.approach_rate"])
            else:
                assert values["inter.approach_rate"] < 0.0

    def test_mutual_approach_product(self) -> None:
        fps = 30.0
        engine = FeatureEngine(smoothing_sigma=0.0, fps=fps)
        speed_px_s = 900.0
        for frame in range(4):
            x1 = 400.0 + speed_px_s / fps * frame
            x2 = 1600.0 - speed_px_s / fps * frame
            records = engine.process(
                [person_pose(1, x1, 300.0), person_pose(2, x2, 300.0)],
                frame_index=frame,
            )
            pair = next(r for r in records if r.pair_track_id is not None)
            values = dict(zip(pair.names, pair.values))
            if frame == 0:
                assert math.isnan(values["inter.mutual_approach"])
            else:
                expected = (speed_px_s * speed_px_s) / (H * H)
                assert abs(values["inter.mutual_approach"] - expected) < 1e-6

    def test_mutual_approach_zero_when_standing(self) -> None:
        engine = FeatureEngine(smoothing_sigma=0.0, fps=30.0)
        values_ever = []
        for frame in range(3):
            records = engine.process(
                [person_pose(1, 400.0, 300.0), person_pose(2, 700.0, 300.0)],
                frame_index=frame,
            )
            pair = next(r for r in records if r.pair_track_id is not None)
            values = dict(zip(pair.names, pair.values))
            values_ever.append(values["inter.mutual_approach"])
        assert all(value == 0.0 for value in values_ever[1:])

    def test_reciprocity_needs_three_frames(self) -> None:
        engine = FeatureEngine(smoothing_sigma=0.0)
        got: list[float] = []
        for frame in range(2):
            records = engine.process(
                [person_pose(1, 400.0, 300.0), person_pose(2, 700.0 + frame, 300.0)],
                frame_index=frame,
            )
            pair = next(r for r in records if r.pair_track_id is not None)
            values = dict(zip(pair.names, pair.values))
            got.append(values["temporal.reciprocity"])
        assert all(math.isnan(value) for value in got)

    def test_reciprocity_correlated_speeds_is_one(self) -> None:
        fps = 30.0
        engine = FeatureEngine(smoothing_sigma=0.0, fps=fps)
        for frame in range(8):
            speed = 60.0 * (frame % 3 + 1)
            x1 = 400.0 + speed / fps * frame
            x2 = 1600.0 - speed / fps * frame
            records = engine.process(
                [person_pose(1, x1, 300.0), person_pose(2, x2, 300.0)],
                frame_index=frame,
            )
            pair = next(r for r in records if r.pair_track_id is not None)
            values = dict(zip(pair.names, pair.values))
            if frame >= 3:
                assert abs(values["temporal.reciprocity"] - 1.0) < 1e-6

    def test_reciprocity_anticorrelated_is_minus_one(self) -> None:
        engine = FeatureEngine(smoothing_sigma=0.0, fps=30.0)
        x1 = [400.0, 401.0, 403.0, 406.0, 410.0, 415.0]
        x2 = [1600.0, 1594.0, 1589.0, 1585.0, 1582.0, 1580.0]
        final: float | None = None
        for frame in range(len(x1)):
            records = engine.process(
                [person_pose(1, x1[frame], 300.0), person_pose(2, x2[frame], 300.0)],
                frame_index=frame,
            )
            pair = next(r for r in records if r.pair_track_id is not None)
            values = dict(zip(pair.names, pair.values))
            final = values["temporal.reciprocity"]
        assert final is not None
        assert abs(final + 1.0) < 1e-6

    def test_reciprocity_clipped_to_unit_range(self) -> None:
        speeds_a = [0.5 * i for i in range(6)]
        assert abs(_pearson(speeds_a, [2.0 * s for s in speeds_a]) - 1.0) < 1e-9
        assert _pearson([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]) == 1.0
        assert math.isnan(_pearson([1.0, 2.0], [3.0, 4.0]))
        assert abs(statistics.correlation(range(6), [5, 4, 3, 2, 1, 0]) + 1) < 1e-9


class TestRobustness:
    def test_geometrically_coincident_pair_no_crash(self) -> None:
        engine = FeatureEngine(smoothing_sigma=0.0)
        records = engine.process(
            [person_pose(1, 400.0, 300.0), person_pose(2, 400.0, 300.0)],
            frame_index=0,
        )
        pair = next(r for r in records if r.pair_track_id is not None)
        values = dict(zip(pair.names, pair.values))
        assert values["inter.distance"] == 0.0
        assert math.isnan(values["inter.mutual_approach"])

    def test_missing_keypoint_in_pair_nan_specific_columns(self) -> None:
        engine = FeatureEngine(smoothing_sigma=0.0)
        records = engine.process(
            [person_pose(1, 400.0, 300.0, conf=0.1), person_pose(2, 700.0, 300.0)],
            frame_index=0,
        )
        ego_1 = next(r for r in records if r.track_id == 1 and r.pair_track_id is None)
        ego_2 = next(r for r in records if r.track_id == 2 and r.pair_track_id is None)
        assert all(math.isnan(v) for v in ego_1.values)
        assert ego_2.values[0] == 0.0
        pair = next(r for r in records if r.pair_track_id is not None)
        values = dict(zip(pair.names, pair.values))
        assert all(math.isnan(v) for v in values.values())

    def test_nan_policy_zero_fill(self) -> None:
        engine = FeatureEngine(smoothing_sigma=0.0, nan_policy="zero_fill")
        records = engine.process([person_pose(1, 400.0, 300.0)], frame_index=0)
        assert records[0].values == (0.0,) * len(EGO_FEATURE_NAMES)

    def test_nan_policy_raise(self) -> None:
        engine = FeatureEngine(smoothing_sigma=0.0, nan_policy="raise")
        with pytest.raises(BehavioralFeatureError):
            engine.process([person_pose(1, 400.0, 300.0)], frame_index=0)

    def test_invalid_nan_policy_rejected(self) -> None:
        with pytest.raises(BehavioralFeatureError):
            FeatureEngine(nan_policy="drop")

    def test_smoothing_reduces_fluctuation(self) -> None:
        raw = FeatureEngine(smoothing_sigma=0.0, fps=30.0)
        smooth = FeatureEngine(smoothing_sigma=2.0, fps=30.0)
        raw_values: list[float] = []
        smooth_values: list[float] = []
        for frame in range(10):
            dx = 6.0 + frame * 2.0
            for engine, sink in ((raw, raw_values), (smooth, smooth_values)):
                records = engine.process(
                    [person_pose(1, 400.0, 300.0, wrist_dx=(dx, 6.0))],
                    frame_index=frame,
                )
                values = dict(zip(records[0].names, records[0].values))
                sink.append(values["motion.arm_angular_vel.L"])
        finite_raw = [v for v in raw_values if math.isfinite(v)]
        finite_smooth = [v for v in smooth_values if math.isfinite(v)]
        assert statistics.pstdev(finite_smooth) <= statistics.pstdev(finite_raw)

    def test_drop_tracks_and_reset(self) -> None:
        engine = FeatureEngine()
        engine.process(
            [person_pose(1, 400.0, 300.0), person_pose(2, 700.0, 300.0)],
            frame_index=0,
        )
        assert engine.cache.active() == {1, 2}
        engine.drop_tracks({1})
        assert engine.cache.active() == {1}
        engine.reset()
        assert engine.cache.active() == set()
