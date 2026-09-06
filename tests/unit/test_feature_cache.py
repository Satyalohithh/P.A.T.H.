"""Tests for the FeatureCache (pose history + derived kinematics)."""

from __future__ import annotations

import math

from safewatch.features.intermediate import FeatureCache
from tests.unit._feature_helpers import H, person_keypoints, person_pose


class TestIngestion:
    def test_push_builds_valid_stats(self) -> None:
        cache = FeatureCache()
        cache.push(person_pose(1, 400.0, 300.0), frame_index=0)
        stats = cache.latest(1)
        assert stats is not None
        assert stats.valid and stats.is_finite
        assert abs(stats.h - H) < 1e-9
        assert stats.com is not None
        assert stats.com[0] == 400.0

    def test_push_missing_keypoints_produces_invalid_stats(self) -> None:
        cache = FeatureCache()
        cache.push(person_pose(1, 400.0, 300.0, conf=0.1), frame_index=0)
        stats = cache.latest(1)
        assert stats is not None
        assert not stats.valid
        assert stats.com is None

    def test_window_returns_oldest_first(self) -> None:
        cache = FeatureCache(window=5)
        for frame in range(3):
            cache.push(person_pose(1, 400.0 + frame, 300.0), frame_index=frame)
        window = cache.window(1)
        assert [stats.frame_index for stats in window] == [0, 1, 2]

    def test_window_trims_to_capacity(self) -> None:
        cache = FeatureCache(window=3)
        for frame in range(6):
            cache.push(person_pose(1, 400.0, 300.0), frame_index=frame)
        assert [stats.frame_index for stats in cache.window(1)] == [3, 4, 5]

    def test_angle_series_lengths(self) -> None:
        cache = FeatureCache(window=4)
        for frame in range(4):
            cache.push(
                person_pose(1, 400.0, 300.0, wrist_dx=(6.0 + frame, 6.0)),
                frame_index=frame,
            )
        assert len(cache.angle_series(1, "arm_l")) == 4
        assert all(math.isfinite(v) for v in cache.angle_series(1, "arm_l"))
        assert len(cache.angle_series(2, "arm_r")) == 0

    def test_active_tracks(self) -> None:
        cache = FeatureCache()
        cache.push(person_pose(1, 0.0, 0.0), frame_index=0)
        cache.push(person_pose(2, 1.0, 0.0), frame_index=0)
        assert cache.active() == {1, 2}
        cache.drop_tracks({1})
        assert cache.active() == {2}


class TestKinematics:
    def test_velocity_between_two_frames(self) -> None:
        cache = FeatureCache()
        cache.push(person_pose(1, 400.0, 300.0), frame_index=0)
        cache.push(person_pose(1, 430.0, 300.0), frame_index=1)
        velocity = cache.velocity(1)
        assert velocity == (30.0, 0.0)

    def test_velocity_needs_two_frames(self) -> None:
        cache = FeatureCache()
        cache.push(person_pose(1, 400.0, 300.0), frame_index=0)
        assert cache.velocity(1) is None

    def test_height_and_com_queries(self) -> None:
        cache = FeatureCache()
        cache.push(person_pose(1, 400.0, 300.0), frame_index=0)
        assert cache.h(1) is not None and abs(cache.h(1) - H) < 1e-9
        assert cache.com(1) == (400.0, cache.latest(1).com[1])
        assert cache.h(99) is None
        assert cache.com(99) is None

    def test_speed_is_hypot_of_velocity(self) -> None:
        cache = FeatureCache()
        cache.push(person_pose(1, 400.0, 300.0), frame_index=0)
        cache.push(person_pose(1, 403.0, 304.0), frame_index=1)
        assert abs(cache.speed(1) - 5.0) < 1e-9


class TestLifecycle:
    def test_drop_and_reset(self) -> None:
        cache = FeatureCache()
        cache.push(person_pose(1, 0.0, 0.0), frame_index=0)
        cache.reset()
        assert cache.active() == set()
        assert cache.window(1) == []

    def test_frame_ids(self) -> None:
        cache = FeatureCache(window=2)
        for frame in (10, 11, 12):
            cache.push(person_pose(1, 0.0, 0.0), frame_index=frame)
        assert cache.frame_ids(1) == [11, 12]


def test_person_keypoints_are_coherent() -> None:
    kp = person_keypoints(400.0, 300.0)
    assert len(kp) == 17
    assert kp[0] == (400.0, 270.0, 0.9)
    assert all(keypoint[2] == 0.9 for keypoint in kp)
