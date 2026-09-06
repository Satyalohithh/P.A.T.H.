"""Behavioral feature engine (Phase 4 scoped subsystem).

Implements the first feature-extraction slice of the frozen catalog:

- A.3 ``pose.torso_lean``         spine lean vs vertical (rad)
- B.1 ``motion.{arm,leg}_angular_vel.{L,R}``  joint-angle temporal derivative
- C.1 ``inter.distance``          CoM distance scaled by mean torso height H
- C.2 ``inter.approach_rate``     negative derivative of C.1
- C.3 ``inter.mutual_approach``   product of both persons' approach speeds
- D.6 ``temporal.reciprocity``    Pearson r of the pair's per-frame speeds

Design notes:

- Streaming/causal: per-frame values use a backward finite difference (the
  causal equivalent of the frozen central-difference formula for B.1/C.2; both
  return the same slope for linear ramps).
- Missing keypoints (below the confidence gate) and insufficient temporal
  context produce NaN for the affected column; aggregation skips NaN.
- Temporal series (B.1, C.2, C.3) are causal-Gaussian smoothed before
  emission when ``smoothing_sigma`` > 0.
- The legacy full-group extractors (``PoseFeatureExtractor`` etc.) and the
  canonical ``FEATURE_NAMES`` column order are left untouched.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

from safewatch.core.constants import (
    DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD,
    EPSILON,
    SMOOTHING_KERNEL_WIDTH,
    SMOOTHING_SIGMA_FRAMES,
    Keypoint,
)
from safewatch.core.schemas.features import FeatureRecord
from safewatch.core.schemas.pose import PoseRecord
from safewatch.core.types import FrameIndex, TimePoint, TrackId, Vec2D
from safewatch.features import geometry
from safewatch.features.intermediate import FeatureCache

NaNPolicy = frozenset({"skip_feature", "zero_fill", "raise"})


class BehavioralFeatureError(ValueError):
    """Raised for invalid runtime configuration of the feature engine."""


def _check_nan_policy(nan_policy: str) -> str:
    if nan_policy not in NaNPolicy:
        raise BehavioralFeatureError(
            f"nan_policy must be one of {sorted(NaNPolicy)}, got {nan_policy!r}"
        )
    return nan_policy


EGO_FEATURE_NAMES: tuple[str, ...] = (
    "pose.torso_lean",
    "motion.arm_angular_vel.L",
    "motion.arm_angular_vel.R",
    "motion.leg_angular_vel.L",
    "motion.leg_angular_vel.R",
)
"""Per-track per-frame features emitted by the engine."""

PAIR_FEATURE_NAMES: tuple[str, ...] = (
    "inter.distance",
    "inter.approach_rate",
    "inter.mutual_approach",
    "temporal.reciprocity",
)
"""Per-pair per-frame features emitted on both member tracks."""

TEMPORAL_FEATURES = frozenset(
    (
        "motion.arm_angular_vel.L",
        "motion.arm_angular_vel.R",
        "motion.leg_angular_vel.L",
        "motion.leg_angular_vel.R",
        "inter.approach_rate",
        "inter.mutual_approach",
    )
)
"""Feature names whose per-frame series are causal-Gaussian smoothed."""

PairKey = tuple[TrackId, TrackId]


def _pair_key(first: TrackId, second: TrackId) -> PairKey:
    return (first, second) if first < second else (second, first)


def _pearson(first: list[float], second: list[float]) -> float:
    """Pearson correlation of two equi-length series (NaN entries skipped)."""

    pairs = [
        (a, b) for a, b in zip(first, second) if math.isfinite(a) and math.isfinite(b)
    ]
    if len(pairs) < 3:
        return float("nan")
    n = len(pairs)
    mean_a = sum(a for a, _ in pairs) / n
    mean_b = sum(b for _, b in pairs) / n
    cov = sum((a - mean_a) * (b - mean_b) for a, b in pairs)
    var_a = sum((a - mean_a) ** 2 for a, _ in pairs)
    var_b = sum((b - mean_b) ** 2 for _, b in pairs)
    if var_a <= EPSILON and var_b <= EPSILON:
        return 1.0  # both constant -> perfectly synchronized
    if var_a <= EPSILON or var_b <= EPSILON:
        return 0.0  # one constant series -> undefined, use neutral 0
    return max(-1.0, min(1.0, cov / math.sqrt(var_a * var_b)))


@dataclass(frozen=True, slots=True)
class PoseView:
    """Snapshot of one track's kinematics used for a pair computation."""

    track_id: TrackId
    frame_index: FrameIndex
    com: Vec2D | None
    velocity: Vec2D | None
    h: float | None


class FeatureEngine:
    """Streaming per-frame behavioral feature extractor.

    Feed it the :class:`PoseRecord` list of one frame (pose estimation stage
    output, already associated to tracks); it returns the :class:`FeatureRecord`
    list for that frame, one ego record per track and one pair record per
    interacting pair (attached to both members).
    """

    def __init__(
        self,
        *,
        keypoint_confidence_threshold: float = DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD,
        window_frames: int = 30,
        smoothing_sigma: float = SMOOTHING_SIGMA_FRAMES,
        smoothing_kernel_width: int = SMOOTHING_KERNEL_WIDTH,
        fps: float = 30.0,
        nan_policy: str = "skip_feature",
    ) -> None:
        _check_nan_policy(nan_policy)
        self._gate = keypoint_confidence_threshold
        self._window = max(1, window_frames)
        self._sigma = smoothing_sigma
        self._kernel_width = max(1, smoothing_kernel_width)
        self._dt = 1.0 / fps if fps > 0.0 else 1.0 / 30.0
        self._nan_policy: str = nan_policy
        self._cache = FeatureCache(
            keypoint_confidence_threshold=keypoint_confidence_threshold,
            window=self._window,
        )
        self._pair_distances: dict[PairKey, deque[tuple[FrameIndex, float]]] = {}
        self._pair_speeds: dict[PairKey, deque[tuple[FrameIndex, float, float]]] = {}
        self._smooth_buffers: dict[tuple[object, ...], list[float]] = {}

    # -- public API --------------------------------------------------------

    def process(
        self,
        records: Iterable[PoseRecord],
        *,
        frame_index: FrameIndex,
        timestamp: TimePoint | None = None,
    ) -> list[FeatureRecord]:
        """Compute feature records for ``records`` (pose records of one frame)."""

        records = list(records)
        if records:
            timestamp = timestamp if timestamp is not None else records[0].timestamp
        else:
            timestamp = timestamp if timestamp is not None else 0.0

        for record in records:
            self._cache.push(record, frame_index)

        emitted: list[FeatureRecord] = []
        for track_id in sorted({record.track_id for record in records}):
            ego_values = self._finalize_values(self._ego_values(track_id))
            emitted.append(
                FeatureRecord(
                    track_id=track_id,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    names=EGO_FEATURE_NAMES,
                    values=ego_values,
                )
            )

        tracks = sorted({record.track_id for record in records})
        for index, first in enumerate(tracks):
            for second in tracks[index + 1 :]:
                pair_values = self._finalize_values(
                    self._pair_values(first, second, frame_index)
                )
                for member in (first, second):
                    other = second if member == first else first
                    emitted.append(
                        FeatureRecord(
                            track_id=member,
                            frame_index=frame_index,
                            timestamp=timestamp,
                            names=PAIR_FEATURE_NAMES,
                            values=pair_values,
                            pair_track_id=other,
                        )
                    )
        return emitted

    def drop_tracks(self, active_track_ids: set[TrackId]) -> None:
        """Evict state for tracks no longer present in ``active_track_ids``."""

        self._cache.drop_tracks(self._cache.active() - active_track_ids)
        stale_pairs = [
            key for key in self._pair_distances if not {*key} <= active_track_ids
        ]
        for pair_key in stale_pairs:
            self._pair_distances.pop(pair_key, None)
            self._pair_speeds.pop(pair_key, None)
        stale_buffers = [
            key for key in self._smooth_buffers if not {*key[1:]} <= active_track_ids
        ]
        for buffer_key in stale_buffers:
            self._smooth_buffers.pop(buffer_key, None)

    def reset(self) -> None:
        self._cache.reset()
        self._pair_distances.clear()
        self._pair_speeds.clear()
        self._smooth_buffers.clear()

    @property
    def cache(self) -> FeatureCache:
        return self._cache

    @property
    def fps(self) -> float:
        return 1.0 / self._dt

    # -- single-person features (A.3, B.1) ----------------------------------

    def _ego_values(self, track_id: TrackId) -> tuple[float, ...]:
        stats = self._cache.latest(track_id)
        if stats is None:
            return (float("nan"),) * len(EGO_FEATURE_NAMES)
        lean = self._torso_lean(track_id)
        arm_l = self._angular_velocity(track_id, "arm_l")
        arm_r = self._angular_velocity(track_id, "arm_r")
        leg_l = self._angular_velocity(track_id, "leg_l")
        leg_r = self._angular_velocity(track_id, "leg_r")
        raw = {
            "pose.torso_lean": lean,
            "motion.arm_angular_vel.L": arm_l,
            "motion.arm_angular_vel.R": arm_r,
            "motion.leg_angular_vel.L": leg_l,
            "motion.leg_angular_vel.R": leg_r,
        }
        return tuple(
            self._smoothed((track_id,), name, raw[name]) for name in EGO_FEATURE_NAMES
        )

    def _torso_lean(self, track_id: TrackId) -> float:
        stats = self._cache.latest(track_id)
        if stats is None or not stats.is_finite:
            return float("nan")
        mid_shoulder = geometry.midpoint(
            geometry.point(stats.keypoints, int(Keypoint.LEFT_SHOULDER), self._gate),
            geometry.point(stats.keypoints, int(Keypoint.RIGHT_SHOULDER), self._gate),
        )
        mid_hip = geometry.midpoint(
            geometry.point(stats.keypoints, int(Keypoint.LEFT_HIP), self._gate),
            geometry.point(stats.keypoints, int(Keypoint.RIGHT_HIP), self._gate),
        )
        if mid_shoulder is None or mid_hip is None:
            return float("nan")
        return geometry.spine_lean_radians(mid_shoulder, mid_hip)

    def _angular_velocity(self, track_id: TrackId, side: str) -> float:
        series = list(self._cache.angle_series(track_id, side))
        if len(series) < 2:
            return float("nan")
        first, second = series[-2], series[-1]
        if not (math.isfinite(first) and math.isfinite(second)):
            return float("nan")
        return (second - first) / self._dt

    # -- pair features (C.1, C.2, C.3, D.6) ----------------------------------

    def _pair_values(
        self,
        first: TrackId,
        second: TrackId,
        frame_index: FrameIndex,
    ) -> tuple[float, ...]:
        key = _pair_key(first, second)
        left = self._view(first)
        right = self._view(second)
        distance = self._distance_h(left, right)
        approach = self._approach_rate(key, frame_index, distance)
        mutual = self._mutual_approach(left, right, distance)
        reciprocity = self._reciprocity(key, frame_index, left, right)
        self._trim_pair(key)

        raw = {
            "inter.distance": distance,
            "inter.approach_rate": approach,
            "inter.mutual_approach": mutual,
            "temporal.reciprocity": reciprocity,
        }
        return tuple(
            self._smoothed(key, name, raw[name]) for name in PAIR_FEATURE_NAMES
        )

    def _view(self, track_id: TrackId) -> PoseView:
        stats = self._cache.latest(track_id)
        if stats is None:
            return PoseView(track_id, 0, None, None, None)
        return PoseView(
            track_id=track_id,
            frame_index=stats.frame_index,
            com=stats.com,
            velocity=self._cache.velocity(track_id),
            h=self._cache.h(track_id),
        )

    def _distance_h(
        self,
        left: PoseView,
        right: PoseView,
    ) -> float:
        if left.com is None or right.com is None or left.h is None or right.h is None:
            return float("nan")
        hip_height_mean = (left.h + right.h) / 2.0
        return geometry.distance(left.com, right.com) / hip_height_mean

    def _approach_rate(
        self,
        key: PairKey,
        frame_index: FrameIndex,
        distance: float,
    ) -> float:
        series = self._pair_distances.setdefault(key, deque())
        if math.isfinite(distance):
            series.append((frame_index, distance))
        if len(series) < 2:
            return float("nan")
        _, previous = series[-2]
        current_frame, current = series[-1]
        dt = max(EPSILON, (current_frame - series[-2][0]) * self._dt)
        return -(current - previous) / dt

    def _mutual_approach(
        self,
        left: PoseView,
        right: PoseView,
        distance: float,
    ) -> float:
        if (
            left.com is None
            or right.com is None
            or left.velocity is None
            or right.velocity is None
            or left.h is None
            or right.h is None
            or not math.isfinite(distance)
        ):
            return float("nan")
        toward = geometry.unit_vector(left.com, right.com)
        if toward is None:
            return float("nan")
        # convert CoM velocity from px/frame to px/s for consistent units
        fps = self.fps
        velocity_left = (left.velocity[0] * fps, left.velocity[1] * fps)
        velocity_right = (right.velocity[0] * fps, right.velocity[1] * fps)
        dot_left = velocity_left[0] * toward[0] + velocity_left[1] * toward[1]
        dot_right = velocity_right[0] * toward[0] + velocity_right[1] * toward[1]
        a_left = max(0.0, dot_left)
        a_right = max(0.0, -dot_right)
        mean_h = (left.h + right.h) / 2.0
        return (a_left * a_right) / (mean_h * mean_h + EPSILON)

    def _reciprocity(
        self,
        key: PairKey,
        frame_index: FrameIndex,
        left: PoseView,
        right: PoseView,
    ) -> float:
        series = self._pair_speeds.setdefault(key, deque())
        speed_left = self._speed_of(left)
        speed_right = self._speed_of(right)
        if speed_left is not None and speed_right is not None:
            series.append((frame_index, speed_left, speed_right))
        if len(series) < 3:
            return float("nan")
        first_speeds = [entry[1] for entry in series]
        second_speeds = [entry[2] for entry in series]
        return _pearson(first_speeds, second_speeds)

    def _speed_of(self, view: PoseView) -> float | None:
        if view.velocity is None:
            return None
        return math.hypot(view.velocity[0], view.velocity[1])

    def _trim_pair(self, key: PairKey) -> None:
        distances = self._pair_distances[key]
        while len(distances) > self._window:
            distances.popleft()
        speeds = self._pair_speeds[key]
        while len(speeds) > self._window:
            speeds.popleft()

    # -- smoothing ------------------------------------------------------------

    def _finalize_values(self, values: tuple[float, ...]) -> tuple[float, ...]:
        """Apply the configured NaN policy to an emitted value row."""

        if self._nan_policy == "skip_feature":
            return values
        result: list[float] = []
        for value in values:
            if math.isfinite(value):
                result.append(value)
            elif self._nan_policy == "raise":
                raise BehavioralFeatureError(
                    "feature value is NaN and nan_policy='raise'"
                )
            else:  # zero_fill
                result.append(0.0)
        return tuple(result)

    def _smoothed(
        self,
        members: tuple[TrackId, ...],
        name: str,
        raw: float,
    ) -> float:
        if name not in TEMPORAL_FEATURES or self._sigma <= 0.0:
            return raw
        key = (name, *members)
        buffer = self._smooth_buffers.setdefault(key, [])
        buffer.append(raw)
        if len(buffer) > self._window:
            del buffer[: len(buffer) - self._window]
        series = geometry.causal_gaussian_smooth(buffer, self._sigma)
        return series[-1] if series else float("nan")


__all__ = [
    "EGO_FEATURE_NAMES",
    "PAIR_FEATURE_NAMES",
    "TEMPORAL_FEATURES",
    "BehavioralFeatureError",
    "FeatureEngine",
    "PoseView",
    "_pearson",
]
