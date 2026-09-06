"""Shared intermediate cache for the feature engine.

The cache is the single source of per-track kinematics consumed by the A/B/C
group extractors: torso height ``H``, center of mass, CoM velocity, joint
angles, and the recent per-frame pose history. Values are computed once on
push and reused across extractors for the same frame.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

from safewatch.core.constants import DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD, Keypoint
from safewatch.core.schemas.pose import PoseRecord
from safewatch.core.types import FrameIndex, TimePoint, TrackId, Vec2D
from safewatch.features import geometry


@dataclass(frozen=True, slots=True)
class PoseStats:
    """Computed kinematics for one track in one frame."""

    frame_index: FrameIndex
    timestamp: TimePoint
    h: float
    com_x: float
    com_y: float
    valid: bool
    keypoints: tuple = ()

    @property
    def com(self) -> Vec2D | None:
        if not self.valid:
            return None
        return (self.com_x, self.com_y)

    @property
    def is_finite(self) -> bool:
        return self.valid and math.isfinite(self.h)


class PoseRow:
    """Per-track rolling window of :class:`PoseStats` plus joint-angle series."""

    __slots__ = ("angles", "stats")

    def __init__(self) -> None:
        self.stats: deque[PoseStats] = deque()
        self.angles: dict[str, deque[float]] = {}


class FeatureCache:
    """Holds per-track pose history and derived kinematics for one stream.

    ``window`` is the fixed per-track history depth (interaction window). Tracks
    that disappear for good should be passed to :meth:`drop_tracks` so their
    state is evicted.
    """

    def __init__(
        self,
        keypoint_confidence_threshold: float = DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD,
        window: int = 30,
    ) -> None:
        self._gate = keypoint_confidence_threshold
        self._window = max(1, window)
        self._rows: dict[TrackId, PoseRow] = {}
        self._com_history: dict[TrackId, deque[tuple[FrameIndex, Vec2D]]] = {}

    # -- ingestion ---------------------------------------------------------

    def push(self, record: PoseRecord, frame_index: FrameIndex = 0) -> None:
        row = self._rows.setdefault(record.track_id, PoseRow())
        stats = self._stats_for(record, frame_index)
        row.stats.append(stats)
        while len(row.stats) > self._window:
            row.stats.popleft()
        angles = self._angles_for(record)
        for side, value in angles.items():
            target = row.angles.setdefault(side, deque())
            target.append(value)
            if len(target) > self._window:
                target.popleft()
        if stats.is_finite:
            com = stats.com
            if com is not None:
                history = self._com_history.setdefault(record.track_id, deque())
                history.append((frame_index, com))
                if len(history) > self._window:
                    history.popleft()

    def drop_tracks(self, track_ids: set[TrackId]) -> None:
        for track_id in track_ids:
            self._rows.pop(track_id, None)
            self._com_history.pop(track_id, None)

    # -- queries -----------------------------------------------------------

    def window(self, track_id: TrackId) -> list[PoseStats]:
        """Most recent ``window`` frames for ``track_id``, oldest first."""

        row = self._rows.get(track_id)
        return list(row.stats) if row else []

    def latest(self, track_id: TrackId) -> PoseStats | None:
        row = self._rows.get(track_id)
        return row.stats[-1] if row and row.stats else None

    def h(self, track_id: TrackId) -> float | None:
        """Most recent torso height for ``track_id``."""

        stats = self.latest(track_id)
        if stats is None or not stats.is_finite:
            return None
        return stats.h

    def com(self, track_id: TrackId) -> Vec2D | None:
        stats = self.latest(track_id)
        return stats.com if stats else None

    def velocity(self, track_id: TrackId) -> Vec2D | None:
        """CoM velocity (px/s) from the two most recent finite frames."""

        history = self._com_history.get(track_id)
        if history is None or len(history) < 2:
            return None
        first = history[-2]
        second = history[-1]
        dt = second[0] - first[0]
        if dt <= 0.0:
            return None
        return (
            (second[1][0] - first[1][0]) / dt,
            (second[1][1] - first[1][1]) / dt,
        )

    def speed(self, track_id: TrackId) -> float | None:
        velocity = self.velocity(track_id)
        if velocity is None:
            return None
        return math.hypot(*velocity)

    def angle_series(self, track_id: TrackId, side: str) -> deque[float]:
        """Joint-angle series (rad) for ``side`` in (arm/leg) x (L/R).

        ``side`` is one of ``arm_l``, ``arm_r``, ``leg_l``, ``leg_r``.
        """

        row = self._rows.get(track_id)
        if row is None:
            return deque()
        return row.angles.get(side, deque())

    def frame_ids(self, track_id: TrackId) -> list[FrameIndex]:
        row = self._rows.get(track_id)
        return [stats.frame_index for stats in row.stats] if row else []

    def active(self) -> set[TrackId]:
        return set(self._rows)

    def reset(self) -> None:
        self._rows.clear()
        self._com_history.clear()

    # -- internals ---------------------------------------------------------

    def _stats_for(self, record: PoseRecord, frame_index: FrameIndex) -> PoseStats:
        h = geometry.torso_height(record.keypoints, self._gate)
        com = geometry.center_of_mass(record.keypoints, self._gate)
        if h is not None and com is not None:
            return PoseStats(
                frame_index=frame_index,
                timestamp=record.timestamp,
                h=h,
                com_x=com[0],
                com_y=com[1],
                valid=True,
                keypoints=record.keypoints,
            )
        return PoseStats(
            frame_index=frame_index,
            timestamp=record.timestamp,
            h=float("nan"),
            com_x=float("nan"),
            com_y=float("nan"),
            valid=False,
            keypoints=record.keypoints,
        )

    def _angles_for(self, record: PoseRecord) -> dict[str, float]:
        kp = record.keypoints
        gate = self._gate
        limb_keys = {
            "arm_l": (
                (Keypoint.LEFT_SHOULDER, Keypoint.LEFT_ELBOW, Keypoint.LEFT_WRIST),
                1,
            ),
            "arm_r": (
                (Keypoint.RIGHT_SHOULDER, Keypoint.RIGHT_ELBOW, Keypoint.RIGHT_WRIST),
                1,
            ),
            "leg_l": (
                (Keypoint.LEFT_HIP, Keypoint.LEFT_KNEE, Keypoint.LEFT_ANKLE),
                1,
            ),
            "leg_r": (
                (Keypoint.RIGHT_HIP, Keypoint.RIGHT_KNEE, Keypoint.RIGHT_ANKLE),
                1,
            ),
        }
        result: dict[str, float] = {}
        for name, (indexes, _) in limb_keys.items():
            pts: list[tuple[float, float]] = [
                p
                for p in (geometry.point(kp, int(index), gate) for index in indexes)
                if p is not None
            ]
            if len(pts) != len(indexes):
                result[name] = float("nan")
                continue
            proximal, joint, distal = pts
            result[name] = geometry.interior_angle(joint, proximal, distal)
        return result


__all__ = ["FeatureCache", "PoseRow", "PoseStats"]
