"""Tracking data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypedDict

from safewatch.core.types import FrameIndex, TrackId


class TrackState(StrEnum):
    """Lifecycle state of a tracked object."""

    NEW = "new"
    ACTIVE = "active"
    LOST = "lost"
    TERMINATED = "terminated"


@dataclass(frozen=True, slots=True)
class Track:
    """A person track with state and last-seen location."""

    track_id: TrackId
    state: TrackState
    first_frame: FrameIndex
    last_frame: FrameIndex
    # (xmin, ymin, xmax, ymax) of the most recent observation.
    last_bbox: tuple[float, float, float, float] | None = None

    @property
    def is_confirmed(self) -> bool:
        raise NotImplementedError("TODO(implementation): Track.is_confirmed")


@dataclass(frozen=True, slots=True)
class TrackAssignment:
    """Association of a detection to an existing track."""

    track_id: TrackId
    detection_id: int
    match_score: float


class TrackFrameView(TypedDict):
    """Serializable per-frame tracker state."""

    frame_index: int
    tracks: list[dict[str, object]]


__all__ = ["Track", "TrackAssignment", "TrackFrameView", "TrackState"]
