"""Tracking data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypedDict

from safewatch.core.schemas.detection import BBox
from safewatch.core.types import Confidence, FrameIndex, TimePoint, TrackId


class TrackState(StrEnum):
    """Lifecycle state of a tracked object."""

    NEW = "new"
    ACTIVE = "active"
    LOST = "lost"
    TERMINATED = "terminated"


@dataclass(frozen=True, slots=True)
class Track:
    """A person track with stable identity, state, and latest observation."""

    track_id: TrackId
    state: TrackState
    first_frame: FrameIndex
    last_frame: FrameIndex
    bbox: BBox
    confidence: Confidence
    timestamp: TimePoint
    age: int
    hits: int

    @property
    def is_confirmed(self) -> bool:
        """A track is confirmed once it has been successfully matched."""

        return self.state in (TrackState.ACTIVE, TrackState.LOST)


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
