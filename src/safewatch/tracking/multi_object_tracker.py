"""Multi-object tracking stage (ByteTrack backend, declared)."""

from __future__ import annotations

from safewatch.core.schemas.detection import Detection, DetectionList
from safewatch.core.schemas.tracking import Track, TrackState
from safewatch.core.types import FrameIndex, TrackId


class MultiObjectTracker:
    """Associates detections across frames into stable person tracks."""

    def __init__(
        self,
        max_age: int = 30,
        min_hits: int = 3,
        iou_threshold: float = 0.3,
    ) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold

    def update(self, frame_index: FrameIndex, detections: DetectionList) -> list[Track]:
        """Update tracks with this frame's detections; return active tracks."""

        raise NotImplementedError("TODO(implementation): MultiObjectTracker.update")

    def reset(self) -> None:
        raise NotImplementedError("TODO(implementation): MultiObjectTracker.reset")

    @property
    def active_tracks(self) -> tuple[Track, ...]:
        raise NotImplementedError(
            "TODO(implementation): MultiObjectTracker.active_tracks"
        )

    @property
    def terminations(self) -> tuple[Track, ...]:
        """Tracks that finished in the last update."""

        raise NotImplementedError(
            "TODO(implementation): MultiObjectTracker.terminations"
        )


__all__ = [
    "Detection",
    "DetectionList",
    "MultiObjectTracker",
    "Track",
    "TrackId",
    "TrackState",
]
