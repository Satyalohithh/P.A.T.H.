"""Detection-to-track association logic."""

from __future__ import annotations

from safewatch.core.schemas.detection import Detection, DetectionList
from safewatch.core.schemas.tracking import Track, TrackAssignment, TrackState
from safewatch.core.types import TrackId


class Association:
    """Computes bipartite assignment between detections and predicted boxes."""

    def __init__(self, iou_threshold: float = 0.3) -> None:
        self.iou_threshold = iou_threshold

    def match(
        self,
        detections: DetectionList,
        tracks: list[Track],
    ) -> tuple[list[TrackAssignment], DetectionList, list[Track]]:
        """Return (matches, unmatched detections, unmatched tracks)."""

        raise NotImplementedError("TODO(implementation): Association.match")

    @staticmethod
    def score(detection: Detection, track: Track) -> float:
        raise NotImplementedError("TODO(implementation): Association.score")


__all__ = [
    "Association",
    "Detection",
    "Track",
    "TrackAssignment",
    "TrackId",
    "TrackState",
]
