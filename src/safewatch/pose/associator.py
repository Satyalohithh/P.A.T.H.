"""Associate raw pose instances with tracked persons by box overlap."""

from __future__ import annotations

from dataclasses import dataclass

from safewatch.core.schemas.detection import BBox
from safewatch.core.schemas.pose import PoseRecord
from safewatch.core.schemas.tracking import Track
from safewatch.core.types import Confidence, Keypoints, TimePoint
from safewatch.pose.keypoint_layout import validate_keypoints

DEFAULT_MATCH_THRESHOLD = 0.5
"""IoU gate for assigning a pose instance to a track box."""


@dataclass(frozen=True, slots=True)
class PoseCandidate:
    """A raw pose instance produced by the pose model before track matching."""

    bbox: BBox
    keypoints: Keypoints
    confidence: Confidence


class PoseAssociator:
    """Greedily matches :class:`PoseCandidate` instances to tracks.

    One pose instance is assigned to at most one track (one-to-one): each
    track is matched to the unused candidate with the highest IoU, falling back
    to no pose when nothing passes :attr:`match_threshold`.
    """

    def __init__(self, match_threshold: float = DEFAULT_MATCH_THRESHOLD) -> None:
        if not 0.0 <= match_threshold <= 1.0:
            raise ValueError("match_threshold must be in [0, 1]")
        self._match_threshold = match_threshold

    def associate(
        self,
        tracks: list[Track],
        candidates: list[PoseCandidate],
        *,
        timestamp: TimePoint,
    ) -> list[PoseRecord]:
        """Match ``candidates`` to ``tracks`` and build per-track records.

        Tracks with no qualifying candidate produce no record. Results keep the
        input track order.
        """

        if not tracks or not candidates:
            return []
        remaining = list(candidates)
        records: list[PoseRecord] = []
        for track in tracks:
            best_index, best_iou = -1, 0.0
            for index, candidate in enumerate(remaining):
                iou = _iou(track.bbox, candidate.bbox)
                if iou > best_iou:
                    best_iou, best_index = iou, index
            if best_index < 0 or best_iou < self._match_threshold:
                continue
            candidate = remaining.pop(best_index)
            validate_keypoints(candidate.keypoints)
            records.append(
                PoseRecord(
                    track_id=track.track_id,
                    timestamp=timestamp,
                    keypoints=candidate.keypoints,
                    confidence=candidate.confidence,
                )
            )
        return records


def _iou(first: BBox, second: BBox) -> float:
    """Intersection-over-union between two axis-aligned boxes."""

    inter_width = max(0.0, min(first.xmax, second.xmax) - max(first.xmin, second.xmin))
    inter_height = max(0.0, min(first.ymax, second.ymax) - max(first.ymin, second.ymin))
    intersection = inter_width * inter_height
    union = first.area() + second.area() - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


__all__ = ["DEFAULT_MATCH_THRESHOLD", "PoseAssociator", "PoseCandidate"]
