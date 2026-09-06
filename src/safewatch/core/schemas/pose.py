"""Pose-estimation data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from safewatch.core.constants import DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD, Keypoint
from safewatch.core.schemas.detection import BBox
from safewatch.core.types import Confidence, Keypoints, TimePoint, TrackId
from safewatch.core.types import Keypoint as KeypointTuple


@dataclass(frozen=True, slots=True)
class PoseResult:
    """Keypoints for one tracked person in one frame."""

    track_id: TrackId
    keypoints: Keypoints
    confidences: tuple[Confidence, ...]

    @property
    def valid_keypoint_count(self) -> int:
        """Number of keypoints passing the default confidence gate."""

        return sum(
            1
            for conf in self.confidences
            if conf >= DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD
        )

    def keypoint(self, index: Keypoint) -> KeypointTuple | None:
        """Return keypoint ``index`` as (x, y, confidence) or None if invalid."""

        raise NotImplementedError("TODO(implementation): PoseResult.keypoint")


@dataclass(frozen=True, slots=True)
class PoseRecord:
    """Keypoints for one tracked person in one frame, matched to a track.

    ``keypoints`` is always aligned with the frozen :class:`Keypoint` order
    (COCO-17: 0 nose through 16 right ankle); each entry carries its own
    ``(x, y, confidence)``. ``confidence`` is the pose instance's own box
    confidence rather than a keypoint average.
    """

    track_id: TrackId
    timestamp: TimePoint
    keypoints: Keypoints
    confidence: Confidence

    @property
    def confidences(self) -> tuple[Confidence, ...]:
        """Per-keypoint confidence, mirroring the ``(x, y, conf)`` layout."""

        return tuple(keypoint[2] for keypoint in self.keypoints)

    def valid_keypoint_count(
        self,
        gate: Confidence = DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD,
    ) -> int:
        """Number of keypoints whose confidence meets ``gate``."""

        return sum(1 for conf in self.confidences if conf >= gate)

    def keypoint(
        self,
        index: Keypoint,
        min_confidence: Confidence = 0.0,
    ) -> KeypointTuple | None:
        """Return keypoint ``index`` as (x, y, confidence) or None if below
        ``min_confidence`` or out of range."""

        if not 0 <= int(index) < len(self.keypoints):
            return None
        keypoint = self.keypoints[int(index)]
        if keypoint[2] < min_confidence:
            return None
        return keypoint


class PoseRow(TypedDict):
    """One row of the exported pose dataset (17 keypoints flattened)."""

    frame_index: int
    track_id: int
    timestamp: float
    keypoints: list[tuple[float, float, float]]


__all__ = ["BBox", "PoseRecord", "PoseResult", "PoseRow"]
