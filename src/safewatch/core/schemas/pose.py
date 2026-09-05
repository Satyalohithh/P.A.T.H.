"""Pose-estimation data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from safewatch.core.constants import DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD, Keypoint
from safewatch.core.types import Confidence, Keypoints, TrackId
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


class PoseRow(TypedDict):
    """One row of the exported pose dataset (17 keypoints flattened)."""

    frame_index: int
    track_id: int
    timestamp: float
    keypoints: list[tuple[float, float, float]]


__all__ = ["PoseResult", "PoseRow"]
