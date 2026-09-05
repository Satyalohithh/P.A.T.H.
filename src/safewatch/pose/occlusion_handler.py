"""Occlusion handling: missing-keypoint strategy and interpolation."""

from __future__ import annotations

from safewatch.core.schemas.pose import PoseResult
from safewatch.core.types import Keypoints, TrackId


class OcclusionHandler:
    """Recovers partially occluded poses via temporal interpolation."""

    def __init__(self, max_gap_frames: int = 5) -> None:
        self.max_gap_frames = max_gap_frames

    def interpolate(self, history: list[PoseResult]) -> list[PoseResult]:
        """Fill missing keypoints across consecutive frames."""

        raise NotImplementedError("TODO(implementation): OcclusionHandler.interpolate")

    @staticmethod
    def mark_invalid(keypoints: Keypoints, confidence_gate: float) -> Keypoints:
        """Zero-out keypoints below the confidence gate."""

        raise NotImplementedError("TODO(implementation): OcclusionHandler.mark_invalid")


__all__ = ["Keypoints", "OcclusionHandler", "PoseResult", "TrackId"]
