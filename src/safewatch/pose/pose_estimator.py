"""Pose estimation stage (keypoint extraction and assembly)."""

from __future__ import annotations

from safewatch.core.schemas.detection import Detection
from safewatch.core.schemas.pose import PoseResult
from safewatch.core.schemas.tracking import Track
from safewatch.core.types import Confidence
from safewatch.ingestion.decoder import Frame


class PoseEstimator:
    """Estimates 17-keypoint COCO poses for detected persons."""

    def __init__(
        self,
        keypoint_confidence_threshold: Confidence = 0.45,
    ) -> None:
        self.keypoint_confidence_threshold = keypoint_confidence_threshold

    def estimate(self, frame: Frame, track: Track) -> PoseResult:
        """Estimate the pose of ``track``'s bounding box region in ``frame``."""

        raise NotImplementedError("TODO(implementation): PoseEstimator.estimate")

    def batch_estimate(self, frame: Frame, tracks: list[Track]) -> list[PoseResult]:
        raise NotImplementedError("TODO(implementation): PoseEstimator.batch_estimate")


__all__ = ["Detection", "PoseEstimator", "PoseResult", "Track"]
