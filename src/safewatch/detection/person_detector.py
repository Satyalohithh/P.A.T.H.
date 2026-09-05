"""Person detection stage (YOLOv8-pose backend, declared)."""

from __future__ import annotations

from safewatch.core import types as core_types
from safewatch.core.schemas.detection import Detection, DetectionList
from safewatch.core.types import Confidence
from safewatch.ingestion.decoder import Frame


class PersonDetector:
    """Detects persons and their keypoints in a single frame."""

    def __init__(
        self,
        weights_path: str,
        conf_threshold: Confidence = 0.25,
    ) -> None:
        self.weights_path = weights_path
        self.conf_threshold = conf_threshold

    def detect(self, frame: Frame) -> DetectionList:
        """Return all person detections in ``frame`` (no tracking yet)."""

        raise NotImplementedError("TODO(implementation): PersonDetector.detect")

    def warmup(self) -> None:
        raise NotImplementedError("TODO(implementation): PersonDetector.warmup")

    @property
    def latency_ms(self) -> float:
        """Rolling median inference latency for the detector."""

        raise NotImplementedError("TODO(implementation): PersonDetector.latency_ms")


__all__ = ["Confidence", "Detection", "DetectionList", "PersonDetector", "core_types"]
