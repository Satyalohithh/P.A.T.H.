"""Person detection stage interface.

Concrete implementations (e.g. :class:`YOLOPersonDetector`) detect persons in
a single frame and return the shared :class:`Detection` list.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from safewatch.core.schemas.detection import DetectionList
from safewatch.core.types import Frame, TimePoint


class PersonDetector(ABC):
    """Interface for frame-level person detection."""

    @abstractmethod
    def detect(self, frame: Frame, *, timestamp: TimePoint = 0.0) -> DetectionList:
        """Return all person detections in ``frame`` (no tracking yet)."""

    @abstractmethod
    def warmup(self) -> None:
        """Load weights / allocate device buffers before steady-state use."""

    @property
    @abstractmethod
    def latency_ms(self) -> float:
        """Rolling median inference latency for the detector."""


__all__ = ["PersonDetector"]
