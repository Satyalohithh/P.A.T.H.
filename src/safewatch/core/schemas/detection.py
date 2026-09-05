"""Detection-stage data contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TypeAlias, TypedDict

from safewatch.core.types import Confidence, FrameIndex, TimePoint, Vec2D


@dataclass(frozen=True, slots=True)
class BBox:
    """Axis-aligned bounding box in image-pixel coordinates."""

    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def center(self) -> Vec2D:
        return ((self.xmin + self.xmax) / 2.0, (self.ymin + self.ymax) / 2.0)

    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def iou(self, other: BBox) -> float:
        """Intersection-over-union with ``other``.

        Formula per the feature extraction spec; raise on degenerate boxes.
        """

        raise NotImplementedError("TODO(implementation): BBox.iou")


@dataclass(frozen=True, slots=True)
class Detection:
    """A single person detection produced by the detection stage."""

    id: int
    bbox: BBox
    confidence: Confidence
    class_id: int
    class_name: str
    timestamp: TimePoint = 0.0


class DetectionFrameView(TypedDict):
    """Serializable view of a detection result for one frame."""

    frame_index: FrameIndex
    timestamp: TimePoint
    detections: list[dict[str, float | int | str]]


DetectionList: TypeAlias = list[Detection]

EMPTY_DETECTIONS: DetectionList = field(default_factory=list)

__all__ = [
    "EMPTY_DETECTIONS",
    "BBox",
    "Detection",
    "DetectionFrameView",
    "DetectionList",
]
