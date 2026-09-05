"""Frame annotation helpers for detection and tracking visualization."""

from __future__ import annotations

import cv2
import numpy as np

from safewatch.core.schemas.detection import BBox, DetectionList
from safewatch.core.schemas.tracking import Track

_BOX_COLOR = (0, 255, 0)
_LABEL_BG = (0, 0, 0)
_TEXT_COLOR = (255, 255, 255)
_LINE_WIDTH = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_MARGIN = 4
_FONT_SCALE = 0.5
_TEXT_LINE_WIDTH = 1


class FrameAnnotator:
    """Draws detections or tracked persons onto a copy of the input frame."""

    def annotate(self, frame: np.ndarray, detections: DetectionList) -> np.ndarray:
        """Return ``frame`` with detection boxes and confidence labels drawn.

        The input frame is left unmodified.
        """

        annotated = frame.copy()
        for detection in detections:
            label = f"{detection.class_name} {detection.confidence:.2f}"
            _draw_box_and_label(annotated, detection.bbox, label)
        return annotated

    def annotate_tracks(self, frame: np.ndarray, tracks: list[Track]) -> np.ndarray:
        """Return ``frame`` with tracked boxes and stable ``Person #ID`` labels.

        The input frame is left unmodified.
        """

        annotated = frame.copy()
        for track in tracks:
            label = f"Person #{track.track_id}"
            _draw_box_and_label(annotated, track.bbox, label)
        return annotated


def _draw_box_and_label(annotated: np.ndarray, bbox: BBox, label: str) -> None:
    """Draw a bounding box with a filled text label above it."""

    x1 = round(bbox.xmin)
    y1 = round(bbox.ymin)
    x2 = round(bbox.xmax)
    y2 = round(bbox.ymax)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), _BOX_COLOR, _LINE_WIDTH)

    (text_w, text_h), baseline = cv2.getTextSize(
        label, _FONT, _FONT_SCALE, _TEXT_LINE_WIDTH
    )
    label_y1 = max(0, y1 - text_h - baseline - _TEXT_MARGIN)
    cv2.rectangle(
        annotated,
        (x1, label_y1),
        (x1 + text_w + 2 * _TEXT_MARGIN, label_y1 + text_h + baseline),
        _LABEL_BG,
        cv2.FILLED,
    )
    cv2.putText(
        annotated,
        label,
        (x1 + _TEXT_MARGIN, label_y1 + text_h),
        _FONT,
        _FONT_SCALE,
        _TEXT_COLOR,
        _TEXT_LINE_WIDTH,
        cv2.LINE_AA,
    )


__all__ = ["FrameAnnotator"]
