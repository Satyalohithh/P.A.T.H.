"""Frame annotation helpers for detection, tracking, and pose visualization."""

from __future__ import annotations

import cv2
import numpy as np

from safewatch.core.constants import (
    DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD,
    Keypoint,
)
from safewatch.core.schemas.detection import BBox, DetectionList
from safewatch.core.schemas.pose import PoseRecord
from safewatch.core.schemas.tracking import Track
from safewatch.core.types import Keypoints
from safewatch.features import geometry
from safewatch.pose.keypoint_layout import COCO_17_KEYPOINT_NAMES, COCO_17_SKELETON

_BOX_COLOR = (0, 255, 0)
_LABEL_BG = (0, 0, 0)
_TEXT_COLOR = (255, 255, 255)
_LINE_WIDTH = 2
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_TEXT_MARGIN = 4
_FONT_SCALE = 0.5
_TEXT_LINE_WIDTH = 1

_SKELETON_COLOR = (0, 165, 255)
_KEYPOINT_RADIUS = 4
_KEYPOINT_FALLBACK_COLOR = (50, 50, 255)
_APPROACH_ARROW_MAX = 2000.0
_KEYPOINT_COLORS: dict[str, tuple[int, int, int]] = {
    "nose": (255, 128, 0),
    "left_eye": (0, 255, 255),
    "right_eye": (0, 255, 255),
    "left_ear": (0, 255, 255),
    "right_ear": (0, 255, 255),
    "left_shoulder": (0, 200, 255),
    "right_shoulder": (0, 200, 255),
    "left_elbow": (255, 0, 255),
    "right_elbow": (255, 0, 255),
    "left_wrist": (255, 0, 128),
    "right_wrist": (255, 0, 128),
    "left_hip": (0, 128, 255),
    "right_hip": (0, 128, 255),
    "left_knee": (255, 255, 0),
    "right_knee": (255, 255, 0),
    "left_ankle": (0, 255, 128),
    "right_ankle": (0, 255, 128),
}


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


class PoseAnnotator:
    """Draws keypoints and skeleton bones onto a copy of the input frame.

    Both drawing helpers are pure: they return new annotated arrays and leave
    the input frame unmodified.
    """

    def __init__(
        self,
        min_confidence: float = DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._min_confidence = min_confidence

    def draw_skeleton(
        self,
        annotated: np.ndarray,
        keypoints: Keypoints,
    ) -> np.ndarray:
        """Return ``annotated`` with skeleton bones drawn between valid joints."""

        result = annotated.copy()
        points = self._valid_points(keypoints)
        for first, second in COCO_17_SKELETON:
            if first in points and second in points:
                cv2.line(
                    result,
                    points[first],
                    points[second],
                    _SKELETON_COLOR,
                    _LINE_WIDTH,
                    cv2.LINE_AA,
                )
        return result

    def draw_keypoints(
        self,
        annotated: np.ndarray,
        keypoints: Keypoints,
    ) -> np.ndarray:
        """Return ``annotated`` with validated keypoints drawn as dots."""

        result = annotated.copy()
        for index, point in self._valid_points(keypoints).items():
            name = COCO_17_KEYPOINT_NAMES[index]
            color = _KEYPOINT_COLORS.get(name, _KEYPOINT_FALLBACK_COLOR)
            cv2.circle(
                result,
                point,
                _KEYPOINT_RADIUS,
                color,
                cv2.FILLED,
                cv2.LINE_AA,
            )
        return result

    def draw_confidence(
        self,
        annotated: np.ndarray,
        keypoints: Keypoints,
    ) -> np.ndarray:
        """Return ``annotated`` with per-keypoint confidence values drawn."""

        result = annotated.copy()
        for index, point in self._valid_points(keypoints).items():
            confidence = keypoints[index][2]
            label = f"{COCO_17_KEYPOINT_NAMES[index]}:{confidence:.2f}"
            cv2.putText(
                result,
                label,
                (point[0] + _TEXT_MARGIN, point[1] - _TEXT_MARGIN),
                _FONT,
                _FONT_SCALE,
                _KEYPOINT_FALLBACK_COLOR,
                _TEXT_LINE_WIDTH,
                cv2.LINE_AA,
            )
        return result

    def annotate(
        self,
        frame: np.ndarray,
        records: list[PoseRecord],
    ) -> np.ndarray:
        """Return ``frame`` with pose keypoints and skeleton for each record.

        The input frame is left unmodified.
        """

        annotated = frame.copy()
        for record in records:
            annotated = self.draw_skeleton(annotated, record.keypoints)
            annotated = self.draw_keypoints(annotated, record.keypoints)
            label = f"Pose #{record.track_id}"
            cv2.putText(
                annotated,
                label,
                (
                    round(record.keypoints[int(Keypoint.NOSE)][0]) + _TEXT_MARGIN,
                    round(record.keypoints[int(Keypoint.NOSE)][1]),
                ),
                _FONT,
                _FONT_SCALE,
                _SKELETON_COLOR,
                _TEXT_LINE_WIDTH,
                cv2.LINE_AA,
            )
        return annotated

    def _valid_points(self, keypoints: Keypoints) -> dict[int, tuple[int, int]]:
        """Map keypoint indices to integer pixels above the confidence gate."""

        return {
            index: (round(keypoints[index][0]), round(keypoints[index][1]))
            for index in range(len(keypoints))
            if keypoints[index][2] >= self._min_confidence
        }


class InteractionAnnotator:
    """Visualizes pair-level interaction features onto a copy of a frame.

    All overlays are pure: the input frame (or pre-annotated copy) is never
    mutated and a new array is returned. CoM positions are computed from the
    passed pose records with the configured keypoint gate.
    """

    def __init__(
        self,
        min_confidence: float = DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD,
    ) -> None:
        self._min_confidence = min_confidence
        self._distance_color = (0, 255, 255)
        self._arrow_color = (255, 0, 255)
        self._label_color = (255, 255, 255)

    def draw_distance(
        self,
        annotated: np.ndarray,
        first: PoseRecord,
        second: PoseRecord,
        distance_h: float,
    ) -> np.ndarray:
        """Draw the CoM connecting line for a pair with its distance label."""

        result = annotated.copy()
        first_com = self._com(first)
        second_com = self._com(second)
        if first_com is None or second_com is None:
            return result
        cv2.line(
            result,
            first_com,
            second_com,
            self._distance_color,
            _LINE_WIDTH,
            cv2.LINE_AA,
        )
        midpoint = (
            (first_com[0] + second_com[0]) // 2,
            (first_com[1] + second_com[1]) // 2,
        )
        self._put_text(result, f"d={distance_h:.2f}H", midpoint)
        return result

    def draw_approach_vectors(
        self,
        annotated: np.ndarray,
        first: PoseRecord,
        second: PoseRecord,
        approach_from_first: float,
        approach_from_second: float,
    ) -> np.ndarray:
        """Draw arrows from each CoM toward the other, scaled by approach speed.

        ``approach_*`` are positive px/s magnitudes (0 when stationary or
        moving away). Arrows grow with the approach but cap at a fixed length.
        """

        result = annotated.copy()
        first_com = self._com(first)
        second_com = self._com(second)
        if first_com is None or second_com is None:
            return result
        maximum = max(approach_from_first, approach_from_second, _APPROACH_ARROW_MAX)
        self._arrow(result, first_com, second_com, approach_from_first / maximum)
        self._arrow(result, second_com, first_com, approach_from_second / maximum)
        return result

    def draw_reciprocity(
        self,
        annotated: np.ndarray,
        first: PoseRecord,
        second: PoseRecord,
        reciprocity: float,
    ) -> np.ndarray:
        """Draw a reciprocity label centered on the pair."""

        result = annotated.copy()
        first_com = self._com(first)
        second_com = self._com(second)
        if first_com is None or second_com is None:
            return result
        midpoint = (
            (first_com[0] + second_com[0]) // 2,
            (first_com[1] + second_com[1]) - 8,
        )
        self._put_text(result, f"recip r={reciprocity:+.2f}", midpoint)
        return result

    # -- internals ---------------------------------------------------------

    def _com(self, record: PoseRecord) -> tuple[int, int] | None:
        com = geometry.center_of_mass(record.keypoints, self._min_confidence)
        if com is None:
            return None
        return (round(com[0]), round(com[1]))

    def _arrow(
        self,
        annotated: np.ndarray,
        origin: tuple[int, int],
        target: tuple[int, int],
        strength: float,
    ) -> None:
        length = geometry.distance(
            (float(origin[0]), float(origin[1])),
            (float(target[0]), float(target[1])),
        )
        if length <= 0.0:
            return
        cv2.arrowedLine(
            annotated,
            origin,
            target,
            self._arrow_color,
            _LINE_WIDTH,
            cv2.LINE_AA,
            tipLength=max(0.05, 0.4 * strength),
        )

    def _put_text(
        self,
        annotated: np.ndarray,
        label: str,
        origin: tuple[int, int],
    ) -> None:
        cv2.putText(
            annotated,
            label,
            (origin[0] + _TEXT_MARGIN, origin[1] - _TEXT_MARGIN),
            _FONT,
            _FONT_SCALE,
            self._label_color,
            _TEXT_LINE_WIDTH,
            cv2.LINE_AA,
        )


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


__all__ = ["FrameAnnotator", "InteractionAnnotator", "PoseAnnotator"]
