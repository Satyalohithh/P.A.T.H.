"""COCO-17 keypoint layout: canonical names, skeleton bones, and validation.

:class:`safewatch.core.constants.Keypoint` is the frozen index enum; its
ordering (0 nose ... 16 right ankle) must be preserved by every pose source.
The constants here back both validation and visualization (skeleton bones).
"""

from __future__ import annotations

from safewatch.core.constants import Keypoint
from safewatch.core.exceptions import PoseEstimationError
from safewatch.core.types import Keypoints

COCO_17_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)

COCO_17_SKELETON: tuple[tuple[int, int], ...] = (
    (0, 1),  # nose -> left eye
    (0, 2),  # nose -> right eye
    (1, 3),  # left eye -> left ear
    (2, 4),  # right eye -> right ear
    (0, 5),  # nose -> left shoulder
    (0, 6),  # nose -> right shoulder
    (5, 7),  # left shoulder -> left elbow
    (7, 9),  # left elbow -> left wrist
    (6, 8),  # right shoulder -> right elbow
    (8, 10),  # right elbow -> right wrist
    (5, 6),  # left shoulder -> right shoulder
    (5, 11),  # left shoulder -> left hip
    (6, 12),  # right shoulder -> right hip
    (11, 12),  # left hip -> right hip
    (11, 13),  # left hip -> left knee
    (13, 15),  # left knee -> left ankle
    (12, 14),  # right hip -> right knee
    (14, 16),  # right knee -> right ankle
)

KEYPOINT_COUNT = len(Keypoint)
"""Number of keypoints in the COCO-17 layout (must stay 17)."""


def validate_keypoints(keypoints: Keypoints) -> None:
    """Validate that ``keypoints`` follows the frozen COCO-17 layout.

    Raises :class:`PoseEstimationError` when the count is not 17 or any entry
    is malformed (not ``(x, y, confidence)`` with confidence in ``[0, 1]``).
    """

    if len(keypoints) != KEYPOINT_COUNT:
        raise PoseEstimationError(
            f"expected {KEYPOINT_COUNT} keypoints in COCO-17 order, "
            f"got {len(keypoints)}"
        )
    for index, keypoint in enumerate(keypoints):
        if len(keypoint) != 3:
            raise PoseEstimationError(
                f"keypoint {index} must be a (x, y, confidence) triplet, "
                f"got {len(keypoint)} values"
            )
        x, y, conf = (float(v) for v in keypoint)
        if x < 0.0 or y < 0.0:
            raise PoseEstimationError(
                f"keypoint {index} coordinate must be non-negative"
            )
        if not 0.0 <= conf <= 1.0:
            raise PoseEstimationError(
                f"keypoint {index} confidence {conf} is outside [0, 1]"
            )


__all__ = [
    "COCO_17_KEYPOINT_NAMES",
    "COCO_17_SKELETON",
    "KEYPOINT_COUNT",
    "Keypoint",
    "validate_keypoints",
]
