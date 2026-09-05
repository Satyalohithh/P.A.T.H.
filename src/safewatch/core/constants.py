"""Enums and default constants for SafeWatch AI.

The numeric values here are FROZEN: changing them corrupts models, saved
feature vectors, and alert history.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum

ENV_PREFIX = "SAFEWATCH_"
"""Prefix for environment-variable configuration overrides."""


class Keypoint(IntEnum):
    """COCO 17-keypoint skeleton layout."""

    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


class BehaviorClass(IntEnum):
    """Four-way behavioral classification target."""

    NORMAL = 0
    PLAYFUL = 1
    SUSPICIOUS = 2
    AGGRESSIVE = 3


class RiskLevel(IntEnum):
    """Risk escalation levels used by the alert engine."""

    LOW = 0
    MEDIUM = 1
    HIGH = 2
    CRITICAL = 3


class AlertStatus(StrEnum):
    """Lifecycle state of an alert."""

    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD = 0.45
"""Keypoints below this confidence are treated as missing."""

MIN_VALID_KEYPOINTS_FOR_POSE = 7
"""Fewer valid keypoints than this invalidates per-person pose features."""

DEFAULT_INTERACTION_WINDOW_FRAMES = 30
"""Sliding window (in frames) that defines an interaction observation."""

DEFAULT_CONTACT_PROXIMITY_GATE_H = 3.0
"""Pair distance (in head units) below which interaction features are computed."""

INTIMATE_ZONE_H = 0.5
PERSONAL_ZONE_H = 1.5
SOCIAL_ZONE_H = 4.0
"""Proxemic zone boundaries expressed in units of head height (H)."""

SMOOTHING_SIGMA_FRAMES = 2.0
"""Gaussian sigma used to smooth raw feature trajectories."""

SMOOTHING_KERNEL_WIDTH = 5
"""Gaussian kernel width (in frames) used for smoothing."""

EPSILON = 1e-9
"""Numerical guard against division by zero."""

FEATURE_GROUPS = ("pose", "motion", "interaction")
"""Feature groups contributing to behavior classification."""

__all__ = [
    "DEFAULT_CONTACT_PROXIMITY_GATE_H",
    "DEFAULT_INTERACTION_WINDOW_FRAMES",
    "DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD",
    "ENV_PREFIX",
    "EPSILON",
    "FEATURE_GROUPS",
    "INTIMATE_ZONE_H",
    "MIN_VALID_KEYPOINTS_FOR_POSE",
    "PERSONAL_ZONE_H",
    "SMOOTHING_KERNEL_WIDTH",
    "SMOOTHING_SIGMA_FRAMES",
    "SOCIAL_ZONE_H",
    "AlertStatus",
    "BehaviorClass",
    "Keypoint",
    "RiskLevel",
]
