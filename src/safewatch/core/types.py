"""Structural type aliases shared across SafeWatch AI.

These aliases are intentionally concrete (built-in types only) so that
modules stay importable before third-party dependencies (numpy, torch, ...)
are installed.
"""

from __future__ import annotations

from typing import TypeAlias

TrackId: TypeAlias = int
"""Identity of a tracked person (assigned by the tracker, stable over frames)."""

StreamId: TypeAlias = str
"""Identity of a video source / stream (e.g. "cam-01")."""

FrameIndex: TypeAlias = int
"""0-based index of a frame within its source stream."""

TimePoint: TypeAlias = float
"""Timestamps in seconds (monotonic source clock or stream-relative)."""

Confidence: TypeAlias = float
"""Probability-like score in [0.0, 1.0]."""

Keypoint: TypeAlias = tuple[float, float, float]
"""A single keypoint as (x, y, confidence). See constants.Keypoint for indices."""

Keypoints: TypeAlias = tuple[Keypoint, ...]
"""Full 17-keypoint COCO pose, aligned with constants.Keypoint order."""

Vec2D: TypeAlias = tuple[float, float]
"""2D vector / point in image-pixel coordinates."""

__all__ = [
    "Confidence",
    "FrameIndex",
    "Keypoint",
    "Keypoints",
    "StreamId",
    "TimePoint",
    "TrackId",
    "Vec2D",
]
