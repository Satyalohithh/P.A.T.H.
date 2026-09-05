"""Keypoint geometry helpers (angles, distances, normalization)."""

from __future__ import annotations

import math

from safewatch.core.types import Keypoint, Keypoints, Vec2D


class KeypointUtils:
    """Pure geometry utilities shared by feature extractors."""

    def __init__(self, epsilon: float = 1e-9) -> None:
        self.epsilon = epsilon

    def vector(self, a: Keypoint, b: Keypoint) -> Vec2D:
        """2D vector from keypoint ``a`` to keypoint ``b``."""

        return (b[0] - a[0], b[1] - a[1])

    def norm(self, v: Vec2D) -> float:
        return math.hypot(v[0], v[1])

    def angle_three_points(
        self, apex: Keypoint, left: Keypoint, right: Keypoint
    ) -> float:
        """Signed angle (radians) at ``apex`` formed by ``left-apex-right``."""

        raise NotImplementedError(
            "TODO(implementation): KeypointUtils.angle_three_points"
        )

    def to_degrees(self, radians: float) -> float:
        return math.degrees(radians)

    def head_height(self, keypoints: Keypoints) -> float:
        """H: distance (px) between toe and head, per the extraction spec."""

        raise NotImplementedError("TODO(implementation): KeypointUtils.head_height")


__all__ = ["Keypoint", "KeypointUtils", "Keypoints", "Vec2D"]
