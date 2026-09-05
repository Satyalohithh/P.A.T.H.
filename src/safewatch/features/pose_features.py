"""Pose feature extractors (catalog group A: A.1-A.7)."""

from __future__ import annotations

from safewatch.core.schemas.features import FeatureVector
from safewatch.core.schemas.pose import PoseResult
from safewatch.features.base import FeatureExtractor


class PoseFeatureExtractor(FeatureExtractor):
    """Computes pose.elbow_angle.*, pose.knee_angle.*, pose.torso_lean,
    pose.shoulder_orientation, pose.upper_asymmetry, pose.complexity,
    pose.com_height for a window of poses.

    Formula/failure-mode spec: docs/fp-doc-02 (feature extraction spec),
    group A entries.
    """

    def __init__(self, nan_policy: str = "skip_feature") -> None:
        super().__init__(
            names=(
                "pose.elbow_angle.L",
                "pose.elbow_angle.R",
                "pose.knee_angle.L",
                "pose.knee_angle.R",
                "pose.torso_lean",
                "pose.shoulder_orientation",
                "pose.upper_asymmetry",
                "pose.complexity",
                "pose.com_height",
            ),
            nan_policy=nan_policy,
        )

    def extract(self, poses: list[PoseResult]) -> FeatureVector:
        raise NotImplementedError("TODO(implementation): PoseFeatureExtractor.extract")

    # Helper stubs mapped to catalog entries (implemented in phase 2).
    def elbow_angle(self, pose: PoseResult, side: str) -> float:
        raise NotImplementedError(
            "TODO(implementation): PoseFeatureExtractor.elbow_angle"
        )

    def knee_angle(self, pose: PoseResult, side: str) -> float:
        raise NotImplementedError(
            "TODO(implementation): PoseFeatureExtractor.knee_angle"
        )

    def torso_lean(self, pose: PoseResult) -> float:
        raise NotImplementedError(
            "TODO(implementation): PoseFeatureExtractor.torso_lean"
        )

    def shoulder_orientation(self, pose: PoseResult) -> float:
        raise NotImplementedError(
            "TODO(implementation): PoseFeatureExtractor.shoulder_orientation"
        )

    def upper_asymmetry(self, pose: PoseResult) -> float:
        raise NotImplementedError(
            "TODO(implementation): PoseFeatureExtractor.upper_asymmetry"
        )

    def complexity(self, pose: PoseResult) -> float:
        raise NotImplementedError(
            "TODO(implementation): PoseFeatureExtractor.complexity"
        )

    def com_height(self, pose: PoseResult) -> float:
        raise NotImplementedError(
            "TODO(implementation): PoseFeatureExtractor.com_height"
        )


__all__ = ["FeatureVector", "PoseFeatureExtractor", "PoseResult"]
