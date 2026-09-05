"""Motion feature extractors (catalog group B: B.1-B.6)."""

from __future__ import annotations

from safewatch.core.schemas.features import FeatureVector
from safewatch.core.schemas.pose import PoseResult
from safewatch.features.base import FeatureExtractor
from safewatch.features.feature_names import FEATURE_NAMES_RAW

MOTION_FEATURE_NAMES = FEATURE_NAMES_RAW["motion"]


class MotionFeatureExtractor(FeatureExtractor):
    """Computes limb angular velocity/acceleration, CoM kinematics, limb
    energy, stride length, and gait regularity over a pose window.

    Formula/failure-mode spec: docs/fp-doc-02 (feature extraction spec),
    group B entries (central differences with frame-time normalization).
    """

    def __init__(self, nan_policy: str = "skip_feature") -> None:
        super().__init__(names=MOTION_FEATURE_NAMES, nan_policy=nan_policy)

    def extract(self, poses: list[PoseResult]) -> FeatureVector:
        raise NotImplementedError(
            "TODO(implementation): MotionFeatureExtractor.extract"
        )

    def limb_angular_velocity(self, poses: list[PoseResult], limb: str) -> float:
        raise NotImplementedError(
            "TODO(implementation): MotionFeatureExtractor.limb_angular_velocity"
        )

    def limb_angular_acceleration(self, poses: list[PoseResult], limb: str) -> float:
        raise NotImplementedError(
            "TODO(implementation): MotionFeatureExtractor.limb_angular_acceleration"
        )

    def center_of_mass(self, pose: PoseResult) -> tuple[float, float]:
        raise NotImplementedError(
            "TODO(implementation): MotionFeatureExtractor.center_of_mass"
        )

    def limb_energy(self, poses: list[PoseResult]) -> float:
        raise NotImplementedError(
            "TODO(implementation): MotionFeatureExtractor.limb_energy"
        )

    def stride_length(self, poses: list[PoseResult]) -> float:
        raise NotImplementedError(
            "TODO(implementation): MotionFeatureExtractor.stride_length"
        )


__all__ = ["FeatureVector", "MotionFeatureExtractor", "PoseResult"]
