"""Base contract for feature extractors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from safewatch.core.schemas.features import FeatureVector
from safewatch.core.schemas.pose import PoseResult


class FeatureExtractorError(Exception):
    """Raised when a feature extractor cannot produce a valid vector."""


@dataclass
class FeatureExtractor(ABC):
    """Abstract single-window feature extractor.

    Concrete extractors compute a contiguous slice of the frozen feature
    vector for the given pose window.
    """

    names: tuple[str, ...]
    nan_policy: str = "skip_feature"  # skip_feature | zero_fill | raise

    @abstractmethod
    def extract(self, poses: list[PoseResult]) -> FeatureVector:
        """Compute the feature vector for a window of poses.

        ``poses`` is ordered by frame index; per catalog B/C features need
        at least two consecutive valid frames.
        """

        raise NotImplementedError("TODO(implementation): FeatureExtractor.extract")

    def validate(self, pose: PoseResult) -> bool:
        """Return True if ``pose`` has enough valid keypoints to use."""

        raise NotImplementedError("TODO(implementation): FeatureExtractor.validate")


FEATURE_EXTRACTOR_REGISTRY: dict[str, type[FeatureExtractor]] = {}


def register_feature_extractor(
    extractor_cls: type[FeatureExtractor],
) -> type[FeatureExtractor]:
    """Class decorator registering an extractor by its names' first member."""

    raise NotImplementedError("TODO(implementation): register_feature_extractor")


class FeatureExtractorWrapper:
    """Registry-backed alias (place to add alternate registries)."""

    registry: dict[str, type[FeatureExtractor]] = field(
        default_factory=lambda: FEATURE_EXTRACTOR_REGISTRY
    )


__all__ = [
    "FEATURE_EXTRACTOR_REGISTRY",
    "FeatureExtractor",
    "FeatureExtractorError",
    "FeatureExtractorWrapper",
    "PoseResult",
    "register_feature_extractor",
]
