"""Data augmentation for sequence inputs."""

from __future__ import annotations

from safewatch.core.schemas.features import FeatureVector


class AugmentationPipeline:
    """Deterministic/noise augmentations for pose-window inputs."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def apply_random_crop_scale(self, vector: FeatureVector) -> FeatureVector:
        raise NotImplementedError(
            "TODO(implementation): AugmentationPipeline.apply_random_crop_scale"
        )

    def add_keypoint_noise(self, vector: FeatureVector) -> FeatureVector:
        raise NotImplementedError(
            "TODO(implementation): AugmentationPipeline.add_keypoint_noise"
        )


__all__ = ["AugmentationPipeline", "FeatureVector"]
