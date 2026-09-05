"""Feature extraction: intake, validation, concatenation."""

from __future__ import annotations

from collections.abc import Iterable

from safewatch.core.schemas.features import FeatureVector
from safewatch.core.schemas.pose import PoseResult
from safewatch.features.base import FeatureExtractor
from safewatch.features.feature_names import FEATURE_NAMES


class FeaturePipeline:
    """Runs the configured extractors and concatenates their outputs."""

    def __init__(
        self,
        extractors: Iterable[FeatureExtractor],
        expected_dimension: int | None = None,
    ) -> None:
        self.extractors = tuple(extractors)
        self.expected_dimension = expected_dimension or len(FEATURE_NAMES)

    def compute(self, poses: list[PoseResult]) -> FeatureVector:
        raise NotImplementedError("TODO(implementation): FeaturePipeline.compute")

    @property
    def feature_order(self) -> tuple[str, ...]:
        raise NotImplementedError("TODO(implementation): FeaturePipeline.feature_order")


__all__ = ["FEATURE_NAMES", "FeaturePipeline", "FeatureVector", "PoseResult"]
