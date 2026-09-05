"""Behavior classifier interface (XGBoost baseline first)."""

from __future__ import annotations

from safewatch.core.schemas.behavior import ClassificationResult
from safewatch.core.schemas.features import FeatureVector
from safewatch.core.types import FrameIndex, TrackId


class BehaviorClassifier:
    """Maps a feature vector to a BehaviorClass with calibrated probabilities."""

    def __init__(self, model_path: str, num_classes: int = 4) -> None:
        self.model_path = model_path
        self.num_classes = num_classes

    def predict(self, vector: FeatureVector) -> ClassificationResult:
        raise NotImplementedError("TODO(implementation): BehaviorClassifier.predict")

    def predict_proba(self, vector: FeatureVector) -> tuple[float, ...]:
        raise NotImplementedError(
            "TODO(implementation): BehaviorClassifier.predict_proba"
        )

    def load(self) -> None:
        raise NotImplementedError("TODO(implementation): BehaviorClassifier.load")

    def unload(self) -> None:
        raise NotImplementedError("TODO(implementation): BehaviorClassifier.unload")


__all__ = [
    "BehaviorClassifier",
    "ClassificationResult",
    "FeatureVector",
    "FrameIndex",
    "TrackId",
]
