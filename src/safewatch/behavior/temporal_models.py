"""Temporal sequence classifiers (BiLSTM-attention, temporal CNN/Transformer)."""

from __future__ import annotations

from safewatch.core.schemas.behavior import ClassificationResult
from safewatch.core.schemas.features import FeatureVector


class TemporalClassifier:
    """Sequence model over frame-level windows for ClassifyWindow."""

    def __init__(self, model_family: str, model_path: str) -> None:
        self.model_family = model_family
        self.model_path = model_path

    def predict(self, window: list[FeatureVector]) -> ClassificationResult:
        raise NotImplementedError("TODO(implementation): TemporalClassifier.predict")

    def load(self) -> None:
        raise NotImplementedError("TODO(implementation): TemporalClassifier.load")


__all__ = ["ClassificationResult", "FeatureVector", "TemporalClassifier"]
