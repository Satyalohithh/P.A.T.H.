"""Ensemble of behavior classifiers with voting."""

from __future__ import annotations

from collections.abc import Iterable

from safewatch.behavior.classifier import BehaviorClassifier
from safewatch.core.schemas.behavior import ClassificationResult
from safewatch.core.schemas.features import FeatureVector


class EnsembleClassifier:
    """Soft-voting ensemble over member classifiers."""

    def __init__(
        self,
        members: Iterable[BehaviorClassifier],
        weights: tuple[float, ...] | None = None,
    ) -> None:
        self.members = tuple(members)
        self.weights = (
            weights if weights is not None else tuple(1.0 for _ in self.members)
        )

    def predict(self, vector: FeatureVector) -> ClassificationResult:
        raise NotImplementedError("TODO(implementation): EnsembleClassifier.predict")


__all__ = [
    "BehaviorClassifier",
    "ClassificationResult",
    "EnsembleClassifier",
    "FeatureVector",
]
