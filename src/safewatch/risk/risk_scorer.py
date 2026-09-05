"""Risk scoring: maps classification outcomes to a risk level."""

from __future__ import annotations

from safewatch.core.constants import RiskLevel
from safewatch.core.schemas.behavior import ClassificationResult
from safewatch.core.schemas.risk import RiskAssessment
from safewatch.core.types import FrameIndex, StreamId, TrackId


class RiskScorer:
    """Computes a risk score and level from classification results."""

    def __init__(
        self,
        thresholds: tuple[float, float, float] = (0.3, 0.6, 0.85),
    ) -> None:
        """``thresholds`` split the score axis into LOW/MEDIUM/HIGH/CRITICAL."""

        self.thresholds = thresholds

    def score(self, result: ClassificationResult, **context: object) -> RiskAssessment:
        raise NotImplementedError("TODO(implementation): RiskScorer.score")

    def level_for(self, score: float) -> RiskLevel:
        raise NotImplementedError("TODO(implementation): RiskScorer.level_for")


__all__ = [
    "ClassificationResult",
    "FrameIndex",
    "RiskAssessment",
    "RiskLevel",
    "RiskScorer",
    "StreamId",
    "TrackId",
]
