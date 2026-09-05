"""Explainability: which features drove a risk assessment."""

from __future__ import annotations

from safewatch.core.schemas.features import FeatureVector
from safewatch.core.schemas.risk import RiskAssessment, RiskExplanation


class Explanator:
    """Produces human-readable explanations for risk assessments."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def explain(
        self,
        assessment: RiskAssessment,
        vector: FeatureVector,
    ) -> RiskExplanation:
        raise NotImplementedError("TODO(implementation): Explanator.explain")


__all__ = ["Explanator", "FeatureVector", "RiskAssessment", "RiskExplanation"]
