"""Feature-importance analysis (SHAP / permutation)."""

from __future__ import annotations

from safewatch.core.schemas.features import FeatureVector


class FeatureImportance:
    """Ranks features by their contribution to predictions."""

    def __init__(self, feature_names: tuple[str, ...]) -> None:
        self.feature_names = feature_names

    def permutation_importance(
        self, model: object, vectors: list[FeatureVector], y: list[int]
    ) -> dict[str, float]:
        raise NotImplementedError(
            "TODO(implementation): FeatureImportance.permutation_importance"
        )

    def shap_values(
        self, model: object, vectors: list[FeatureVector]
    ) -> dict[str, float]:
        raise NotImplementedError("TODO(implementation): FeatureImportance.shap_values")


__all__ = ["FeatureImportance", "FeatureVector"]
