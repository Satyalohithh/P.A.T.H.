"""Feature-importance analysis (SHAP / permutation)."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from safewatch.core.schemas.features import FeatureVector


class FeatureImportance:
    """Ranks features by their contribution to predictions."""

    def __init__(self, feature_names: tuple[str, ...]) -> None:
        self.feature_names = feature_names

    def gain_importance(self, model: object) -> dict[str, float]:
        """Return normalized XGBoost gain importance aligned to feature names.

        ``model`` must expose a ``feature_importances_`` attribute (an
        ``xgboost.XGBClassifier`` after :meth:`fit`); missing features map to
        zero. Values are normalized to sum to 1.0.
        """

        importances = getattr(model, "feature_importances_", None)
        if importances is None:
            raise ValueError(
                "model does not expose feature_importances_; expected a fitted "
                "xgboost XGBClassifier"
            )
        raw = np.asarray(importances, dtype=np.float64).ravel()
        if raw.size != len(self.feature_names):
            raise ValueError(
                f"model importance dim ({raw.size}) does not match feature names "
                f"({len(self.feature_names)})"
            )
        total = float(raw.sum())
        if total <= 0:
            return {name: 0.0 for name in self.feature_names}
        return {
            name: float(value) / total
            for name, value in zip(self.feature_names, raw, strict=True)
        }

    def plot_importance(
        self,
        model: object,
        output_path: str | Path,
        top_k: int | None = None,
    ) -> str:
        """Render a horizontal bar chart of gain importance to ``output_path``."""

        importance = self.gain_importance(model)
        names = list(importance.keys())
        values = [importance[name] for name in names]
        order = np.argsort(values)[::-1]
        names = [names[i] for i in order]
        values = [values[i] for i in order]
        if top_k is not None:
            names = names[:top_k]
            values = values[:top_k]
        _fig, axis = plt.subplots(figsize=(6.5, max(2.0, len(names) * 0.4)))
        axis.barh(names[::-1], values[::-1], color="#4c72b0")
        axis.set_xlabel("Normalized gain importance")
        axis.set_title("XGBoost feature importance")
        axis.grid(axis="x", alpha=0.3)
        figure = cast(plt.Figure, axis.figure)
        figure.tight_layout()
        figure.savefig(str(output_path), dpi=150)
        plt.close(figure)
        return str(output_path)

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
