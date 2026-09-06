"""Typed configuration for the baseline behavior classifier.

Parses ``configs/models/xgboost_baseline.yaml``:

.. code-block:: yaml

    model:
      family: xgboost
      objective: multi:softprob
      num_class: 4
    hyperparameters: {n_estimators: 300, ...}
    data:
      feature_window_frames: 30
      feature_group: all
    evaluation:
      cv_folds: 5
      scoring: f1_macro
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import TypeAdapter, ValidationError

from safewatch.core.constants import BehaviorClass

XGBOOST_FAMILY = "xgboost"
XGBOOST_OBJECTIVE = "multi:softprob"


class XGBoostBaselineConfig:
    """Frozen, validated view of the XGBoost baseline configuration."""

    def __init__(self, raw: dict[str, Any]) -> None:
        self._raw = _validate(raw)
        model = self._raw["model"]
        hyperparameters = self._raw["hyperparameters"]
        data = self._raw["data"]
        evaluation = self._raw["evaluation"]
        self.model_family: str = model["family"]
        self.objective: str = model["objective"]
        self.num_class: int = model["num_class"]
        self.hyperparameters: dict[str, int | float] = {
            str(key): _as_number(value)
            for key, value in hyperparameters.items()
            if _is_number(value)
        }
        self.feature_window_frames: int = data["feature_window_frames"]
        self.feature_group: str = data["feature_group"]
        self.cv_folds: int = evaluation["cv_folds"]
        self.scoring: str = evaluation["scoring"]
        if self.model_family != XGBOOST_FAMILY:
            raise ValueError(
                f"unsupported model family {self.model_family!r}; "
                f"expected {XGBOOST_FAMILY!r}"
            )
        if self.objective != XGBOOST_OBJECTIVE:
            raise ValueError(
                f"unsupported objective {self.objective!r}; "
                f"expected {XGBOOST_OBJECTIVE!r}"
            )
        if self.num_class != len(BehaviorClass):
            raise ValueError(
                f"num_class must equal the number of behavior classes "
                f"({len(BehaviorClass)}), got {self.num_class}"
            )

    @classmethod
    def from_file(cls, path: str | Path) -> XGBoostBaselineConfig:
        with open(path, encoding="utf-8") as handle:
            raw: dict[str, Any] = yaml.safe_load(handle)
        return cls(raw)

    def to_hyperparameters(self) -> dict[str, int | float]:
        return dict(self.hyperparameters)


def _validate(raw: dict[str, Any]) -> dict[str, Any]:
    try:
        as_any = TypeAdapter(dict[str, Any]).validate_python(raw)
    except ValidationError as exc:
        raise ValueError(f"invalid baseline configuration: {exc}") from exc
    model = as_any.get("model")
    if not isinstance(model, dict) or "family" not in model:
        raise ValueError("baseline configuration requires model.family")
    if "objective" not in model or "num_class" not in model:
        raise ValueError("baseline configuration requires model.objective/num_class")
    hyperparameters = as_any.get("hyperparameters")
    if not isinstance(hyperparameters, dict):
        raise TypeError("baseline configuration requires hyperparameters")
    data = as_any.get("data")
    if not isinstance(data, dict):
        data = {}
    evaluation = as_any.get("evaluation")
    if not isinstance(evaluation, dict):
        evaluation = {}
    return {
        "model": {
            "family": str(model["family"]),
            "objective": str(model["objective"]),
            "num_class": int(model["num_class"]),
        },
        "hyperparameters": hyperparameters,
        "data": {
            "feature_window_frames": int(data.get("feature_window_frames", 30)),
            "feature_group": str(data.get("feature_group", "all")),
        },
        "evaluation": {
            "cv_folds": int(evaluation.get("cv_folds", 5)),
            "scoring": str(evaluation.get("scoring", "f1_macro")),
        },
    }


def _as_number(value: object) -> int | float:
    if isinstance(value, bool):
        raise TypeError(f"expected a number, got {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    raise ValueError(f"expected a number, got {value!r}")


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


__all__ = [
    "XGBOOST_FAMILY",
    "XGBOOST_OBJECTIVE",
    "XGBoostBaselineConfig",
]
