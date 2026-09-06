"""Behavior classifier interface (XGBoost baseline first).

The baseline maps a per-interaction-window feature vector to a
:class:`BehaviorClass` using XGBoost ``multi:softprob`` (see
``configs/models/xgboost_baseline.yaml`` and ``docs/model_design.md``).
Models are persisted as XGBoost native JSON plus a ``.meta.json`` sidecar
carrying feature names, class count, and label order.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost as xgb

from safewatch.behavior.label_schema import LABEL_NAMES
from safewatch.core.constants import BehaviorClass
from safewatch.core.schemas.behavior import ClassificationResult
from safewatch.core.schemas.features import FeatureVector
from safewatch.core.types import FrameIndex, TrackId

TRAIN_SEED = 42
"""Deterministic seed for reproducible baseline training."""

_NUM_CLASSES_DEFAULT = len(BehaviorClass)


class BehaviorClassifier:
    """Maps a feature vector to a BehaviorClass with calibrated probabilities."""

    def __init__(
        self,
        model_path: str | None = None,
        num_classes: int = _NUM_CLASSES_DEFAULT,
        feature_names: tuple[str, ...] | None = None,
    ) -> None:
        self.model_path = model_path
        self.num_classes = num_classes
        self.feature_names = feature_names
        self._model: xgb.XGBClassifier | None = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self) -> xgb.XGBClassifier:
        return self._require_model()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | list[int],
        feature_names: tuple[str, ...] | None = None,
        **hyperparameters: float,
    ) -> BehaviorClassifier:
        """Train the XGBoost baseline on a labeled feature matrix."""

        matrix = np.asarray(X, dtype=np.float32)
        labels = np.asarray(y, dtype=np.int64).ravel()
        if matrix.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {matrix.shape}")
        if matrix.shape[0] != labels.shape[0]:
            raise ValueError(
                f"X rows ({matrix.shape[0]}) and y rows ({labels.shape[0]}) must match"
            )
        if labels.size == 0:
            raise ValueError("cannot train on an empty dataset")
        if len(np.unique(labels)) < 2:
            raise ValueError("training data must contain at least two labels")
        if labels.min() < 0 or labels.max() >= self.num_classes:
            raise ValueError(
                f"labels must be in [0, {self.num_classes}), got "
                f"range [{labels.min()}, {labels.max()}]"
            )
        if feature_names is not None:
            if len(feature_names) != matrix.shape[1]:
                raise ValueError(
                    f"feature_names ({len(feature_names)}) must match X columns "
                    f"({matrix.shape[1]})"
                )
            self.feature_names = tuple(feature_names)
        params: dict[str, object] = dict(hyperparameters)
        params.setdefault("objective", "multi:softprob")
        params.setdefault("num_class", self.num_classes)
        params.setdefault("random_state", TRAIN_SEED)
        params.setdefault("n_jobs", -1)
        if "n_estimators" not in params:
            params["n_estimators"] = 100
        model = xgb.XGBClassifier(**params)
        model.fit(matrix, labels)
        self._model = model
        return self

    def predict(self, vector: FeatureVector) -> ClassificationResult:
        """Predict the class for one feature vector."""

        self._require_feature_alignment(vector.names)
        matrix = np.asarray(vector.values, dtype=np.float32).reshape(1, -1)
        probabilities = self._probabilities(matrix)[0]
        label = int(np.argmax(probabilities))
        return ClassificationResult(
            label=BehaviorClass(label),
            probabilities=tuple(float(p) for p in probabilities),
            frame_index=vector.frame_index,
            track_id=vector.track_id,
        )

    def predict_proba(self, vector: FeatureVector) -> tuple[float, ...]:
        """Return per-class probabilities for one feature vector."""

        self._require_feature_alignment(vector.names)
        matrix = np.asarray(vector.values, dtype=np.float32).reshape(1, -1)
        return tuple(float(p) for p in self._probabilities(matrix)[0])

    def predict_batch(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict labels and probabilities for a batch feature matrix.

        Returns ``(labels, probabilities)`` where ``labels`` are integer class
        ids in ``[0, num_classes)`` and ``probabilities`` has shape
        ``(n, num_classes)``.
        """

        matrix = np.asarray(X, dtype=np.float32)
        probabilities = self._probabilities(matrix)
        labels: np.ndarray = np.argmax(probabilities, axis=1)
        return labels, probabilities

    def save(self, path: str | Path) -> None:
        """Persist the model as XGBoost native JSON plus a meta sidecar."""

        model = self._require_model()
        model.save_model(str(path))
        meta = {
            "family": "xgboost",
            "num_classes": self.num_classes,
            "feature_names": list(self.feature_names or ()),
            "labels": [LABEL_NAMES[cls] for cls in BehaviorClass],
        }
        meta_path = Path(str(path) + ".meta.json")
        with open(meta_path, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, indent=2)

    def load(self) -> None:
        """Load a model saved by :meth:`save` from ``model_path``."""

        if self.model_path is None:
            raise ValueError("model_path is required to load a model")
        model = xgb.XGBClassifier()
        model.load_model(self.model_path)
        self._model = model
        meta_path = Path(str(self.model_path) + ".meta.json")
        if meta_path.is_file():
            with open(meta_path, encoding="utf-8") as handle:
                meta = json.load(handle)
            if "feature_names" in meta:
                self.feature_names = tuple(meta["feature_names"])
            if "num_classes" in meta:
                self.num_classes = int(meta["num_classes"])

    def unload(self) -> None:
        """Release the in-memory model and free its native resources."""

        self._model = None

    def _probabilities(self, matrix: np.ndarray) -> np.ndarray:
        model = self._require_model()
        probabilities = model.predict_proba(matrix)
        if probabilities.shape[1] != self.num_classes:
            probabilities = _pad_probabilities(probabilities, self.num_classes)
        return probabilities

    def _require_feature_alignment(self, names: tuple[str, ...]) -> None:
        if self.feature_names is not None and names != self.feature_names:
            raise ValueError(
                f"feature names {names} do not match the model's feature "
                f"order {self.feature_names}"
            )
        if self.feature_names is None and len(names) == 0:
            raise ValueError(
                "feature vector has unknown dimension; provide feature_names "
                "at construction or training time"
            )

    def _require_model(self) -> xgb.XGBClassifier:
        if self._model is None:
            raise RuntimeError("model is not loaded; call fit() or load() first")
        return self._model


def _pad_probabilities(probabilities: np.ndarray, num_classes: int) -> np.ndarray:
    if probabilities.shape[1] > num_classes:
        probabilities = probabilities[:, :num_classes]
        total = probabilities.sum(axis=1, keepdims=True)
        probabilities = probabilities / np.where(total > 0, total, 1.0)
    else:
        padded = np.zeros((probabilities.shape[0], num_classes), dtype=np.float32)
        padded[:, : probabilities.shape[1]] = probabilities
        probabilities = padded
    return probabilities


__all__ = [
    "TRAIN_SEED",
    "BehaviorClassifier",
    "ClassificationResult",
    "FeatureVector",
    "FrameIndex",
    "TrackId",
]
