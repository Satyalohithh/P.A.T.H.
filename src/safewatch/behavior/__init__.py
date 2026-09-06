"""Behavior classification: label schema, dataset tooling, classifiers."""

from __future__ import annotations

from safewatch.behavior.classifier import TRAIN_SEED, BehaviorClassifier
from safewatch.behavior.config import (
    XGBOOST_FAMILY,
    XGBOOST_OBJECTIVE,
    XGBoostBaselineConfig,
)
from safewatch.behavior.dataset import (
    DatasetSplit,
    assign_labels,
    feature_names,
    load_dataset,
    save_dataset,
    train_test_split,
    validate_labels,
)
from safewatch.behavior.ensemble import EnsembleClassifier
from safewatch.behavior.label_schema import (
    LABEL_NAMES,
    LABELS,
    label_name,
    label_value,
    parse_label,
)
from safewatch.behavior.temporal_models import TemporalClassifier
from safewatch.behavior.windowing import build_pair_windows

__all__ = [
    "LABELS",
    "LABEL_NAMES",
    "TRAIN_SEED",
    "XGBOOST_FAMILY",
    "XGBOOST_OBJECTIVE",
    "BehaviorClassifier",
    "DatasetSplit",
    "EnsembleClassifier",
    "TemporalClassifier",
    "XGBoostBaselineConfig",
    "assign_labels",
    "build_pair_windows",
    "feature_names",
    "label_name",
    "label_value",
    "load_dataset",
    "parse_label",
    "save_dataset",
    "train_test_split",
    "validate_labels",
]
