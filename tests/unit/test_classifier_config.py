"""Tests for the XGBoost baseline config parser."""

from __future__ import annotations

import pathlib

import pytest

from safewatch.behavior.config import (
    XGBOOST_FAMILY,
    XGBOOST_OBJECTIVE,
    XGBoostBaselineConfig,
)

CONFIG_PATH = pathlib.Path("configs/models/xgboost_baseline.yaml")


def test_load_from_real_config() -> None:
    if not CONFIG_PATH.exists():
        pytest.skip("config file not found in environment")
    config = XGBoostBaselineConfig.from_file(CONFIG_PATH)
    assert config.model_family == XGBOOST_FAMILY
    assert config.objective == XGBOOST_OBJECTIVE
    assert config.num_class == 4
    assert config.cv_folds == 5
    assert config.feature_window_frames == 30


def test_to_hyperparameters_returns_dict() -> None:
    config = XGBoostBaselineConfig(
        {
            "model": {
                "family": "xgboost",
                "objective": "multi:softprob",
                "num_class": 4,
            },
            "hyperparameters": {"n_estimators": 50, "max_depth": 3, "lr": 0.1},
            "data": {"feature_window_frames": 30, "feature_group": "all"},
            "evaluation": {"cv_folds": 5, "scoring": "f1_macro"},
        }
    )
    hp = config.to_hyperparameters()
    assert hp["n_estimators"] == 50
    assert hp["max_depth"] == 3
    assert hp["lr"] == 0.1


def test_rejects_wrong_family() -> None:
    with pytest.raises(ValueError, match="unsupported model family"):
        XGBoostBaselineConfig(
            {
                "model": {
                    "family": "xgpt",
                    "objective": "multi:softprob",
                    "num_class": 4,
                },
                "hyperparameters": {},
            }
        )


def test_rejects_wrong_objective() -> None:
    with pytest.raises(ValueError, match="unsupported objective"):
        XGBoostBaselineConfig(
            {
                "model": {
                    "family": "xgboost",
                    "objective": "binary:logistic",
                    "num_class": 4,
                },
                "hyperparameters": {},
            }
        )


def test_rejects_wrong_num_class() -> None:
    with pytest.raises(ValueError, match="num_class must equal"):
        XGBoostBaselineConfig(
            {
                "model": {
                    "family": "xgboost",
                    "objective": "multi:softprob",
                    "num_class": 3,
                },
                "hyperparameters": {},
            }
        )


def test_rejects_missing_model_key() -> None:
    with pytest.raises(ValueError, match="model.family"):
        XGBoostBaselineConfig({"hyperparameters": {}})


def test_rejects_missing_hyperparameters() -> None:
    with pytest.raises(TypeError, match="hyperparameters"):
        XGBoostBaselineConfig(
            {
                "model": {
                    "family": "xgboost",
                    "objective": "multi:softprob",
                    "num_class": 4,
                }
            }
        )


def test_defaults_when_data_and_evaluation_absent() -> None:
    config = XGBoostBaselineConfig(
        {
            "model": {
                "family": "xgboost",
                "objective": "multi:softprob",
                "num_class": 4,
            },
            "hyperparameters": {"n_estimators": 20},
        }
    )
    assert config.feature_window_frames == 30
    assert config.feature_group == "all"
    assert config.cv_folds == 5
    assert config.scoring == "f1_macro"
