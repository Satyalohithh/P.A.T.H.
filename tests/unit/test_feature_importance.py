"""Tests for feature-importance analysis and rendering."""

from __future__ import annotations

import numpy as np
import pytest

from safewatch.evaluation.feature_importance import FeatureImportance

NAMES = ("a", "b", "c")
VECTOR_IMPORTANT = (0.5, 0.3, 0.2)


class _FakeModel:
    def __init__(self, importance: np.ndarray) -> None:
        self.feature_importances_ = importance


def test_gain_importance_normalized_and_aligned() -> None:
    model = _FakeModel(np.asarray(VECTOR_IMPORTANT))
    importance = FeatureImportance(NAMES).gain_importance(model)
    assert math_close(sum(importance.values()), 1.0)
    assert list(importance) == ["a", "b", "c"]
    assert importance["a"] == pytest.approx(0.5)


def test_gain_importance_unused_features_map_to_zero() -> None:
    model = _FakeModel(np.asarray([0.0, 0.3, 0.7]))
    importance = FeatureImportance(NAMES).gain_importance(model)
    assert importance["a"] == 0.0
    assert math_close(sum(importance.values()), 1.0)


def test_gain_importance_rejects_dimension_mismatch() -> None:
    model = _FakeModel(np.asarray([0.5, 0.5]))
    with pytest.raises(ValueError, match="does not match"):
        FeatureImportance(NAMES).gain_importance(model)


def test_gain_importance_rejects_unfitted_model() -> None:
    with pytest.raises(ValueError, match="feature_importances_"):
        FeatureImportance(NAMES).gain_importance(object())


def test_zero_total_importance_yields_zeros() -> None:
    model = _FakeModel(np.zeros(3))
    importance = FeatureImportance(NAMES).gain_importance(model)
    assert all(value == 0.0 for value in importance.values())


def test_plot_writes_png(tmp_path) -> None:
    model = _FakeModel(np.asarray(VECTOR_IMPORTANT))
    output = tmp_path / "importance.png"
    FeatureImportance(NAMES).plot_importance(model, output, top_k=2)
    assert output.is_file()
    assert output.stat().st_size > 0
    with open(output, "rb") as handle:
        assert handle.read(8) == b"\x89PNG\r\n\x1a\n"


def test_permutation_and_shap_remain_todos() -> None:
    importance = FeatureImportance(NAMES)
    with pytest.raises(NotImplementedError):
        importance.permutation_importance(object(), [], [])
    with pytest.raises(NotImplementedError):
        importance.shap_values(object(), [])


def math_close(left: float, right: float) -> bool:
    return abs(left - right) < 1e-9
