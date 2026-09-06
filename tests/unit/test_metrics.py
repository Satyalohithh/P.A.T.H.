"""Tests for the evaluation metrics and confusion analysis."""

from __future__ import annotations

import numpy as np
import pytest

from safewatch.evaluation.confusion_analysis import ConfusionAnalysis
from safewatch.evaluation.metrics import Metrics, MetricsReport


def _tiny_case() -> tuple[list[int], list[int]]:
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 0, 1]
    return y_true, y_pred


def test_accuracy_hand_case() -> None:
    y_true, y_pred = _tiny_case()
    assert Metrics().accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_per_class_scores_hand_case() -> None:
    metrics = Metrics()
    y_true, y_pred = _tiny_case()
    assert metrics.class_precision(y_true, y_pred, 0) == pytest.approx(0.5)
    assert metrics.class_recall(y_true, y_pred, 0) == pytest.approx(0.5)
    assert metrics.class_precision(y_true, y_pred, 1) == pytest.approx(0.5)
    assert metrics.class_recall(y_true, y_pred, 1) == pytest.approx(0.5)


def test_macro_f1_hand_case() -> None:
    metrics = Metrics()
    y_true, y_pred = _tiny_case()
    assert metrics.macro_f1(y_true, y_pred) == pytest.approx(0.25)


def test_confusion_matrix_shape_and_values() -> None:
    y_true, y_pred = _tiny_case()
    confusion = Metrics().confusion_matrix(y_true, y_pred)
    assert confusion.shape == (4, 4)
    assert confusion[0, 0] == 1
    assert confusion[0, 1] == 1
    assert confusion[1, 0] == 1
    assert confusion[1, 1] == 1
    assert confusion[2, 2] == 0


def test_report_structure() -> None:
    report = Metrics().report(*_tiny_case())
    assert isinstance(report, MetricsReport)
    assert set(report.per_class) == {0, 1, 2, 3}
    assert report.per_class[0].f1 == pytest.approx(0.5)
    assert report.per_class[2].support == 0


def test_report_zero_division_is_harmless() -> None:
    report = Metrics().report([0, 0], [1, 1])
    assert report.accuracy == 0.0
    assert report.per_class[0].precision == 0.0
    assert report.per_class[1].recall == 0.0


def test_confusion_analysis_worst_offdiagonal() -> None:
    confusion = [[3, 1, 0, 0], [2, 5, 3, 0], [0, 0, 4, 1], [1, 0, 0, 9]]
    analysis = ConfusionAnalysis()
    assert analysis.worst_offdiagonal(confusion) == (1, 2)


def test_confusion_analysis_matrix_returns_lists_of_ints() -> None:
    y_true = [0, 1, 2, 3]
    y_pred = [0, 1, 2, 3]
    matrix = ConfusionAnalysis().matrix(y_true, y_pred)
    assert matrix == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    assert all(isinstance(value, int) for row in matrix for value in row)


def test_confusion_analysis_numpy_matrix_supported() -> None:
    matrix = np.eye(4, dtype=int)
    assert ConfusionAnalysis().worst_offdiagonal(matrix.tolist()) == (0, 1)


def test_roc_auc_still_stubbed() -> None:
    with pytest.raises(NotImplementedError):
        Metrics().roc_auc_binary([0, 1], [0.1, 0.9])
