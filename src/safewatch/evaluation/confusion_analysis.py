"""Confusion-matrix analysis."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix as _sk_confusion_matrix

from safewatch.behavior.label_schema import LABEL_NAMES
from safewatch.core.constants import BehaviorClass


class ConfusionAnalysis:
    """Builds and summarizes the confusion matrix."""

    def __init__(self, num_classes: int = len(BehaviorClass)) -> None:
        self.num_classes = num_classes

    def matrix(self, y_true: Sequence[int], y_pred: Sequence[int]) -> list[list[int]]:
        cm = _sk_confusion_matrix(
            y_true, y_pred, labels=_labels(self.num_classes)
        ).tolist()
        return [[int(value) for value in row] for row in cm]

    def worst_offdiagonal(self, matrix: list[list[int]]) -> tuple[int, int]:
        """Return the (true, predicted) pair with the most confusions."""

        best: tuple[int, int] | None = None
        for true_idx, row in enumerate(matrix):
            for pred_idx, value in enumerate(row):
                if true_idx == pred_idx:
                    continue
                if best is None or value > matrix[best[0]][best[1]]:
                    best = (true_idx, pred_idx)
        if best is None:
            raise ValueError("confusion matrix has no off-diagonal entries")
        return best


def save_confusion_matrix(confusion: np.ndarray, output_path: str) -> str:
    """Render a confusion matrix as a value-annotated heatmap PNG.

    Confusion truth is on the y-axis (rows) and predictions on the x-axis
    (columns); class names come from the behavior label schema.
    """

    labels = [LABEL_NAMES[cls] for cls in BehaviorClass]
    _fig, axis = plt.subplots(figsize=(6, 5))
    matrix = np.asarray(confusion)
    im = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(matrix.shape[1]))
    axis.set_xticklabels(labels[: matrix.shape[1]], rotation=30, ha="right")
    axis.set_yticks(range(matrix.shape[0]))
    axis.set_yticklabels(labels[: matrix.shape[0]])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(
                j,
                i,
                str(int(matrix[i, j])),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() * 0.5 else "black",
            )
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    axis.set_title("Confusion matrix")
    figure = cast(plt.Figure, axis.figure)
    figure.colorbar(im, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return str(output_path)


def _labels(num_classes: int) -> list[int]:
    return list(range(num_classes))


__all__ = [
    "BehaviorClass",
    "ConfusionAnalysis",
    "save_confusion_matrix",
]
