"""Evaluation metrics (macro F1, class-wise precision/recall, ROC-AUC)."""

from __future__ import annotations

from safewatch.core.constants import BehaviorClass


class Metrics:
    """Computes the metric set tracked for every experiment."""

    def __init__(self, num_classes: int = len(BehaviorClass)) -> None:
        self.num_classes = num_classes

    def macro_f1(self, y_true: list[int], y_pred: list[int]) -> float:
        raise NotImplementedError("TODO(implementation): Metrics.macro_f1")

    def class_precision(self, y_true: list[int], y_pred: list[int], cls: int) -> float:
        raise NotImplementedError("TODO(implementation): Metrics.class_precision")

    def class_recall(self, y_true: list[int], y_pred: list[int], cls: int) -> float:
        raise NotImplementedError("TODO(implementation): Metrics.class_recall")

    def roc_auc_binary(self, y_true: list[int], scores: list[float]) -> float:
        raise NotImplementedError("TODO(implementation): Metrics.roc_auc_binary")


__all__ = ["BehaviorClass", "Metrics"]
