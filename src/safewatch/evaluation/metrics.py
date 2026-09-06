"""Evaluation metrics (macro F1, class-wise precision/recall, ROC-AUC)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
)

from safewatch.core.constants import BehaviorClass


@dataclass(frozen=True, slots=True)
class ClassScores:
    """Per-class precision / recall / F1."""

    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True, slots=True)
class MetricsReport:
    """The full metric set for one evaluation run."""

    accuracy: float
    macro_f1: float
    per_class: dict[int, ClassScores]
    confusion: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    num_classes: int = len(BehaviorClass)

    def class_f1(self, cls: int) -> float:
        return self.per_class[cls].f1


class Metrics:
    """Computes the metric set tracked for every experiment."""

    def __init__(self, num_classes: int = len(BehaviorClass)) -> None:
        self.num_classes = num_classes

    def accuracy(self, y_true: list[int], y_pred: list[int]) -> float:
        return float(accuracy_score(y_true, y_pred))

    def macro_f1(self, y_true: list[int], y_pred: list[int]) -> float:
        report = self._report(y_true, y_pred)
        return report.macro_f1

    def class_precision(self, y_true: list[int], y_pred: list[int], cls: int) -> float:
        report = self._report(y_true, y_pred)
        return report.per_class[cls].precision

    def class_recall(self, y_true: list[int], y_pred: list[int], cls: int) -> float:
        report = self._report(y_true, y_pred)
        return report.per_class[cls].recall

    def class_f1(self, y_true: list[int], y_pred: list[int], cls: int) -> float:
        report = self._report(y_true, y_pred)
        return report.per_class[cls].f1

    def confusion_matrix(self, y_true: list[int], y_pred: list[int]) -> np.ndarray:
        return sk_confusion_matrix(  # type: ignore[no-any-return]
            y_true, y_pred, labels=_labels(self.num_classes)
        )

    def report(self, y_true: list[int], y_pred: list[int]) -> MetricsReport:
        return self._report(y_true, y_pred)

    def _report(self, y_true: list[int], y_pred: list[int]) -> MetricsReport:
        labels = _labels(self.num_classes)
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, zero_division=0
        )
        per_class = {
            label: ClassScores(
                precision=float(precision[i]),
                recall=float(recall[i]),
                f1=float(f1[i]),
                support=int(support[i]),
            )
            for i, label in enumerate(labels)
        }
        macro = float(np.mean(f1)) if len(f1) else 0.0
        return MetricsReport(
            accuracy=float(accuracy_score(y_true, y_pred)),
            macro_f1=macro,
            per_class=per_class,
            confusion=self.confusion_matrix(y_true, y_pred),
            num_classes=self.num_classes,
        )

    def roc_auc_binary(self, y_true: list[int], scores: list[float]) -> float:
        raise NotImplementedError("TODO(implementation): Metrics.roc_auc_binary")


def _labels(num_classes: int) -> list[int]:
    return list(range(num_classes))


__all__ = [
    "BehaviorClass",
    "ClassScores",
    "Metrics",
    "MetricsReport",
]
