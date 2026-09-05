"""Confusion-matrix analysis."""

from __future__ import annotations

from collections.abc import Sequence

from safewatch.core.constants import BehaviorClass


class ConfusionAnalysis:
    """Builds and summarizes the confusion matrix."""

    def __init__(self, num_classes: int = len(BehaviorClass)) -> None:
        self.num_classes = num_classes

    def matrix(self, y_true: Sequence[int], y_pred: Sequence[int]) -> list[list[int]]:
        raise NotImplementedError("TODO(implementation): ConfusionAnalysis.matrix")

    def worst_offdiagonal(self, matrix: list[list[int]]) -> tuple[int, int]:
        """Return the (true, predicted) pair with the most confusions."""

        raise NotImplementedError(
            "TODO(implementation): ConfusionAnalysis.worst_offdiagonal"
        )


__all__ = ["BehaviorClass", "ConfusionAnalysis"]
