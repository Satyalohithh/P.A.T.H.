"""Calibration analysis (reliability diagrams)."""

from __future__ import annotations

from safewatch.core.schemas.behavior import ClassificationResult


class CalibrationAnalysis:
    """Evaluates probability calibration per class."""

    def __init__(self, bins: int = 10) -> None:
        self.bins = bins

    def expected_calibration_error(
        self, results: list[ClassificationResult], y_true: list[int]
    ) -> float:
        raise NotImplementedError(
            "TODO(implementation): CalibrationAnalysis.expected_calibration_error"
        )


__all__ = ["CalibrationAnalysis", "ClassificationResult"]
