"""Probability calibration (isotonic / temperature scaling)."""

from __future__ import annotations

from safewatch.core.schemas.behavior import ClassificationResult


class Calibrator:
    """Post-hoc probability calibration for classifier outputs."""

    def __init__(self, method: str = "isotonic") -> None:
        self.method = method

    def fit(self, labels: list[int], scores: list[tuple[float, ...]]) -> None:
        raise NotImplementedError("TODO(implementation): Calibrator.fit")

    def transform(self, result: ClassificationResult) -> ClassificationResult:
        raise NotImplementedError("TODO(implementation): Calibrator.transform")


__all__ = ["Calibrator", "ClassificationResult"]
