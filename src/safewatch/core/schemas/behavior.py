"""Behavior-classification data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from safewatch.core.constants import BehaviorClass
from safewatch.core.types import FrameIndex, TrackId


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Prediction for one interaction window."""

    label: BehaviorClass
    probabilities: tuple[float, ...]
    frame_index: FrameIndex = 0
    track_id: TrackId = -1

    def __post_init__(self) -> None:
        if len(self.probabilities) != len(BehaviorClass):
            raise ValueError(
                f"probabilities ({len(self.probabilities)}) must match the "
                f"number of behavior classes ({len(BehaviorClass)})"
            )


class ClassificationRow(TypedDict):
    """One row of the exported behavior-prediction dataset."""

    frame_index: int
    track_id: int
    label: int
    probabilities: list[float]


__all__ = ["ClassificationResult", "ClassificationRow"]
