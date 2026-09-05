"""Feature-extraction data contracts.

Feature identifiers follow the frozen catalog naming:

- pose:    ``pose.<name>``      (catalog group A)
- motion:  ``motion.<name>``    (catalog group B)
- inter:   ``inter.<name>``     (catalog group C)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from safewatch.core.types import FrameIndex, TrackId


@dataclass(frozen=True, slots=True)
class FeatureVector:
    """Aligned (names, values) pair for one track in one window."""

    names: tuple[str, ...]
    values: tuple[float, ...]
    frame_index: FrameIndex = 0
    track_id: TrackId = -1

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError(
                f"names ({len(self.names)}) and values ({len(self.values)}) "
                "must have equal length"
            )

    @property
    def dimension(self) -> int:
        return len(self.values)


class FeatureVectorRow(TypedDict):
    """One row of the exported feature dataset."""

    frame_index: int
    track_id: int
    names: list[str]
    values: list[float]


class WindowSummary(TypedDict):
    """Aggregate statistics over a sliding interaction window."""

    track_id: int
    window_start: int
    window_end: int
    mean: float
    std: float
    min: float
    max: float


__all__ = ["FeatureVector", "FeatureVectorRow", "WindowSummary"]
