"""Feature-extraction data contracts.

Feature identifiers follow the frozen catalog naming:

- pose:    ``pose.<name>``      (catalog group A)
- motion:  ``motion.<name>``    (catalog group B)
- inter:   ``inter.<name>``     (catalog group C)
- temporal: ``temporal.<name>`` (catalog group D; window-level)

The canonical ``FEATURE_GROUPS``/``FEATURE_NAMES`` column order in
``features/feature_names.py`` is frozen (A-C). Group-D ids such as
``temporal.reciprocity`` are carried inside :class:`FeatureRecord` only and
do not extend the canonical vector in this phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from safewatch.core.types import FrameIndex, TimePoint, TrackId


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


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    """Per-frame feature values for one track in a fixed name order.

    ``values`` entries may be NaN for features that cannot be computed (missing
    keypoints below the confidence gate, insufficient temporal context, ...);
    downstream aggregation skips NaN per the configured ``nan_policy``.
    ``pair_track_id`` is set for pair-level features (catalog C/D) so the same
    value is attributed to both interacting tracks. ``video_id`` identifies the
    source clip so window aggregation can scope rows to one video.
    """

    track_id: TrackId
    frame_index: FrameIndex
    timestamp: TimePoint
    names: tuple[str, ...]
    values: tuple[float, ...]
    pair_track_id: TrackId | None = None
    video_id: str | None = None

    def __post_init__(self) -> None:
        if len(self.names) != len(self.values):
            raise ValueError(
                f"names ({len(self.names)}) and values ({len(self.values)}) "
                "must have equal length"
            )


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


__all__ = ["FeatureRecord", "FeatureVector", "FeatureVectorRow", "WindowSummary"]
