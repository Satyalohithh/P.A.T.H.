"""Risk-assessment data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from safewatch.core.constants import RiskLevel
from safewatch.core.types import FrameIndex, StreamId, TrackId


@dataclass(frozen=True, slots=True)
class RiskAssessment:
    """Risk score plus explanation for one interaction window."""

    score: float
    level: RiskLevel
    contributing_factors: tuple[tuple[str, float], ...] = ()
    stream_id: StreamId = "cam-00"
    frame_index: FrameIndex = 0
    track_id: TrackId = -1

    def __post_init__(self) -> None:
        if not isinstance(self.level, RiskLevel):
            try:
                normalized = RiskLevel(self.level)
            except (TypeError, ValueError):
                raise ValueError(f"Invalid risk level: {self.level!r}")
            object.__setattr__(self, "level", normalized)


@dataclass(frozen=True, slots=True)
class RiskExplanation:
    """Human-readable rationale used by the explainability module."""

    assessment: RiskAssessment
    top_factors: tuple[str, ...] = ()


class RiskEvent(TypedDict):
    """Serializable risk event for the alert engine."""

    stream_id: str
    frame_index: int
    track_id: int
    score: float
    level: int


__all__ = ["RiskAssessment", "RiskEvent", "RiskExplanation"]
