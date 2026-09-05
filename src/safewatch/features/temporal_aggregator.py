"""Sliding-window temporal aggregation of per-frame feature vectors."""

from __future__ import annotations

from safewatch.core.schemas.features import FeatureVector, WindowSummary
from safewatch.core.types import TrackId


class TemporalAggregator:
    """Rolls per-frame feature vectors into interaction windows."""

    def __init__(self, window_frames: int = 30) -> None:
        self.window_frames = window_frames

    def push(self, vector: FeatureVector) -> None:
        raise NotImplementedError("TODO(implementation): TemporalAggregator.push")

    def window(self, track_id: TrackId) -> FeatureVector:
        """Aggregate the current window for ``track_id`` (mean/std pooling)."""

        raise NotImplementedError("TODO(implementation): TemporalAggregator.window")

    def summary(self, track_id: TrackId) -> WindowSummary:
        raise NotImplementedError("TODO(implementation): TemporalAggregator.summary")

    def reset(self) -> None:
        raise NotImplementedError("TODO(implementation): TemporalAggregator.reset")


__all__ = ["FeatureVector", "TemporalAggregator", "WindowSummary"]
