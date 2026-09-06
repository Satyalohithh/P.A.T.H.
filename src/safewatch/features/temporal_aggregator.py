"""Sliding-window temporal aggregation of per-frame feature vectors."""

from __future__ import annotations

import math
from collections import deque

from safewatch.core.schemas.features import FeatureVector, WindowSummary
from safewatch.core.types import TrackId


class TemporalAggregator:
    """Rolls per-frame feature vectors into interaction windows of fixed depth.

    Pooling is column-wise: :meth:`window` returns the per-column mean over the
    window (skipping NaN), and :meth:`summary` gives overall mean/std/min/max
    over the finite entries. Features polluted with NaN do not bias the pools.
    """

    def __init__(self, window_frames: int = 30) -> None:
        self.window_frames = max(1, window_frames)
        self._windows: dict[TrackId, deque[FeatureVector]] = {}

    def push(self, vector: FeatureVector) -> None:
        window = self._windows.setdefault(vector.track_id, deque())
        window.append(vector)
        while len(window) > self.window_frames:
            window.popleft()

    def window(self, track_id: TrackId) -> FeatureVector:
        """Aggregate the current window for ``track_id`` (per-column mean)."""

        entries = list(self._windows.get(track_id, ()))
        if not entries:
            return FeatureVector(names=(), values=())
        names = entries[-1].names
        pooled: list[float] = []
        for index in range(len(names)):
            column = [
                entry.values[index]
                for entry in entries
                if index < len(entry.names)
                and entry.names == names
                and math.isfinite(entry.values[index])
            ]
            pooled.append(sum(column) / len(column) if column else float("nan"))
        return FeatureVector(
            names=names,
            values=tuple(pooled),
            frame_index=max(entry.frame_index for entry in entries),
            track_id=track_id,
        )

    def summary(self, track_id: TrackId) -> WindowSummary:
        entries = list(self._windows.get(track_id, ()))
        frames = [entry.frame_index for entry in entries]
        if not entries:
            return WindowSummary(
                track_id=track_id,
                window_start=0,
                window_end=0,
                mean=float("nan"),
                std=float("nan"),
                min=float("nan"),
                max=float("nan"),
            )
        values = [
            value for entry in entries for value in entry.values if math.isfinite(value)
        ]
        mean = sum(values) / len(values) if values else float("nan")
        if values:
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = math.sqrt(max(0.0, variance))
        else:
            std = float("nan")
        return WindowSummary(
            track_id=track_id,
            window_start=min(frames),
            window_end=max(frames),
            mean=mean,
            std=std,
            min=min(values) if values else float("nan"),
            max=max(values) if values else float("nan"),
        )

    def reset(self) -> None:
        self._windows.clear()


__all__ = ["FeatureVector", "TemporalAggregator", "WindowSummary"]
