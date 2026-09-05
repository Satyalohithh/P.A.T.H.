"""Feature dataset loading and window generation."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

from safewatch.core.schemas.behavior import ClassificationRow
from safewatch.core.schemas.features import FeatureVector


class FeatureDataLoader:
    """Yields (window, label) pairs from extracted feature stores."""

    def __init__(self, source_paths: Iterable[str], window_frames: int = 30) -> None:
        self.source_paths = tuple(source_paths)
        self.window_frames = window_frames

    def load(self) -> Iterator[tuple[FeatureVector, int]]:
        raise NotImplementedError("TODO(implementation): FeatureDataLoader.load")

    def save_rows(self, rows: Iterable[ClassificationRow]) -> None:
        raise NotImplementedError("TODO(implementation): FeatureDataLoader.save_rows")


__all__ = ["ClassificationRow", "FeatureDataLoader", "FeatureVector"]
