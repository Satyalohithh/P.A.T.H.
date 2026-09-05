"""Track state bookkeeping (identity, age, confirmation, removal)."""

from __future__ import annotations

from safewatch.core.schemas.tracking import Track, TrackState
from safewatch.core.types import FrameIndex, TrackId


class TrackManager:
    """Owns the track table and applies lifecycle transitions."""

    def __init__(self, max_age: int = 30, min_hits: int = 3) -> None:
        self.max_age = max_age
        self.min_hits = min_hits

    def get(self, track_id: TrackId) -> Track | None:
        raise NotImplementedError("TODO(implementation): TrackManager.get")

    def upsert(self, track: Track) -> None:
        raise NotImplementedError("TODO(implementation): TrackManager.upsert")

    def transition(self, track_id: TrackId, state: TrackState) -> None:
        raise NotImplementedError("TODO(implementation): TrackManager.transition")

    def prune(self, now_frame: FrameIndex) -> tuple[Track, ...]:
        """Delete tracks unseen for ``max_age``; return terminated tracks."""

        raise NotImplementedError("TODO(implementation): TrackManager.prune")


__all__ = ["Track", "TrackManager", "TrackState"]
