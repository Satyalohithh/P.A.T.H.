"""Tracker runtime interface.

A :class:`Tracker` associates detections across frames into stable tracks.
Concrete implementations (e.g. :class:`ByteTrackTracker`) must never leak
their backend's types; the public surface is exclusively project models.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from safewatch.core.schemas.detection import DetectionList
from safewatch.core.schemas.tracking import Track


@runtime_checkable
class Tracker(Protocol):
    """Protocol implemented by concrete multi-object trackers."""

    def initialize(self) -> None:
        """Allocate backend state before the first update.

        Must be idempotent: calling ``initialize`` more than once is a no-op.
        """

        ...

    def update(self, detections: DetectionList) -> list[Track]:
        """Advance tracking one frame with ``detections`` and return the
        confirmed/active tracks for that frame.

        Track IDs are stable across frames for the same person.
        """

        ...

    def reset(self) -> None:
        """Forget all tracks and restart identity assignment from scratch."""

        ...


__all__ = ["Tracker"]
