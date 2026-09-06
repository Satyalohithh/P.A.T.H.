"""Pose estimation stage interface.

Concrete implementations (e.g. :class:`YOLOPoseEstimator`) run a single
full-frame keypoint inference and associate the resulting instances with the
current :class:`Track` list, producing one :class:`PoseRecord` per matched
person. Only project model types cross this boundary.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from safewatch.core.schemas.pose import PoseRecord
from safewatch.core.schemas.tracking import Track
from safewatch.core.types import Frame, TimePoint


class PoseEstimator(ABC):
    """Interface for per-frame body-keypoint estimation over tracks."""

    @abstractmethod
    def estimate(
        self,
        frame: Frame,
        track: Track,
        *,
        timestamp: TimePoint = 0.0,
    ) -> PoseRecord | None:
        """Estimate the pose of ``track``'s region in ``frame``.

        Returns ``None`` when no pose instance matches the track box.
        """

    @abstractmethod
    def batch_estimate(
        self,
        frame: Frame,
        tracks: list[Track],
        *,
        timestamp: TimePoint = 0.0,
    ) -> list[PoseRecord]:
        """Estimate poses for every matched track in ``frame``.

        Runs a single inference per frame; each output record carries a stable
        ``track_id`` matching a member of ``tracks``.
        """

    @abstractmethod
    def warmup(self) -> None:
        """Load weights / allocate device buffers before steady-state use."""

    @property
    @abstractmethod
    def latency_ms(self) -> float:
        """Rolling median inference latency for the estimator."""


__all__ = ["PoseEstimator"]
