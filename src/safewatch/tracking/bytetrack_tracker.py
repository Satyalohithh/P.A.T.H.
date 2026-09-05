"""ByteTrack-based multi-object tracking.

:class:`ByteTrackTracker` wraps Ultralytics' maintained ``BYTETracker`` and
adapts it to the project's domain models. All ByteTrack/Ultralytics types are
sealed inside this module: the tracker's public surface only accepts
:class:`Detection` and produces :class:`Track`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np

from safewatch.core.config import TrackingConfig
from safewatch.core.exceptions import TrackingError
from safewatch.core.schemas.detection import BBox, Detection, DetectionList
from safewatch.core.schemas.tracking import Track, TrackState
from safewatch.core.types import FrameIndex, TimePoint
from safewatch.tracking.interfaces import Tracker
from safewatch.utils.logging import get_logger

PERSON_CLASS_NAME = "person"
"""Low-score matching threshold for ByteTrack's second association stage."""
LOW_SCORE_THRESHOLD = 0.1

TrackerArgs: TypeAlias = Any
"""Ultralytics-compatible args namespace consumed by the ByteTrack backend."""

TrackerBackend: TypeAlias = Any
"""Instances of the Ultralytics BYTETracker backend."""


@dataclass(frozen=True, slots=True)
class _TrackerArgs:
    """Parameter namespace matching what ``BYTETracker`` reads from ``args``."""

    track_high_thresh: float
    track_low_thresh: float
    new_track_thresh: float
    track_buffer: int
    match_thresh: float
    fuse_score: bool = True


class _DetectionBatch:
    """Results-like carrier expected by ``BYTETracker.update``.

    Exposes ``conf``, ``xywh``, and ``cls`` as numpy arrays plus boolean-mask
    indexing, mirroring the small surface Ultralytics reads from detections.
    """

    __slots__ = ("_cls", "_conf", "_xywh")

    def __init__(
        self,
        conf: np.ndarray,
        xywh: np.ndarray,
        cls: np.ndarray,
    ) -> None:
        self._conf = conf
        self._xywh = xywh
        self._cls = cls

    @property
    def conf(self) -> np.ndarray:
        return self._conf

    @property
    def xywh(self) -> np.ndarray:
        return self._xywh

    @property
    def cls(self) -> np.ndarray:
        return self._cls

    def __len__(self) -> int:
        return len(self._conf)

    def __getitem__(self, mask: np.ndarray) -> _DetectionBatch:
        return _DetectionBatch(
            conf=self._conf[mask],
            xywh=self._xywh[mask],
            cls=self._cls[mask],
        )


class ByteTrackTracker(Tracker):
    """Person tracking through the Ultralytics ByteTrack backend."""

    def __init__(
        self,
        config: TrackingConfig,
        *,
        logger: logging.Logger | None = None,
        tracker_factory: Callable[[_TrackerArgs], TrackerBackend] | None = None,
        person_class_id: int = 0,
    ) -> None:
        self._config = config
        self._logger = logger or get_logger("safewatch.tracking.bytetrack")
        self._tracker_factory = tracker_factory or self._default_factory
        self._person_class_id = person_class_id
        self._tracker: TrackerBackend | None = None
        self._batch_detections: list[Detection] = []

    # -- Tracker -------------------------------------------------------------

    def initialize(self) -> None:
        if self._tracker is not None:
            return
        try:
            self._tracker = self._tracker_factory(self._build_args())
        except Exception as exc:
            self._tracker = None
            raise TrackingError(
                f"failed to initialize ByteTrack tracker: {exc}"
            ) from exc
        self._logger.info(
            "tracker initialized",
            extra={"backend": "bytetrack", "config": self._config.model_dump()},
        )

    def update(self, detections: DetectionList) -> list[Track]:
        tracker = self._require_tracker()
        batch = _batch_from_detections(
            detections=detections,
            person_class_name=PERSON_CLASS_NAME,
            person_class_id=self._person_class_id,
        )
        self._batch_detections = _person_detections(detections)
        try:
            tracker.update(batch)
        except Exception as exc:
            raise TrackingError(f"ByteTrack update failed: {exc}") from exc
        return self._to_tracks(tracker)

    def reset(self) -> None:
        self._batch_detections = []
        if self._tracker is not None:
            self._tracker.reset()
        self._logger.info("tracker reset", extra={"backend": "bytetrack"})

    # -- internals -----------------------------------------------------------

    def _build_args(self) -> _TrackerArgs:
        config = self._config
        return _TrackerArgs(
            track_high_thresh=config.track_thresh,
            track_low_thresh=LOW_SCORE_THRESHOLD,
            new_track_thresh=config.track_thresh,
            track_buffer=config.track_buffer,
            match_thresh=config.match_thresh,
        )

    def _default_factory(self, args: _TrackerArgs) -> TrackerBackend:
        try:
            from ultralytics.trackers import BYTETracker
        except ImportError as exc:  # pragma: no cover
            raise TrackingError(
                "ultralytics is not installed; run `uv sync --group dev`"
            ) from exc
        return BYTETracker(args)

    def _require_tracker(self) -> TrackerBackend:
        self.initialize()
        tracker = self._tracker
        if tracker is None:
            raise TrackingError("tracker is not initialized")
        return tracker

    def _to_tracks(self, tracker: TrackerBackend) -> list[Track]:
        tracks: list[Track] = []
        for st in getattr(tracker, "tracked_stracks", ()):
            if not getattr(st, "is_activated", False):
                continue
            track = self._to_track(st)
            if track.hits >= self._config.min_hits:
                tracks.append(track)
        tracks.sort(key=lambda track: track.track_id)
        return tracks

    def _to_track(self, strack: Any) -> Track:
        xyxy = np.asarray(strack.xyxy, dtype=float)
        xmin, ymin, xmax, ymax = xyxy.tolist()
        confidence = float(getattr(strack, "score", self._config.track_thresh))
        return Track(
            track_id=int(strack.track_id),
            state=TrackState.ACTIVE,
            first_frame=FrameIndex(getattr(strack, "start_frame", 0)),
            last_frame=FrameIndex(getattr(strack, "frame_id", 0)),
            bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
            confidence=confidence,
            timestamp=self._timestamp_for(strack),
            age=_track_age(strack),
            hits=int(getattr(strack, "tracklet_len", 0)),
        )

    def _timestamp_for(self, strack: Any) -> TimePoint:
        idx = getattr(strack, "idx", None)
        if idx is None:
            return 0.0
        try:
            index = int(idx)
        except (TypeError, ValueError):
            return 0.0
        if 0 <= index < len(self._batch_detections):
            return self._batch_detections[index].timestamp
        return 0.0


def _track_age(strack: Any) -> int:
    """Frames observed since the track was first registered."""

    return max(
        0,
        int(getattr(strack, "frame_id", 0))
        - int(getattr(strack, "start_frame", 0))
        + 1,
    )


def _person_detections(detections: DetectionList) -> list[Detection]:
    """Return detection rows whose class is ``person``."""

    return [
        detection
        for detection in detections
        if detection.class_name == PERSON_CLASS_NAME
    ]


def _batch_from_detections(
    detections: DetectionList,
    *,
    person_class_name: str,
    person_class_id: int,
) -> _DetectionBatch:
    """Convert project detections into the backend's results-like input."""

    persons = [
        detection
        for detection in detections
        if detection.class_name == person_class_name
    ]
    count = len(persons)
    conf = np.zeros((count,), dtype=np.float32)
    xywh = np.zeros((count, 4), dtype=np.float32)
    cls = np.full((count,), person_class_id, dtype=np.float32)
    for index, detection in enumerate(persons):
        bbox = detection.bbox
        width = bbox.width
        height = bbox.height
        cx, cy = bbox.center
        conf[index] = detection.confidence
        xywh[index] = (cx, cy, width, height)
    return _DetectionBatch(conf=conf, xywh=xywh, cls=cls)


__all__ = ["PERSON_CLASS_NAME", "ByteTrackTracker"]
