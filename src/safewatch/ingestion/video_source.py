"""Concrete frame source implementations.

A shared :class:`_OpenCvSource` base wraps an ``cv2.VideoCapture`` so that
webcam and video-file sources share resource lifecycle, timing, and metadata
bookkeeping. Concrete classes provide the capture initializer and source
identity.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Self

import cv2
import numpy as np

from safewatch.core.exceptions import SourceError
from safewatch.core.schemas.stream import (
    FrameMetadata,
    VideoSourceConfig,
    VideoSourceType,
)
from safewatch.core.types import FrameIndex, TimePoint
from safewatch.ingestion.interfaces import FrameSource
from safewatch.utils.logging import get_logger


class _OpenCvSource(FrameSource, ABC):
    """Shared OpenCV-backed frame source with resource-safe lifecycle."""

    def __init__(
        self,
        config: VideoSourceConfig,
        logger: logging.Logger | None = None,
    ) -> None:
        self._config = config
        self._cap: cv2.VideoCapture | None = None
        self._frame_index: FrameIndex = -1
        self._started: TimePoint = 0.0
        self._logger = logger or get_logger(
            f"safewatch.ingestion.{type(self).__name__}"
        )

    # -- abstract hook for concrete sources ---------------------------------

    @abstractmethod
    def _open_capture(self) -> cv2.VideoCapture:
        """Build and open the underlying capture for this source."""

    @property
    def stream_id(self) -> str:
        return self._config.stream_id

    @property
    def is_open(self) -> bool:
        return self._cap is not None

    def open(self) -> None:
        if self.is_open:
            self._logger.debug("source already open; open() is a no-op")
            return
        try:
            self._cap = self._open_capture()
        except Exception as exc:
            self._cap = None
            raise SourceError(
                f"failed to open source {self.stream_id!r}: {exc}"
            ) from exc
        if not self._cap or not self._cap.isOpened():
            self._cap = None
            raise SourceError(f"failed to open source {self.stream_id!r}")
        self._frame_index = -1
        self._started = time.monotonic()
        self._logger.info("source opened", extra={"stream_id": self.stream_id})

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._logger.debug("source closed", extra={"stream_id": self.stream_id})

    def read(self) -> tuple[FrameMetadata, np.ndarray] | None:
        cap = self._require_capture()
        ok, frame = cap.read()
        if not ok:
            self._logger.info(
                "end of stream (read returned no frame)",
                extra={"stream_id": self.stream_id},
            )
            return None
        self._frame_index += 1
        height, width = frame.shape[:2]
        metadata = FrameMetadata(
            stream_id=self.stream_id,
            frame_index=self._frame_index,
            timestamp=self._elapsed_seconds(),
            width=width,
            height=height,
        )
        return metadata, frame

    def __enter__(self) -> Self:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- internals -----------------------------------------------------------

    def _require_capture(self) -> cv2.VideoCapture:
        if self._cap is None:
            raise SourceError(
                f"source {self.stream_id!r} is not open; call open() first"
            )
        return self._cap

    def _elapsed_seconds(self) -> TimePoint:
        return time.monotonic() - self._started


class VideoFileSource(_OpenCvSource):
    """Reads frames from a video file."""

    def __init__(
        self,
        path: str,
        stream_id: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        config = VideoSourceConfig(
            stream_id=stream_id or f"file:{path}",
            uri=path,
            source_type=VideoSourceType.FILE,
        )
        super().__init__(config=config, logger=logger)
        self._path = path

    def _open_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self._path)


class WebcamSource(_OpenCvSource):
    """Reads frames from a local webcam by device index."""

    def __init__(
        self,
        device_index: int = 0,
        stream_id: str | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        config = VideoSourceConfig(
            stream_id=stream_id or f"webcam:{device_index}",
            uri=str(device_index),
            source_type=VideoSourceType.WEBCAM,
        )
        super().__init__(config=config, logger=logger)
        self._device_index = device_index

    def _open_capture(self) -> cv2.VideoCapture:
        return cv2.VideoCapture(self._device_index)


__all__ = ["VideoFileSource", "WebcamSource"]
