"""Stream-ingestion data contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from safewatch.core.types import FrameIndex, StreamId, TimePoint


class VideoSourceType(StrEnum):
    """Kind of video source feeding the pipeline."""

    RTSP = "rtsp"
    RTMP = "rtmp"
    FILE = "file"
    WEBCAM = "webcam"


@dataclass(frozen=True, slots=True)
class VideoSource:
    """Descriptor of one input video stream."""

    stream_id: StreamId
    uri: str
    source_type: VideoSourceType = VideoSourceType.FILE
    nominal_fps: float = 30.0


@dataclass(frozen=True, slots=True)
class FrameMetadata:
    """Per-frame metadata common to every pipeline stage."""

    stream_id: StreamId
    frame_index: FrameIndex
    timestamp: TimePoint
    width: int = 0
    height: int = 0


__all__ = ["FrameMetadata", "VideoSource", "VideoSourceType"]
