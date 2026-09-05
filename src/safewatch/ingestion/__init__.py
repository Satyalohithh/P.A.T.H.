"""Frame ingestion: source interfaces, decoders, frame buffering, and lifecycle."""

from __future__ import annotations

from safewatch.core.schemas.stream import VideoSourceConfig
from safewatch.ingestion.decoder import VideoDecoder
from safewatch.ingestion.frame_buffer import FrameBuffer, FrameBufferFullError
from safewatch.ingestion.interfaces import FrameSource
from safewatch.ingestion.stream_manager import StreamManager
from safewatch.ingestion.video_source import VideoFileSource, WebcamSource

__all__ = [
    "FrameBuffer",
    "FrameBufferFullError",
    "FrameSource",
    "StreamManager",
    "VideoDecoder",
    "VideoFileSource",
    "VideoSourceConfig",
    "WebcamSource",
]
