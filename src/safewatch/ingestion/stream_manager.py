"""Lifecycle management for multiple concurrent video streams."""

from __future__ import annotations

from collections.abc import Iterable

from safewatch.core.schemas.stream import VideoSourceConfig
from safewatch.core.types import StreamId
from safewatch.ingestion.decoder import VideoDecoder
from safewatch.ingestion.frame_buffer import FrameBuffer


class StreamManager:
    """Attaches decoders and buffers to each registered stream."""

    def __init__(self, sources: Iterable[VideoSourceConfig]) -> None:
        self._sources = {source.stream_id: source for source in sources}
        self._decoders: dict[StreamId, VideoDecoder | None] = {
            sid: None for sid in self._sources
        }
        self._buffers: dict[StreamId, FrameBuffer | None] = {
            sid: None for sid in self._sources
        }

    def start(self, stream_id: StreamId) -> None:
        raise NotImplementedError("TODO(implementation): StreamManager.start")

    def stop(self, stream_id: StreamId) -> None:
        raise NotImplementedError("TODO(implementation): StreamManager.stop")

    def stop_all(self) -> None:
        raise NotImplementedError("TODO(implementation): StreamManager.stop_all")

    def buffer(self, stream_id: StreamId) -> FrameBuffer:
        raise NotImplementedError("TODO(implementation): StreamManager.buffer")

    @property
    def active_streams(self) -> tuple[StreamId, ...]:
        raise NotImplementedError("TODO(implementation): StreamManager.active_streams")


__all__ = ["FrameBuffer", "StreamManager", "VideoDecoder", "VideoSourceConfig"]
