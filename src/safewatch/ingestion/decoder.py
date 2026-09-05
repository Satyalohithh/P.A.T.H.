"""Video decoding (OpenCV / ffmpeg based, declared but not yet implemented)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Self

from safewatch.core.schemas.stream import FrameMetadata, VideoSourceConfig
from safewatch.core.types import Frame, FrameIndex, TimePoint


class VideoDecoder:
    """Decodes frames from a :class:`VideoSourceConfig` at nominal frame rate."""

    def __init__(self, source: VideoSourceConfig) -> None:
        self.source = source

    def open(self) -> None:
        raise NotImplementedError("TODO(implementation): VideoDecoder.open")

    def close(self) -> None:
        raise NotImplementedError("TODO(implementation): VideoDecoder.close")

    def __enter__(self) -> Self:
        raise NotImplementedError("TODO(implementation): VideoDecoder.__enter__")

    def __exit__(self, *exc: object) -> None:
        raise NotImplementedError("TODO(implementation): VideoDecoder.__exit__")

    def frames(self) -> Iterator[tuple[FrameMetadata, Frame]]:
        """Yield (metadata, frame) pairs until end-of-stream."""

        raise NotImplementedError("TODO(implementation): VideoDecoder.frames")

    def seek(self, frame_index: FrameIndex) -> None:
        raise NotImplementedError("TODO(implementation): VideoDecoder.seek")

    def current_timestamp(self) -> TimePoint:
        """Monotonic or container timestamp of the last decoded frame."""

        raise NotImplementedError(
            "TODO(implementation): VideoDecoder.current_timestamp"
        )


__all__ = ["Frame", "VideoDecoder"]
