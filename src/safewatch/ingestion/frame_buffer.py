"""Bounded frame buffer decoupling ingestion from downstream processing."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from safewatch.core.schemas.stream import FrameMetadata
from safewatch.ingestion.decoder import Frame


class FrameBufferFullError(Exception):
    """Raised when a bounded buffer rejects a push."""


@dataclass
class FrameBuffer:
    """Thread-safe bounded buffer with drop-oldest semantics."""

    max_size: int = 16
    drop_oldest: bool = True
    _buffer: deque[tuple[FrameMetadata, Frame]] = field(
        default_factory=deque, init=False, repr=False
    )

    def push(self, metadata: FrameMetadata, frame: Frame) -> None:
        raise NotImplementedError("TODO(implementation): FrameBuffer.push")

    def pop(self) -> tuple[FrameMetadata, Frame] | None:
        raise NotImplementedError("TODO(implementation): FrameBuffer.pop")

    def clear(self) -> None:
        raise NotImplementedError("TODO(implementation): FrameBuffer.clear")

    @property
    def size(self) -> int:
        return len(self._buffer)


__all__ = ["FrameBuffer", "FrameBufferFullError"]
