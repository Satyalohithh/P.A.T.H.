"""Frame source runtime interface.

A :class:`FrameSource` is the runtime abstraction that produces frames from a
video input. It is distinct from its immutable configuration counterpart
:class:`safewatch.core.schemas.stream.VideoSourceConfig`.
"""

from __future__ import annotations

from typing import Protocol, Self, runtime_checkable

from safewatch.core.schemas.stream import FrameMetadata
from safewatch.core.types import Frame


@runtime_checkable
class FrameSource(Protocol):
    """Protocol implemented by concrete frame sources (webcam, video file)."""

    @property
    def stream_id(self) -> str:
        """Stable identity of this source, carried into every frame's metadata."""

        ...

    @property
    def is_open(self) -> bool:
        """Whether the underlying capture is currently open."""

        ...

    def open(self) -> None:
        """Open the underlying capture.

        Must be idempotent: calling ``open`` on an already-open source is a
        no-op. Raises a typed source-specific error when opening fails.
        """

        ...

    def close(self) -> None:
        """Release the underlying capture and mark the source closed.

        Must be idempotent and safe to call multiple times.
        """

        ...

    def read(self) -> tuple[FrameMetadata, Frame] | None:
        """Read the next frame.

        Returns ``(metadata, frame)`` on success, or ``None`` at end-of-stream
        / when the capture cannot be read. Metadata carries the frame index
        and stream timestamp.
        """

        ...

    def __enter__(self) -> Self:
        """Open the source and return self for use as a context manager."""

        ...

    def __exit__(self, *exc: object) -> None:
        """Close the source on context exit."""

        ...


__all__ = ["FrameSource"]
