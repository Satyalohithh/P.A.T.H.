"""Stream lifecycle API (register/start/stop streams)."""

from __future__ import annotations

from safewatch.core.schemas.stream import VideoSourceConfig


class StreamsRouter:
    """REST endpoints for stream management."""

    def list_streams(self) -> list[dict[str, object]]:
        raise NotImplementedError("TODO(implementation): StreamsRouter.list_streams")

    def register_stream(self, source: VideoSourceConfig) -> dict[str, object]:
        raise NotImplementedError("TODO(implementation): StreamsRouter.register_stream")

    def stop_stream(self, stream_id: str) -> None:
        raise NotImplementedError("TODO(implementation): StreamsRouter.stop_stream")


__all__ = ["StreamsRouter", "VideoSourceConfig"]
