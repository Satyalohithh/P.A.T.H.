"""Video read/write helpers wrapping the decoder and cv2 writers."""

from __future__ import annotations

from safewatch.core.schemas.stream import VideoSourceConfig
from safewatch.ingestion.decoder import VideoDecoder


class VideoIO:
    """Open/close helpers and output containers for processed streams."""

    def __init__(self) -> None:
        self._decoders: list[VideoDecoder] = []

    def open_source(self, source: VideoSourceConfig) -> VideoDecoder:
        decoder = VideoDecoder(source)
        self._decoders.append(decoder)
        return decoder

    def close_source(self, decoder: VideoDecoder) -> None:
        self._decoders = [
            existing for existing in self._decoders if existing is not decoder
        ]

    def write_annotated_clip(self, output_path: str, fps: float) -> None:
        raise NotImplementedError("TODO(implementation): VideoIO.write_annotated_clip")


__all__ = ["VideoIO", "VideoSourceConfig"]
