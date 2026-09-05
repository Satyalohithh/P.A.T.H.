"""Tests for webcam/file frame sources (mock cv2, no real capture)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from safewatch.core.exceptions import SourceError
from safewatch.ingestion.video_source import VideoFileSource, WebcamSource


class _MockCapture:
    """Fake cv2.VideoCapture readable once, then end-of-stream."""

    def __init__(self, opened: bool = True, frames: int = 1) -> None:
        self._opened = opened
        self._remaining = frames
        self._released = False

    def isOpened(self) -> bool:
        return self._opened and not self._released

    def release(self) -> None:
        self._released = True

    def read(self) -> tuple[bool, Any]:
        if not self.isOpened() or self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((48, 64, 3), dtype=np.uint8)


@pytest.fixture
def mock_video_capture(monkeypatch: pytest.MonkeyPatch) -> Iterator[_MockCapture]:
    capture = _MockCapture()

    class _Factory:
        def __call__(self, *args: Any) -> _MockCapture:
            return capture

    monkeypatch.setattr(
        "cv2.VideoCapture",
        _Factory(),  # type: ignore[attr-defined]
    )
    yield capture


def test_video_file_opens_and_reads(mock_video_capture: _MockCapture) -> None:
    with VideoFileSource(path="sample.mp4") as source:
        assert source.is_open
        result = source.read()
        assert result is not None
        metadata, frame = result
        assert metadata.stream_id == "file:sample.mp4"
        assert metadata.frame_index == 0
        assert frame.shape == (48, 64, 3)
        assert source.read() is None

    assert not source.is_open


def test_webcam_source_identifier(mock_video_capture: _MockCapture) -> None:
    source = WebcamSource(device_index=0)
    assert source.stream_id == "webcam:0"
    source.close()


def test_open_is_idempotent(mock_video_capture: _MockCapture) -> None:
    source = VideoFileSource(path="sample.mp4")
    source.open()
    source.open()  # no-op, must not raise
    source.close()
    source.close()
    assert not source.is_open


def test_read_without_open_raises_source_error() -> None:
    source = VideoFileSource(path="sample.mp4")
    with pytest.raises(SourceError):
        source.read()


def test_failed_open_raises_source_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ClosedFactory:
        def __call__(self, *args: Any) -> _MockCapture:
            return _MockCapture(opened=False)

    monkeypatch.setattr(
        "cv2.VideoCapture",
        _ClosedFactory(),  # type: ignore[attr-defined]
    )
    with pytest.raises(SourceError):
        VideoFileSource(path="sample.mp4").open()


def test_file_source_config_type() -> None:
    source = VideoFileSource(path="sample.mp4")
    assert source._config.source_type.value == "file"


def test_webcam_source_config_type() -> None:
    source = WebcamSource(device_index=3)
    assert source._config.source_type.value == "webcam"
