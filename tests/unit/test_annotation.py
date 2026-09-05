"""Tests for FrameAnnotator drawing on frame copies."""

from __future__ import annotations

import numpy as np

from safewatch.core.schemas.detection import BBox, Detection
from safewatch.utils.annotation import FrameAnnotator


def _frame(width: int = 64, height: int = 48) -> np.ndarray:
    return np.zeros((height, width, 3), dtype=np.uint8)


def test_annotate_returns_same_shape() -> None:
    frame = _frame()
    annotator = FrameAnnotator()
    detections = [
        Detection(
            id=0,
            bbox=BBox(xmin=5, ymin=5, xmax=20, ymax=15),
            confidence=0.9,
            class_id=0,
            class_name="person",
        )
    ]
    out = annotator.annotate(frame, detections)
    assert out.shape == frame.shape
    assert out.dtype == frame.dtype


def test_annotate_does_not_mutate_input() -> None:
    frame = _frame()
    original = frame.copy()
    annotator = FrameAnnotator()
    detections = [
        Detection(
            id=0,
            bbox=BBox(xmin=5, ymin=5, xmax=20, ymax=15),
            confidence=0.9,
            class_id=0,
            class_name="person",
        )
    ]
    annotator.annotate(frame, detections)
    assert np.array_equal(frame, original)


def test_annotate_draws_pixels_on_output() -> None:
    frame = _frame()
    annotator = FrameAnnotator()
    detections = [
        Detection(
            id=0,
            bbox=BBox(xmin=5, ymin=5, xmax=20, ymax=15),
            confidence=0.9,
            class_id=0,
            class_name="person",
        )
    ]
    out = annotator.annotate(frame, detections)
    assert not np.array_equal(out, frame)


def test_annotate_empty_detections_is_identity_copy() -> None:
    frame = _frame()
    out = FrameAnnotator().annotate(frame, [])
    assert np.array_equal(out, frame)
    assert out is not frame
