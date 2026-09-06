"""Tests for PoseAnnotator drawing helpers."""

from __future__ import annotations

import numpy as np

from safewatch.core.schemas.pose import PoseRecord
from safewatch.core.types import Keypoints
from safewatch.utils.annotation import PoseAnnotator


def make_keypoints(diagonal_value: float = 0.9) -> Keypoints:
    """17 COCO-17 keypoints rising diagonally across a 100x100 frame."""

    return tuple(
        (5.0 + index, 5.0 + index * 5.0, diagonal_value) for index in range(17)
    )


def _blank() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _drawn_pixels(annotated: np.ndarray) -> int:
    return int(np.count_nonzero(annotated))


class TestPoseAnnotator:
    def test_draws_keypoints_and_returns_new_array(self) -> None:
        annotator = PoseAnnotator()
        frame = _blank()
        result = annotator.draw_keypoints(frame, make_keypoints())
        assert result is not frame
        assert result.shape == frame.shape
        assert _drawn_pixels(result) > 0
        assert _drawn_pixels(frame) == 0

    def test_draw_skeleton_connects_bones(self) -> None:
        annotator = PoseAnnotator()
        frame = _blank()
        result = annotator.draw_skeleton(frame, make_keypoints())
        assert result.shape == frame.shape
        assert _drawn_pixels(result) > 0

    def test_draw_confidence_labels_each_keypoint(self) -> None:
        annotator = PoseAnnotator()
        result = annotator.draw_confidence(_blank(), make_keypoints())
        assert _drawn_pixels(result) > 0

    def test_low_confidence_keypoints_are_skipped(self) -> None:
        annotator = PoseAnnotator(min_confidence=0.5)
        low = make_keypoints(diagonal_value=0.2)
        assert _drawn_pixels(annotator.draw_keypoints(_blank(), low)) == 0
        assert _drawn_pixels(annotator.draw_skeleton(_blank(), low)) == 0

    def test_annotate_leaves_input_unchanged(self) -> None:
        annotator = PoseAnnotator()
        frame = _blank()
        record = PoseRecord(
            track_id=4,
            timestamp=1.0,
            keypoints=make_keypoints(),
            confidence=0.6,
        )
        result = annotator.annotate(frame, [record])
        assert result is not frame
        assert _drawn_pixels(result) > 0
        assert _drawn_pixels(frame) == 0

    def test_annotate_with_no_records_is_copy(self) -> None:
        annotator = PoseAnnotator()
        frame = _blank()
        result = annotator.annotate(frame, [])
        assert result is not frame
        assert np.array_equal(result, frame)

    def test_annotation_missing_keypoints_is_noop(self) -> None:
        annotator = PoseAnnotator()
        assert _drawn_pixels(annotator.draw_skeleton(_blank(), ())) == 0
        assert _drawn_pixels(annotator.draw_keypoints(_blank(), ())) == 0

    def test_min_confidence_defaults(self) -> None:
        assert PoseAnnotator()._min_confidence == 0.45

    def test_keypoint_visibility_matrix_is_binary(self) -> None:
        annotator = PoseAnnotator()
        keypoints_binary = [(0.0, 0.0, 1.0)] * 17
        result = annotator.draw_skeleton(_blank(), keypoints_binary)
        assert _drawn_pixels(result) > 0
