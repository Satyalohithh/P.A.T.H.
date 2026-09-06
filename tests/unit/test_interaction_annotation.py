"""Tests for the InteractionAnnotator pair overlays."""

from __future__ import annotations

import numpy as np

from safewatch.core.schemas.pose import PoseRecord
from safewatch.features.behavioral_features import FeatureEngine
from safewatch.utils.annotation import InteractionAnnotator
from tests.unit._feature_helpers import person_pose


def _blank() -> np.ndarray:
    return np.zeros((400, 400, 3), dtype=np.uint8)


def _pair() -> tuple[PoseRecord, PoseRecord]:
    return person_pose(1, 100.0, 100.0), person_pose(2, 300.0, 100.0)


def _drawn_pixels(annotated: np.ndarray) -> int:
    return int(np.count_nonzero(annotated))


class TestInteractionAnnotator:
    def test_draw_distance_leaves_input_unchanged(self) -> None:
        first, second = _pair()
        annotator = InteractionAnnotator()
        frame = _blank()
        result = annotator.draw_distance(frame, first, second, distance_h=1.7)
        assert result is not frame
        assert result.shape == frame.shape
        assert _drawn_pixels(result) > 0
        assert _drawn_pixels(frame) == 0

    def test_draw_approach_vectors(self) -> None:
        first, second = _pair()
        annotator = InteractionAnnotator()
        result = annotator.draw_approach_vectors(
            _blank(), first, second, approach_from_first=900.0, approach_from_second=0.0
        )
        assert _drawn_pixels(result) > 0

    def test_draw_reciprocity(self) -> None:
        first, second = _pair()
        annotator = InteractionAnnotator()
        result = annotator.draw_reciprocity(_blank(), first, second, reciprocity=0.93)
        assert _drawn_pixels(result) > 0

    def test_missing_keypoints_are_noop(self) -> None:
        _, second = _pair()
        low = person_pose(3, 50.0, 50.0, conf=0.1)
        annotator = InteractionAnnotator(min_confidence=0.45)
        assert _drawn_pixels(annotator.draw_distance(_blank(), low, second, 1.0)) == 0
        assert (
            _drawn_pixels(
                annotator.draw_approach_vectors(_blank(), low, second, 1.0, 0.0)
            )
            == 0
        )
        assert (
            _drawn_pixels(annotator.draw_reciprocity(_blank(), low, second, 1.0)) == 0
        )

    def test_empty_pair_avoids_zero_length_arrow(self) -> None:
        annotator = InteractionAnnotator()
        first, second = _pair()
        result = annotator.draw_approach_vectors(
            _blank(), first, second, approach_from_first=0.0, approach_from_second=0.0
        )
        assert result is not None
        assert result.shape == (400, 400, 3)


class TestEngineToAnnotator:
    def test_pair_values_pipeline_into_overlays(self) -> None:
        engine = FeatureEngine(smoothing_sigma=0.0, fps=30.0)
        annotator = InteractionAnnotator()
        frame = _blank()
        for frame_index in range(3):
            records = engine.process(
                [
                    person_pose(1, 100.0 + 5.0 * frame_index, 100.0),
                    person_pose(2, 300.0 - 5.0 * frame_index, 100.0),
                ],
                frame_index=frame_index,
            )
        pair = next(r for r in records if r.pair_track_id is not None)
        values = dict(zip(pair.names, pair.values))
        first = person_pose(1, 100.0 + 5.0 * 2, 100.0)
        second = person_pose(2, 300.0 - 5.0 * 2, 100.0)
        result = annotator.draw_distance(
            annotator.draw_approach_vectors(
                frame,
                first,
                second,
                approach_from_first=values["inter.distance"],
                approach_from_second=values["inter.distance"],
            ),
            first,
            second,
            distance_h=values["inter.distance"],
        )
        assert _drawn_pixels(result) > 0


class TestPublicSurface:
    def test_annotators_exported(self) -> None:
        from safewatch.utils.annotation import __all__ as exports

        assert "InteractionAnnotator" in exports
        assert "PoseAnnotator" in exports
