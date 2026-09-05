"""Unit tests for the ByteTrack adapter (pure mapping, no backend)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from safewatch.core.config import TrackingConfig
from safewatch.core.schemas.detection import BBox, Detection
from safewatch.core.schemas.tracking import TrackState
from safewatch.core.types import FrameIndex
from safewatch.tracking.bytetrack_tracker import (
    ByteTrackTracker,
    _batch_from_detections,
    _person_detections,
    _track_age,
)


def _person(
    xmin: float, ymin: float, xmax: float, ymax: float, conf: float
) -> Detection:
    return Detection(
        id=0,
        bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
        confidence=conf,
        class_id=0,
        class_name="person",
        timestamp=1.0,
    )


class TestBatchFromDetections:
    def test_converts_xyxy_to_xywh_and_filters_persons(self) -> None:
        persons = [
            _person(10, 20, 50, 60, 0.9),  # w40 h40 center (30, 40)
            _person(100, 100, 140, 140, 0.7),  # w40 h40 center (120, 120)
        ]
        other = Detection(
            id=2,
            bbox=BBox(0, 0, 10, 10),
            confidence=0.9,
            class_id=2,
            class_name="car",
        )
        batch = _batch_from_detections(
            detections=[*persons, other],
            person_class_name="person",
            person_class_id=0,
        )

        assert len(batch) == 2
        np.testing.assert_allclose(batch.conf, [0.9, 0.7])
        np.testing.assert_allclose(batch.xywh[0], [30, 40, 40, 40])
        np.testing.assert_allclose(batch.xywh[1], [120, 120, 40, 40])
        np.testing.assert_allclose(batch.cls, [0, 0])

    def test_empty_detections_yield_empty_2d_batch(self) -> None:
        batch = _batch_from_detections(
            detections=[],
            person_class_name="person",
            person_class_id=0,
        )
        assert len(batch) == 0
        assert batch.xywh.shape == (0, 4)
        assert batch.conf.shape == (0,)

    def test_mask_indexing_retains_slice(self) -> None:
        dets = [_person(10, 20, 50, 60, 0.9), _person(100, 100, 140, 140, 0.7)]
        batch = _batch_from_detections(
            dets, person_class_name="person", person_class_id=0
        )
        mask = np.asarray([True, False])
        sliced = batch[mask]
        np.testing.assert_allclose(sliced.xywh, batch.xywh[mask])


class TestPersonDetections:
    def test_filters_by_class_name(self) -> None:
        dets = [
            _person(10, 20, 50, 60, 0.9),
            Detection(
                id=1,
                bbox=BBox(0, 0, 10, 10),
                confidence=0.9,
                class_id=2,
                class_name="car",
            ),
        ]
        assert [d.class_name for d in _person_detections(dets)] == ["person"]


class TestAgeMapping:
    def test_age_is_frame_span(self) -> None:
        strack = SimpleNamespace(frame_id=7, start_frame=5)
        assert _track_age(strack) == 3

    def test_age_clamped_positive(self) -> None:
        strack = SimpleNamespace(frame_id=2, start_frame=9)
        assert _track_age(strack) == 0


class TestToTracksWithFakeBackend:
    def test_activated_track_maps_to_domain_track(self) -> None:
        strack = SimpleNamespace(
            track_id=3,
            is_activated=True,
            xyxy=np.asarray([10.0, 20.0, 50.0, 60.0]),
            score=0.85,
            start_frame=4,
            frame_id=6,
            tracklet_len=2,
            idx=1,
        )
        tracker = ByteTrackTracker(TrackingConfig())
        tracker._batch_detections = [
            _person(0, 0, 10, 10, 0.5),
            _person(10, 20, 50, 60, 0.85),
        ]
        [track] = tracker._to_tracks(SimpleNamespace(tracked_stracks=[strack]))

        assert track.track_id == 3
        assert track.state == TrackState.ACTIVE
        assert track.first_frame == FrameIndex(4)
        assert track.last_frame == FrameIndex(6)
        assert (track.bbox.xmin, track.bbox.ymin, track.bbox.xmax, track.bbox.ymax) == (
            10.0,
            20.0,
            50.0,
            60.0,
        )
        assert track.confidence == pytest.approx(0.85)
        assert track.timestamp == pytest.approx(1.0)
        assert track.age == 3
        assert track.hits == 2
        assert track.is_confirmed is True

    def test_unactivated_track_is_dropped(self) -> None:
        strack = SimpleNamespace(
            track_id=3,
            is_activated=False,
            xyxy=np.asarray([0.0, 0.0, 10.0, 10.0]),
            score=0.9,
            start_frame=1,
            frame_id=1,
            tracklet_len=0,
        )
        tracker = ByteTrackTracker(TrackingConfig())
        tracks = tracker._to_tracks(SimpleNamespace(tracked_stracks=[strack]))
        assert tracks == []

    def test_min_hits_filters_low_tracklet_len(self) -> None:
        strack = SimpleNamespace(
            track_id=1,
            is_activated=True,
            xyxy=np.asarray([0.0, 0.0, 10.0, 10.0]),
            score=0.9,
            start_frame=1,
            frame_id=3,
            tracklet_len=2,
        )
        tracker = ByteTrackTracker(TrackingConfig(min_hits=3))
        tracks = tracker._to_tracks(SimpleNamespace(tracked_stracks=[strack]))
        assert tracks == []

        strack.tracklet_len = 3
        [track] = tracker._to_tracks(SimpleNamespace(tracked_stracks=[strack]))
        assert track.hits == 3

    def test_timestamp_falls_back_when_idx_out_of_range(self) -> None:
        strack = SimpleNamespace(
            track_id=1,
            is_activated=True,
            xyxy=np.asarray([0.0, 0.0, 10.0, 10.0]),
            score=0.9,
            start_frame=1,
            frame_id=1,
            tracklet_len=3,
            idx=99,
        )
        tracker = ByteTrackTracker(TrackingConfig())
        tracker._batch_detections = []
        [track] = tracker._to_tracks(SimpleNamespace(tracked_stracks=[strack]))
        assert track.timestamp == 0.0

    def test_backend_args_come_from_config(self) -> None:
        tracker = ByteTrackTracker(
            TrackingConfig(track_thresh=0.6, match_thresh=0.7, track_buffer=45)
        )
        args = tracker._build_args()
        assert args.track_high_thresh == 0.6
        assert args.new_track_thresh == 0.6
        assert args.track_low_thresh == 0.1
        assert args.match_thresh == 0.7
        assert args.track_buffer == 45
        assert args.fuse_score is True
