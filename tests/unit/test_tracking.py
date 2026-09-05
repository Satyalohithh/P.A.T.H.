"""Integration tests driving the real Ultralytics ByteTrack backend."""

from __future__ import annotations

import pytest

from safewatch.core.config import TrackingConfig
from safewatch.core.schemas.detection import BBox, Detection
from safewatch.tracking import ByteTrackTracker, Tracker


def _person(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    *,
    conf: float,
    ts: float,
    did: int,
) -> Detection:
    return Detection(
        id=did,
        bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
        confidence=conf,
        class_id=0,
        class_name="person",
        timestamp=ts,
    )


def _tracker(**overrides) -> ByteTrackTracker:
    config = TrackingConfig(**overrides)
    tracker = ByteTrackTracker(config)
    tracker.initialize()
    return tracker


def _assert_box_close(
    track, xmin: float, ymin: float, xmax: float, ymax: float
) -> None:
    tolerance = 8.0
    assert abs(track.bbox.xmin - xmin) <= tolerance
    assert abs(track.bbox.ymin - ymin) <= tolerance
    assert abs(track.bbox.xmax - xmax) <= tolerance
    assert abs(track.bbox.ymax - ymax) <= tolerance


def test_tracker_is_protocol_compliant() -> None:
    tracker = ByteTrackTracker(TrackingConfig())
    assert isinstance(tracker, Tracker)
    assert all(
        callable(getattr(tracker, name)) for name in ("initialize", "update", "reset")
    )


def test_empty_updates_return_no_tracks() -> None:
    tracker = _tracker()
    assert tracker.update([]) == []
    assert tracker.update([]) == []


def test_single_person_keeps_stable_track_across_frames() -> None:
    tracker = _tracker()
    seen_hits: list[int] = []
    for frame in range(1, 12):
        app = tracker.update([_person(10, 10, 60, 60, conf=0.9, ts=frame, did=0)])
        if frame == 1:
            assert app == [], "first hit is not yet confirmed"
            continue
        assert len(app) == 1
        track = app[0]
        assert track.track_id == 1
        assert track.is_confirmed is True
        assert track.age == frame
        seen_hits.append(track.hits)
        assert track.confidence == pytest.approx(0.9, abs=1e-3)
        _assert_box_close(track, 10, 10, 60, 60)
    assert seen_hits == list(range(1, 11))


def test_two_people_keep_distinct_stable_ids() -> None:
    tracker = _tracker()
    by_frame: list[set[int]] = []
    left_box = (10.0, 10.0, 60.0, 60.0)
    right_box = (120.0, 10.0, 170.0, 60.0)
    for frame in range(1, 9):
        tracks = tracker.update(
            [
                _person(*left_box, conf=0.9, ts=frame, did=0),
                _person(*right_box, conf=0.95, ts=frame, did=1),
            ]
        )
        if frame == 1:
            assert tracks == [], "first hits are not yet confirmed"
            continue
        assert len(tracks) == 2
        by_frame.append({t.track_id for t in tracks})
        for track in tracks:
            if track.bbox.xmin < 100:
                _assert_box_close(track, *left_box)
            else:
                _assert_box_close(track, *right_box)
    assert all(ids == {1, 2} for ids in by_frame)


def test_lost_track_is_recovered_with_same_id() -> None:
    tracker = _tracker()
    outputs: list[list[int]] = []
    for frame in range(1, 4):
        tracks = tracker.update([_person(10, 10, 60, 60, conf=0.9, ts=frame, did=0)])
        outputs.append([t.track_id for t in tracks])
    for frame in range(4, 10):
        assert tracker.update([]) == []
    for frame in range(10, 13):
        tracks = tracker.update([_person(10, 10, 60, 60, conf=0.9, ts=frame, did=0)])
        outputs.append([t.track_id for t in tracks])

    assert outputs[0] == []
    assert outputs[1:3] == [[1], [1]]
    assert outputs[-3:] == [[], [1], [1]], (
        "recovered track re-emits same id after re-confirmation"
    )


def test_reset_starts_a_fresh_track_sequence() -> None:
    tracker = _tracker()
    first = tracker.update([_person(10, 10, 60, 60, conf=0.9, ts=1.0, did=0)])
    assert first == []

    tracker.reset()

    reappear = tracker.update([_person(10, 10, 60, 60, conf=0.9, ts=2.0, did=0)])
    assert reappear == []
    second = tracker.update([_person(10, 10, 60, 60, conf=0.9, ts=3.0, did=0)])
    assert [t.track_id for t in second] == [1]
    assert second[0].hits == 1


def test_timestamp_is_copied_from_matched_detection() -> None:
    tracker = _tracker()
    last_timestamp: float | None = None
    for frame in range(1, 4):
        tracks = tracker.update(
            [_person(10, 10, 60, 60, conf=0.9, ts=float(frame), did=0)]
        )
        if tracks:
            last_timestamp = tracks[0].timestamp
    assert last_timestamp == pytest.approx(3.0)


def test_min_hits_gates_track_emission() -> None:
    tracker = _tracker(min_hits=2)
    outputs: list[int] = []
    for frame in range(1, 6):
        tracks = tracker.update(
            [_person(10, 10, 60, 60, conf=0.9, ts=float(frame), did=0)]
        )
        outputs.append(len(tracks))
    assert outputs == [0, 0, 1, 1, 1]


def test_low_confidence_detections_can_recover_existing_track() -> None:
    tracker = _tracker()
    for frame in range(1, 4):
        tracker.update([_person(10, 10, 60, 60, conf=0.9, ts=float(frame), did=0)])
    for frame in range(4, 7):
        tracks = tracker.update(
            [_person(12, 10, 62, 60, conf=0.3, ts=float(frame), did=0)]
        )
        assert [t.track_id for t in tracks] == [1], (
            "existing track kept via low-score stage"
        )


def test_tracking_error_wraps_backend_failure() -> None:
    from safewatch.core.exceptions import TrackingError

    def boom(_args) -> None:
        raise RuntimeError("kalman exploded")

    tracker = ByteTrackTracker(TrackingConfig(), tracker_factory=boom)
    with pytest.raises(TrackingError, match="initialize"):
        tracker.initialize()


def test_detection_order_has_no_effect_on_identity() -> None:
    tracker = _tracker()
    boxes = [
        (10.0, 10.0, 60.0, 60.0),
        (120.0, 10.0, 170.0, 60.0),
        (60.0, 90.0, 110.0, 140.0),
    ]
    original = {}
    for frame in range(1, 5):
        rotated = boxes[(frame - 1) % len(boxes) :] + boxes[: (frame - 1) % len(boxes)]
        detections = [
            _person(*bbox, conf=0.9, ts=float(frame), did=index)
            for index, bbox in enumerate(rotated)
        ]
        tracks = tracker.update(detections)
        if frame == 1:
            assert tracks == []
            continue
        assert len(tracks) == 3
        original = {t.track_id: (t.bbox.xmin, t.bbox.ymin) for t in tracks}
    assert len(original) == 3
