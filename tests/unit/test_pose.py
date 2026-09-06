"""Tests for the pose schema, layout validation, associator, and estimator."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from safewatch.core.config import PoseConfig
from safewatch.core.exceptions import PoseEstimationError
from safewatch.core.schemas.detection import BBox
from safewatch.core.schemas.pose import PoseRecord
from safewatch.core.schemas.tracking import Track, TrackState
from safewatch.core.types import Keypoints
from safewatch.pose.associator import (
    DEFAULT_MATCH_THRESHOLD,
    PoseAssociator,
    PoseCandidate,
)
from safewatch.pose.keypoint_layout import (
    COCO_17_KEYPOINT_NAMES,
    COCO_17_SKELETON,
    KEYPOINT_COUNT,
    validate_keypoints,
)
from safewatch.pose.yolo_pose_estimator import (
    KEYPOINTS_PER_INSTANCE,
    YOLOPoseEstimator,
)


def make_keypoints(confidence: float = 0.9, offset: float = 0.0) -> Keypoints:
    """17 COCO-17 keypoints laid out in a small diagonal line."""

    return tuple(
        (10.0 + offset + index * 3.0, 20.0 + index * 2.0, confidence)
        for index in range(17)
    )


def make_track(
    track_id: int, xmin: float, ymin: float, xmax: float, ymax: float
) -> Track:
    return Track(
        track_id=track_id,
        state=TrackState.ACTIVE,
        first_frame=1,
        last_frame=1,
        bbox=BBox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax),
        confidence=0.9,
        timestamp=1.0,
        age=1,
        hits=1,
    )


class TestPoseRecord:
    def test_confidences_reflect_keypoint_triplets(self) -> None:
        keypoints = make_keypoints(confidence=0.8)
        record = PoseRecord(
            track_id=3, timestamp=1.5, keypoints=keypoints, confidence=0.7
        )
        assert record.confidences == (0.8,) * 17
        assert record.confidence == 0.7

    def test_valid_keypoint_count_respects_gate(self) -> None:
        keypoints = ((0.0, 0.0, 0.9),) * 10 + ((0.0, 0.0, 0.2),) * 7
        record = PoseRecord(
            track_id=1, timestamp=0.0, keypoints=keypoints, confidence=0.5
        )
        assert record.valid_keypoint_count(gate=0.5) == 10
        assert record.valid_keypoint_count(gate=0.0) == 17

    def test_keypoint_returns_triplet_or_none(self) -> None:
        keypoints = ((10.0, 5.0, 0.9),) + ((0.0, 0.0, 0.1),) * 16
        record = PoseRecord(
            track_id=1, timestamp=0.0, keypoints=keypoints, confidence=0.5
        )
        assert record.keypoint(0) == (10.0, 5.0, 0.9)
        assert record.keypoint(0, min_confidence=0.95) is None
        assert record.keypoint(999) is None

    def test_is_frozen(self) -> None:
        record = PoseRecord(
            track_id=1, timestamp=0.0, keypoints=make_keypoints(), confidence=0.5
        )
        with pytest.raises(AttributeError):
            record.track_id = 2


class TestKeypointLayout:
    def test_layout_is_coco_17(self) -> None:
        assert KEYPOINT_COUNT == 17
        assert len(COCO_17_KEYPOINT_NAMES) == 17
        assert COCO_17_KEYPOINT_NAMES[0] == "nose"
        assert COCO_17_KEYPOINT_NAMES[16] == "right_ankle"
        for first, second in COCO_17_SKELETON:
            assert 0 <= first < 17 and 0 <= second < 17
            assert first != second

    def test_validate_accepts_valid_keypoints(self) -> None:
        validate_keypoints(make_keypoints())

    def test_validate_rejects_wrong_count(self) -> None:
        with pytest.raises(PoseEstimationError, match="17"):
            validate_keypoints(make_keypoints()[:5])

    def test_validate_rejects_negative_confidence(self) -> None:
        bad = tuple((0.0, 0.0, -0.1) for _ in range(17))
        with pytest.raises(PoseEstimationError, match="conf"):
            validate_keypoints(bad)

    def test_validate_rejects_non_triplet(self) -> None:
        bad = ((0.0, 0.0, 0.5, 1.0),) + ((0.0, 0.0, 0.5),) * 16
        with pytest.raises(PoseEstimationError, match="triplet"):
            validate_keypoints(bad)


class TestPoseAssociator:
    def test_associates_by_iou_and_preserves_track_order(self) -> None:
        tracks = [make_track(1, 0, 0, 50, 50), make_track(2, 80, 80, 130, 130)]
        candidates = [
            PoseCandidate(
                bbox=BBox(85, 85, 125, 125), keypoints=make_keypoints(), confidence=0.8
            ),
            PoseCandidate(
                bbox=BBox(5, 5, 45, 45), keypoints=make_keypoints(), confidence=0.6
            ),
        ]
        records = PoseAssociator().associate(tracks, candidates, timestamp=7.0)
        assert [record.track_id for record in records] == [1, 2]
        assert records[0].confidence == 0.6
        assert records[0].timestamp == 7.0

    def test_one_to_one_never_duplicates_candidates(self) -> None:
        tracks = [
            make_track(1, 0, 0, 50, 50),
            make_track(2, 10, 10, 40, 40),
            make_track(3, 100, 100, 150, 150),
        ]
        candidates = [
            PoseCandidate(
                bbox=BBox(0, 0, 50, 50), keypoints=make_keypoints(), confidence=0.9
            ),
            PoseCandidate(
                bbox=BBox(100, 100, 150, 150),
                keypoints=make_keypoints(),
                confidence=0.7,
            ),
        ]
        records = PoseAssociator().associate(tracks, candidates, timestamp=0.0)
        assert len(records) == 2
        assert {record.track_id for record in records} == {1, 3}

    def test_below_threshold_omits_track(self) -> None:
        tracks = [make_track(1, 0, 0, 50, 50)]
        candidates = [
            PoseCandidate(
                bbox=BBox(200, 200, 250, 250),
                keypoints=make_keypoints(),
                confidence=0.9,
            )
        ]
        records = PoseAssociator().associate(tracks, candidates, timestamp=0.0)
        assert records == []

    def test_empty_inputs_return_empty(self) -> None:
        assert (
            PoseAssociator().associate(
                [],
                [PoseCandidate(BBox(0, 0, 1, 1), make_keypoints(), 0.5)],
                timestamp=0.0,
            )
            == []
        )
        assert (
            PoseAssociator().associate([make_track(1, 0, 0, 50, 50)], [], timestamp=0.0)
            == []
        )

    def test_default_threshold(self) -> None:
        assert DEFAULT_MATCH_THRESHOLD == 0.5

    def test_rejects_invalid_threshold(self) -> None:
        with pytest.raises(ValueError):
            PoseAssociator(match_threshold=1.5)

    def test_malformed_candidate_raises(self) -> None:
        tracks = [make_track(1, 0, 0, 50, 50)]
        candidates = [
            PoseCandidate(
                bbox=BBox(5, 5, 45, 45), keypoints=make_keypoints()[:3], confidence=0.5
            )
        ]
        with pytest.raises(PoseEstimationError):
            PoseAssociator().associate(tracks, candidates, timestamp=0.0)


class _T:
    """Renderable array that answers .cpu().numpy() like ultralytics tensors."""

    def __init__(self, arr: np.ndarray) -> None:
        self._arr = arr

    def cpu(self) -> _T:
        return self

    def numpy(self) -> np.ndarray:
        return self._arr


class _FakeKeypoints:
    def __init__(self, data: np.ndarray | None) -> None:
        self.data = _T(data) if data is not None else None


class _FakeBoxes:
    def __init__(self, xyxy: list, conf: list) -> None:
        self.xyxy = _T(np.asarray(xyxy, dtype=float))
        self.conf = _T(np.asarray(conf, dtype=float))


class _FakeResult:
    def __init__(
        self, boxes: _FakeBoxes | None, keypoints: _FakeKeypoints | None
    ) -> None:
        self.boxes = boxes
        self.keypoints = keypoints

    def __len__(self) -> int:
        return 1 if self.boxes is not None and self.keypoints is not None else 0


class _FakeModel:
    def __init__(
        self,
        results: list[_FakeResult] | None = None,
        fail: bool = False,
        bad_count: int | None = None,
    ) -> None:
        self.results = results or []
        self.fail = fail
        self.bad_count = bad_count
        self.calls: list[dict[str, Any]] = []

    def __call__(self, frame: Any, **kwargs: Any) -> list[_FakeResult]:
        self.calls.append(kwargs)
        if self.fail:
            raise RuntimeError("inference boom")
        if self.bad_count is not None:
            boxes = _FakeBoxes(xyxy=[[0, 0, 50, 50]], conf=[0.8])
            data = np.tile([0.0, 0.0, 0.9], (self.bad_count, 1)).reshape(
                1, self.bad_count, 3
            )
            return [_FakeResult(boxes, _FakeKeypoints(data))]
        return self.results


def _make_estimator(
    results: list[_FakeResult] | None = None,
    config: PoseConfig | None = None,
    fail: bool = False,
    bad_count: int | None = None,
) -> tuple[YOLOPoseEstimator, _FakeModel]:
    model = _FakeModel(results, fail=fail, bad_count=bad_count)
    estimator = YOLOPoseEstimator(
        config or PoseConfig(),
        model_factory=lambda weights: model,
    )
    return estimator, model


def _pose_result(offset: float, count: int = 1) -> _FakeResult:
    boxes = _FakeBoxes(
        xyxy=[
            [
                offset + i * 100,
                offset + i * 100,
                offset + i * 100 + 50,
                offset + i * 100 + 50,
            ]
            for i in range(count)
        ],
        conf=[0.7] * count,
    )
    data = np.zeros((count, KEYPOINTS_PER_INSTANCE, 3), dtype=float)
    for instance in range(count):
        for kp in range(KEYPOINTS_PER_INSTANCE):
            data[instance, kp] = [offset + instance * 100 + kp, 10.0, 0.9]
    return _FakeResult(boxes, _FakeKeypoints(data))


class TestYoloPoseEstimator:
    def test_batch_estimate_maps_instances_to_tracks(self) -> None:
        estimator, model = _make_estimator([_pose_result(0.0)])
        tracks = [make_track(1, 0, 0, 50, 50)]
        records = estimator.batch_estimate(
            np.zeros((100, 100, 3), np.uint8), tracks, timestamp=2.0
        )
        assert len(records) == 1
        assert records[0].track_id == 1
        assert records[0].timestamp == 2.0
        assert records[0].confidence == 0.7
        assert len(model.calls) == 1
        assert model.calls[0]["conf"] == PoseConfig().conf_threshold
        assert model.calls[0]["device"] == "cpu"

    def test_multiple_people_get_distinct_tracks(self) -> None:
        estimator, _ = _make_estimator([_pose_result(0.0, count=2)])
        tracks = [make_track(1, 0, 0, 50, 50), make_track(2, 100, 100, 150, 150)]
        records = estimator.batch_estimate(np.zeros((200, 200, 3), np.uint8), tracks)
        assert [record.track_id for record in records] == [1, 2]

    def test_empty_tracks_token_inference(self) -> None:
        estimator, model = _make_estimator([_pose_result(0.0)])
        assert estimator.batch_estimate(np.zeros((100, 100, 3), np.uint8), []) == []
        assert model.calls == []

    def test_empty_results_return_empty(self) -> None:
        estimator, _ = _make_estimator([])
        tracks = [make_track(1, 0, 0, 50, 50)]
        assert estimator.batch_estimate(np.zeros((100, 100, 3), np.uint8), tracks) == []

    def test_missing_boxes_or_keypoints_returns_empty(self) -> None:
        estimator, _ = _make_estimator([_FakeResult(None, None)])
        tracks = [make_track(1, 0, 0, 50, 50)]
        assert estimator.batch_estimate(np.zeros((100, 100, 3), np.uint8), tracks) == []

    def test_estimate_single_track_helper(self) -> None:
        estimator, _ = _make_estimator([_pose_result(0.0)])
        record = estimator.estimate(
            np.zeros((100, 100, 3), np.uint8), make_track(5, 0, 0, 50, 50)
        )
        assert record is not None
        assert record.track_id == 5

    def test_wrong_keypoint_count_raises(self) -> None:
        estimator, _ = _make_estimator(bad_count=7)
        tracks = [make_track(1, 0, 0, 50, 50)]
        with pytest.raises(PoseEstimationError, match="17"):
            estimator.batch_estimate(np.zeros((100, 100, 3), np.uint8), tracks)

    def test_inference_error_raises_pose_error(self) -> None:
        estimator, _ = _make_estimator(fail=True)
        tracks = [make_track(1, 0, 0, 50, 50)]
        with pytest.raises(PoseEstimationError):
            estimator.batch_estimate(np.zeros((100, 100, 3), np.uint8), tracks)

    def test_model_load_failure_raises_pose_error(self) -> None:
        def boom(weights: str) -> Any:
            raise RuntimeError("no weights")

        estimator = YOLOPoseEstimator(PoseConfig(), model_factory=boom)
        with pytest.raises(PoseEstimationError):
            estimator._get_model()

    def test_latency_ms_rolling_median(self) -> None:
        estimator, _ = _make_estimator([_pose_result(0.0)])
        assert estimator.latency_ms == 0.0
        estimator.batch_estimate(
            np.zeros((100, 100, 3), np.uint8), [make_track(1, 0, 0, 50, 50)]
        )
        assert estimator.latency_ms >= 0.0

    def test_warmup_runs_inference(self) -> None:
        estimator, model = _make_estimator([_pose_result(0.0)])
        estimator.warmup()
        assert len(model.calls) == 1

    def test_weights_respected(self) -> None:
        config = PoseConfig(weights="custom.pt", conf_threshold=0.4, device="cpu")
        model = _FakeModel([_pose_result(0.0)])
        estimator = YOLOPoseEstimator(config, model_factory=lambda weights: model)
        estimator.batch_estimate(
            np.zeros((100, 100, 3), np.uint8), [make_track(1, 0, 0, 50, 50)]
        )
        assert model.calls[0]["conf"] == 0.4
