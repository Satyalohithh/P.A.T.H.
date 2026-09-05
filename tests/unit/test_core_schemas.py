"""Behavioral tests for the frozen data contracts and model invariants."""

from __future__ import annotations

import pytest

from safewatch.core.constants import AlertStatus, BehaviorClass
from safewatch.core.schemas.alert import Alert, AlertEvent
from safewatch.core.schemas.behavior import ClassificationResult
from safewatch.core.schemas.detection import BBox
from safewatch.core.schemas.features import FeatureVector
from safewatch.core.schemas.pose import PoseResult
from safewatch.core.schemas.risk import RiskAssessment


class TestFeatureVector:
    def test_equal_lengths_ok(self) -> None:
        vector = FeatureVector(names=("a", "b"), values=(1.0, 2.0))
        assert vector.dimension == 2

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            FeatureVector(names=("a", "b"), values=(1.0,))


class TestClassificationResult:
    def test_probabilities_match_class_count(self) -> None:
        result = ClassificationResult(
            label=BehaviorClass.NORMAL,
            probabilities=(0.7, 0.2, 0.05, 0.05),
        )
        assert result.label is BehaviorClass.NORMAL

    def test_probability_mismatch_raises(self) -> None:
        with pytest.raises(ValueError):
            ClassificationResult(label=BehaviorClass.NORMAL, probabilities=(0.5,))


class TestBBox:
    def test_width_height_center(self) -> None:
        box = BBox(xmin=0.0, ymin=0.0, xmax=10.0, ymax=20.0)
        assert box.width == 10.0
        assert box.height == 20.0
        assert box.center == (5.0, 10.0)

    def test_iou_is_stub(self) -> None:
        with pytest.raises(NotImplementedError):
            BBox(0, 0, 10, 10).iou(BBox(0, 0, 5, 5))


class TestPoseResult:
    def test_valid_keypoint_count_filters_confidences(self) -> None:
        pose = PoseResult(
            track_id=1,
            keypoints=tuple((i, i, 0.5) for i in range(5)),
            confidences=(0.5, 0.4, 0.9, 0.44, 0.46),
        )
        assert pose.valid_keypoint_count == 3

    def test_keypoint_lookup_is_stub(self, sample_pose: PoseResult) -> None:
        with pytest.raises(NotImplementedError):
            sample_pose.keypoint(0)


class TestAlert:
    def test_create_defaults_to_active(self) -> None:
        risk = RiskAssessment(score=0.9, level=2)
        alert = Alert.create(stream_id="cam-01", risk=risk, message="x")
        assert alert.status is AlertStatus.ACTIVE
        assert alert.id

    def test_alert_event_serializable(self) -> None:
        risk = RiskAssessment(score=0.9, level=2)
        alert = Alert.create(stream_id="cam-01", risk=risk, message="x")
        event = AlertEvent(alert)
        assert event["risk"]["score"] == 0.9
        assert event["status"] == "active"
