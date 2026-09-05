"""Verifies every declared stub raises the canonical NotImplementedError."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest

from safewatch.alerts.alert_manager import AlertManager
from safewatch.alerts.cooldown import CooldownTracker
from safewatch.alerts.dispatcher import AlertDispatcher
from safewatch.behavior.classifier import BehaviorClassifier
from safewatch.core.config import ConfigLoader
from safewatch.core.pipeline import Pipeline, PipelineContext
from safewatch.core.schemas.detection import BBox
from safewatch.core.schemas.pose import PoseResult
from safewatch.detection.person_detector import PersonDetector
from safewatch.features.interaction_features import InteractionFeatureExtractor
from safewatch.features.motion_features import MotionFeatureExtractor
from safewatch.features.pose_features import PoseFeatureExtractor
from safewatch.ingestion.decoder import VideoDecoder
from safewatch.ingestion.frame_buffer import FrameBuffer
from safewatch.pose.pose_estimator import PoseEstimator
from safewatch.risk.risk_scorer import RiskScorer
from safewatch.tracking.association import Association
from safewatch.tracking.multi_object_tracker import MultiObjectTracker

_CANONICAL = "TODO(implementation)"

STUB_CALLS: dict[str, Callable[[], Any]] = {
    "BBox.iou": lambda: BBox(0, 0, 10, 10).iou(BBox(0, 0, 5, 5)),
    "PoseResult.keypoint": lambda: PoseResult(
        track_id=1, keypoints=(), confidences=()
    ).keypoint(0),
    "ConfigLoader.load": lambda: ConfigLoader().load("x"),
    "Pipeline.add_stage": lambda: Pipeline(stages=[]).add_stage(object()),
    "Pipeline.run": lambda: Pipeline(stages=[]).run(
        PipelineContext(settings=object()), None
    ),
    "FrameBuffer.push": lambda: FrameBuffer(max_size=1).push(object(), object()),
    "VideoDecoder.open": lambda: VideoDecoder(object()).open(),
    "PersonDetector.detect": lambda: PersonDetector("weights.pt").detect(object()),
    "MultiObjectTracker.update": lambda: MultiObjectTracker().update(0, []),
    "Association.match": lambda: Association().match([], []),
    "PoseEstimator.estimate": lambda: PoseEstimator().estimate(object(), object()),
    "PoseFeatureExtractor.extract": lambda: PoseFeatureExtractor().extract([]),
    "MotionFeatureExtractor.extract": lambda: MotionFeatureExtractor().extract([]),
    "InteractionFeatureExtractor.extract": lambda: (
        InteractionFeatureExtractor().extract([])
    ),
    "BehaviorClassifier.predict": lambda: BehaviorClassifier("model.pt").predict(
        object()
    ),
    "RiskScorer.score": lambda: RiskScorer().score(object()),
    "CooldownTracker.in_cooldown": lambda: CooldownTracker({"r": 3.0}).in_cooldown(
        "r", "cam"
    ),
    "AlertManager.create": lambda: AlertManager().create("cam", object(), "m"),
    "AlertDispatcher.dispatch": lambda: AlertDispatcher().dispatch(object()),
}


@pytest.mark.parametrize("name", list(STUB_CALLS))
def test_stub_raises_canonical_not_implemented(name: str) -> None:
    with pytest.raises(NotImplementedError) as exc_info:
        STUB_CALLS[name]()
    assert _CANONICAL in str(exc_info.value)
