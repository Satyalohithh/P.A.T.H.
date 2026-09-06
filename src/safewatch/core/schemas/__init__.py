"""Typed schemas for SafeWatch AI."""

from __future__ import annotations

from safewatch.core.schemas.alert import Alert, AlertEvent, AlertReply
from safewatch.core.schemas.behavior import ClassificationResult, ClassificationRow
from safewatch.core.schemas.dataset import Annotation, DatasetRecord
from safewatch.core.schemas.detection import BBox, Detection, DetectionFrameView
from safewatch.core.schemas.features import (
    FeatureVector,
    FeatureVectorRow,
    WindowSummary,
)
from safewatch.core.schemas.pose import PoseResult, PoseRow
from safewatch.core.schemas.risk import RiskAssessment, RiskExplanation
from safewatch.core.schemas.stream import FrameMetadata, VideoSourceConfig
from safewatch.core.schemas.tracking import Track, TrackAssignment, TrackState

__all__ = [
    "Alert",
    "AlertEvent",
    "AlertReply",
    "Annotation",
    "BBox",
    "ClassificationResult",
    "ClassificationRow",
    "DatasetRecord",
    "Detection",
    "DetectionFrameView",
    "FeatureVector",
    "FeatureVectorRow",
    "FrameMetadata",
    "PoseResult",
    "PoseRow",
    "RiskAssessment",
    "RiskExplanation",
    "Track",
    "TrackAssignment",
    "TrackState",
    "VideoSourceConfig",
    "WindowSummary",
]
