"""Core types, enums, constants, exceptions, configuration and pipeline contracts."""

from __future__ import annotations

from safewatch.core.constants import (
    AlertStatus,
    BehaviorClass,
    Keypoint,
    RiskLevel,
)
from safewatch.core.exceptions import (
    AlertDispatchError,
    ClassificationError,
    DataValidationError,
    DetectionError,
    FeatureExtractionError,
    ModelRegistryError,
    PoseEstimationError,
    RiskEvaluationError,
    SafeWatchConfigError,
    SafeWatchError,
    SafeWatchPipelineError,
    StorageError,
    TrackingError,
)
from safewatch.core.types import (
    Confidence,
    FrameIndex,
    Keypoints,
    StreamId,
    TimePoint,
    TrackId,
    Vec2D,
)
from safewatch.core.types import (
    Keypoint as KeypointTuple,
)

__all__ = [
    "AlertDispatchError",
    "AlertStatus",
    "BehaviorClass",
    "ClassificationError",
    "Confidence",
    "DataValidationError",
    "DetectionError",
    "FeatureExtractionError",
    "FrameIndex",
    "Keypoint",
    "KeypointTuple",
    "Keypoints",
    "ModelRegistryError",
    "PoseEstimationError",
    "RiskEvaluationError",
    "RiskLevel",
    "SafeWatchConfigError",
    "SafeWatchError",
    "SafeWatchPipelineError",
    "StorageError",
    "StreamId",
    "TimePoint",
    "TrackId",
    "TrackingError",
    "Vec2D",
]
