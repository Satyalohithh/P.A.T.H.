"""Exception hierarchy for SafeWatch AI.

Every subsystem-specific error derives from a common ancestor so callers
can catch `SafeWatchError` as a safe default boundary.
"""

from __future__ import annotations


class SafeWatchError(Exception):
    """Base class for all SafeWatch AI errors."""


class SafeWatchConfigError(SafeWatchError):
    """Raised for invalid, missing, or contradictory configuration."""


class SafeWatchPipelineError(SafeWatchError):
    """Base class for failures inside the video-analysis pipeline."""


class DetectionError(SafeWatchPipelineError):
    """Person detection stage failure."""


class TrackingError(SafeWatchPipelineError):
    """Multi-object tracking stage failure."""


class PoseEstimationError(SafeWatchPipelineError):
    """Keypoint estimation stage failure."""


class FeatureExtractionError(SafeWatchPipelineError):
    """Feature extraction stage failure."""


class ClassificationError(SafeWatchPipelineError):
    """Behavior classification failure."""


class RiskEvaluationError(SafeWatchPipelineError):
    """Risk scoring failure."""


class AlertDispatchError(SafeWatchPipelineError):
    """Alert delivery failure (websocket, database, webhook)."""


class StorageError(SafeWatchError):
    """Persistence-layer failure (database, object storage)."""


class DataValidationError(SafeWatchError):
    """Raised when input data violates a declared schema/invariant."""


class ModelRegistryError(SafeWatchError):
    """Model loading / versioning failure."""


__all__ = [
    "AlertDispatchError",
    "ClassificationError",
    "DataValidationError",
    "DetectionError",
    "FeatureExtractionError",
    "ModelRegistryError",
    "PoseEstimationError",
    "RiskEvaluationError",
    "SafeWatchConfigError",
    "SafeWatchError",
    "SafeWatchPipelineError",
    "StorageError",
    "TrackingError",
]
