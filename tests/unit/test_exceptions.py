"""Tests for the exception hierarchy."""

from __future__ import annotations

import pytest

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


def test_subsystem_errors_share_common_ancestors() -> None:
    pipeline_errors = (
        DetectionError,
        TrackingError,
        PoseEstimationError,
        FeatureExtractionError,
        ClassificationError,
        RiskEvaluationError,
        AlertDispatchError,
    )
    for error in pipeline_errors:
        assert issubclass(error, SafeWatchPipelineError)
        assert issubclass(error, SafeWatchError)


def test_config_error_is_direct_safe_watch_error() -> None:
    assert issubclass(SafeWatchConfigError, SafeWatchError)
    assert not issubclass(SafeWatchConfigError, SafeWatchPipelineError)


def test_storage_and_model_errors() -> None:
    assert issubclass(StorageError, SafeWatchError)
    assert issubclass(ModelRegistryError, SafeWatchError)
    assert issubclass(DataValidationError, SafeWatchError)


def test_catch_root_ancestor() -> None:
    with pytest.raises(SafeWatchError):
        raise SafeWatchConfigError("bad config")
