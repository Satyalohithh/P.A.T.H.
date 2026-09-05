"""Shared pytest fixtures for SafeWatch AI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from safewatch.core.config import Settings
from safewatch.core.constants import Keypoint
from safewatch.core.schemas.features import FeatureVector
from safewatch.core.schemas.pose import PoseResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def settings() -> Settings:
    return Settings.from_mapping({"log_level": "DEBUG", "environment": "test"})


@pytest.fixture
def sample_pose() -> PoseResult:
    """A pose with all 17 keypoints confident (confidence 0.9 each)."""

    keypoints = tuple(
        (float(idx * 10), float(idx * 10), 0.9) for idx in range(len(Keypoint))
    )
    confidences = tuple(0.9 for _ in range(len(Keypoint)))
    return PoseResult(track_id=1, keypoints=keypoints, confidences=confidences)


@pytest.fixture
def sample_feature_vector() -> FeatureVector:
    """A 9-d pose feature vector matching the pose feature names."""

    names = tuple(
        f"pose.{name}"
        for name in (
            "elbow_angle.L",
            "elbow_angle.R",
            "knee_angle.L",
            "knee_angle.R",
            "torso_lean",
            "shoulder_orientation",
            "upper_asymmetry",
            "complexity",
            "com_height",
        )
    )
    return FeatureVector(names=names, values=tuple(range(9)), frame_index=0, track_id=1)


@pytest.fixture
def sample_poses_path() -> Path:
    return FIXTURES_DIR / "sample_poses.json"


@pytest.fixture
def sample_features_path() -> Path:
    return FIXTURES_DIR / "sample_features.json"


def load_fixture(path: Path) -> object:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
