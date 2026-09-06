"""Pose estimation stage."""

from __future__ import annotations

from safewatch.pose.associator import PoseAssociator, PoseCandidate
from safewatch.pose.estimator import PoseEstimator
from safewatch.pose.keypoint_layout import (
    COCO_17_KEYPOINT_NAMES,
    COCO_17_SKELETON,
    KEYPOINT_COUNT,
    validate_keypoints,
)
from safewatch.pose.keypoint_utils import KeypointUtils
from safewatch.pose.occlusion_handler import OcclusionHandler
from safewatch.pose.yolo_pose_estimator import YOLOPoseEstimator

__all__ = [
    "COCO_17_KEYPOINT_NAMES",
    "COCO_17_SKELETON",
    "KEYPOINT_COUNT",
    "KeypointUtils",
    "OcclusionHandler",
    "PoseAssociator",
    "PoseCandidate",
    "PoseEstimator",
    "YOLOPoseEstimator",
    "validate_keypoints",
]
