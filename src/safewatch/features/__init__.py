"""Feature extraction stage (pose, motion, interaction).

Frozen feature catalog:
- Group A (pose):        pose.elbow_angle.L/R, pose.knee_angle.L/R,
                         pose.torso_lean, pose.shoulder_orientation,
                         pose.upper_asymmetry, pose.complexity, pose.com_height
- Group B (motion):      motion.arm_angular_vel.L/R, motion.leg_angular_vel.L/R,
                         motion.arm_angular_acc.L/R, motion.leg_angular_acc.L/R,
                         motion.com_vel.x/y, motion.com_speed, motion.com_acc.*,
                         motion.limb_energy, motion.stride_length, motion.gait_regularity
- Group C (interaction): inter.distance, inter.approach_rate, inter.mutual_approach,
                         inter.facing_angle, inter.confrontation_index,
                         inter.energy_ratio, inter.pose_mirroring,
                         inter.contact_proximity, inter.reach_vector, inter.retreat_velocity
"""

from __future__ import annotations

from safewatch.features.base import (
    FeatureExtractor,
    FeatureExtractorError,
    register_feature_extractor,
)
from safewatch.features.feature_names import FEATURE_NAMES, feature_id
from safewatch.features.feature_pipeline import FeaturePipeline
from safewatch.features.interaction_features import InteractionFeatureExtractor
from safewatch.features.motion_features import MotionFeatureExtractor
from safewatch.features.pose_features import PoseFeatureExtractor
from safewatch.features.temporal_aggregator import TemporalAggregator

__all__ = [
    "FEATURE_NAMES",
    "FeatureExtractor",
    "FeatureExtractorError",
    "FeaturePipeline",
    "InteractionFeatureExtractor",
    "MotionFeatureExtractor",
    "PoseFeatureExtractor",
    "TemporalAggregator",
    "feature_id",
    "register_feature_extractor",
]
