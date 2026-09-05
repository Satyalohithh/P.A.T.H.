"""Canonical feature identifiers and helpers.

Identifiers are frozen; do not rename or reorder existing entries.
"""

from __future__ import annotations

from safewatch.core.constants import FEATURE_GROUPS

FEATURE_NAMES_RAW = {
    "pose": (
        "pose.elbow_angle.L",
        "pose.elbow_angle.R",
        "pose.knee_angle.L",
        "pose.knee_angle.R",
        "pose.torso_lean",
        "pose.shoulder_orientation",
        "pose.upper_asymmetry",
        "pose.complexity",
        "pose.com_height",
    ),
    "motion": (
        "motion.arm_angular_vel.L",
        "motion.arm_angular_vel.R",
        "motion.leg_angular_vel.L",
        "motion.leg_angular_vel.R",
        "motion.arm_angular_acc.L",
        "motion.arm_angular_acc.R",
        "motion.leg_angular_acc.L",
        "motion.leg_angular_acc.R",
        "motion.com_vel.x",
        "motion.com_vel.y",
        "motion.com_speed",
        "motion.com_acc.x",
        "motion.com_acc.y",
        "motion.com_acc_mag",
        "motion.limb_energy",
        "motion.stride_length",
        "motion.gait_regularity",
    ),
    "interaction": (
        "inter.distance",
        "inter.approach_rate",
        "inter.mutual_approach",
        "inter.facing_angle",
        "inter.confrontation_index",
        "inter.energy_ratio",
        "inter.pose_mirroring",
        "inter.contact_proximity",
        "inter.reach_vector",
        "inter.retreat_velocity",
    ),
}

FEATURE_NAMES: tuple[str, ...] = tuple(
    name for group in FEATURE_GROUPS for name in FEATURE_NAMES_RAW[group]
)
"""Flattened canonical feature-vector column order."""

FEATURE_NAME_INDEX: dict[str, int] = {name: i for i, name in enumerate(FEATURE_NAMES)}


def feature_id(group: str, name: str) -> str:
    """Build a canonical feature id, e.g. ``feature_id("pose", "torso_lean")``."""

    raise NotImplementedError("TODO(implementation): feature_id")


__all__ = ["FEATURE_NAMES", "FEATURE_NAMES_RAW", "FEATURE_NAME_INDEX", "feature_id"]
