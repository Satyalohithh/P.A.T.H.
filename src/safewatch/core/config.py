"""Runtime configuration: a frozen Pydantic Settings object and YAML loader.

`Settings` validates everything through Pydantic, so invalid input fails fast
with a typed :class:`SafeWatchConfigError`. The YAML-backed loader
(:class:`YamlConfigLoader`) and the :func:`load_settings` convenience entry
point resolve configuration in this precedence order (lowest to highest):

1. Pydantic field defaults.
2. The YAML file (``--config`` / ``configs/default.yaml``).
3. Environment variables prefixed with ``SAFEWATCH_`` (e.g.
   ``SAFEWATCH_LOG_LEVEL``, ``SAFEWATCH_DETECTION_CONF_THRESHOLD``).
4. Explicit overrides passed to ``load_settings(overrides=...)``.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from safewatch.core.constants import (
    DEFAULT_CONTACT_PROXIMITY_GATE_H,
    DEFAULT_INTERACTION_WINDOW_FRAMES,
    DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD,
    ENV_PREFIX,
    INTIMATE_ZONE_H,
    MIN_VALID_KEYPOINTS_FOR_POSE,
    PERSONAL_ZONE_H,
    SMOOTHING_KERNEL_WIDTH,
    SMOOTHING_SIGMA_FRAMES,
    SOCIAL_ZONE_H,
)
from safewatch.core.exceptions import SafeWatchConfigError

DEFAULT_CONFIG_PATH = "configs/default.yaml"

_ConfigMapping: TypeAlias = dict[str, Any]
"""Alias for the loader output / Settings input keyspace."""


class DetectionConfig(BaseModel):
    """YOLO person-detection settings (YOLOv8n detection weights by default)."""

    model_config = ConfigDict(frozen=True)

    weights: str = "yolov8n.pt"
    conf_threshold: float = Field(default=0.45, ge=0.0, le=1.0)
    device: Literal["cuda", "cpu", "auto"] = "auto"
    person_class: int = Field(default=0, ge=0)


class TrackingConfig(BaseModel):
    """Multi-object tracking settings (ByteTrack backend)."""

    model_config = ConfigDict(frozen=True)

    backend: Literal["bytetrack"] = "bytetrack"
    track_thresh: float = Field(default=0.5, ge=0.0, le=1.0)
    match_thresh: float = Field(default=0.8, ge=0.0, le=1.0)
    track_buffer: int = Field(default=30, ge=1)
    min_hits: int = Field(default=1, ge=1)


class PoseConfig(BaseModel):
    """YOLO pose-estimation settings (YOLOv8n-pose weights by default)."""

    model_config = ConfigDict(frozen=True)

    weights: str = "yolov8n-pose.pt"
    conf_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    device: Literal["cuda", "cpu", "auto"] = "auto"


class Settings(BaseModel):
    """Immutable runtime settings validated by Pydantic."""

    model_config = ConfigDict(frozen=True)

    environment: str = "development"
    log_level: str = "INFO"
    log_format: str = "json"

    keypoint_confidence_threshold: float = DEFAULT_KEYPOINT_CONFIDENCE_THRESHOLD
    min_valid_keypoints_for_pose: int = MIN_VALID_KEYPOINTS_FOR_POSE
    interaction_window_frames: int = DEFAULT_INTERACTION_WINDOW_FRAMES
    contact_proximity_gate_h: float = DEFAULT_CONTACT_PROXIMITY_GATE_H
    intimate_zone_h: float = INTIMATE_ZONE_H
    personal_zone_h: float = PERSONAL_ZONE_H
    social_zone_h: float = SOCIAL_ZONE_H
    smoothing_sigma_frames: float = SMOOTHING_SIGMA_FRAMES
    smoothing_kernel_width: int = SMOOTHING_KERNEL_WIDTH

    detection: DetectionConfig = DetectionConfig()
    """Person-detection sub-configuration (weights, confidence, device)."""

    tracking: TrackingConfig = TrackingConfig()
    """Multi-object tracking sub-configuration (backend, thresholds)."""

    pose: PoseConfig = PoseConfig()
    """Pose-estimation sub-configuration (weights, confidence, device)."""

    extra: dict[str, Any] = Field(default_factory=dict)
    """Unrecognized overrides are preserved here instead of rejected."""

    # -- construction helpers ----------------------------------------------

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Settings:
        """Build settings from a plain mapping.

        Keys are normalized (lowercase, ``_`` for ``-``). Unknown keys are moved
        into ``extra``; ``detection_*`` keys are folded into the nested
        ``detection`` model. Invalid values raise :class:`SafeWatchConfigError`.
        """

        coerced: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        nested: dict[str, dict[str, Any]] = {}
        field_names = set(cls.model_fields)

        for key, raw in values.items():
            python_key = key.strip().lower().replace("-", "_")
            if python_key in field_names:
                coerced[python_key] = raw
                continue
            prefix, sep, suffix = python_key.partition("_")
            if (
                sep
                and prefix in _NESTED_MODEL_NAMES
                and suffix in _nested_field_names(prefix)
            ):
                nested.setdefault(prefix, {})[suffix] = raw
                continue
            extra[python_key] = raw

        coerced.update(nested)
        try:
            return cls(**coerced, extra=extra)
        except ValidationError as exc:
            raise SafeWatchConfigError(f"Invalid configuration values: {exc}") from exc

    @classmethod
    def from_environment(cls, prefix: str = ENV_PREFIX) -> Settings:
        """Build settings from environment variables carrying ``prefix``.

        Variable ``SAFEWATCH_LOG_LEVEL=DEBUG`` maps to ``log_level="DEBUG"`` and
        ``SAFEWATCH_DETECTION_CONF_THRESHOLD=0.6`` to ``detection.conf_threshold``.
        """

        values: dict[str, Any] = {}
        for key, raw in os.environ.items():
            if not key.startswith(prefix):
                continue
            values[key[len(prefix) :]] = raw
        return cls.from_mapping(values)

    def as_dict(self) -> dict[str, Any]:
        """Serialize to plain nested dicts (excluding the ``extra`` bucket)."""

        return self.model_dump(exclude={"extra"})


def _nested_field_names(model_name: str) -> set[str]:
    mapping: dict[str, type[BaseModel]] = {
        "detection": DetectionConfig,
        "tracking": TrackingConfig,
        "pose": PoseConfig,
    }
    return set(mapping[model_name].model_fields)


_NESTED_MODEL_NAMES = {"detection", "tracking", "pose"}


class ConfigLoader(ABC):
    """Base class for configuration-file loaders.

    Implementations must resolve relative references and return a settings-ready
    flat mapping (see :class:`YamlConfigLoader`).
    """

    @abstractmethod
    def load(self, path: str) -> _ConfigMapping:
        """Load ``path`` into a settings-ready mapping."""


class YamlConfigLoader(ConfigLoader):
    """Loads the runtime YAML and flattens it into settings keys.

    Nested sections (``logging``, ``detection``, ``pose``, ``interaction``,
    ``smoothing``) map onto the flat/structured ``Settings`` fields following
    the precedence documented in ``configs/README.md``.
    """

    def load(self, path: str) -> _ConfigMapping:
        try:
            with Path(path).open("r", encoding="utf-8") as handle:
                raw = yaml.safe_load(handle)
        except OSError as exc:
            raise SafeWatchConfigError(f"Config file not readable: {path}") from exc
        except yaml.YAMLError as exc:
            raise SafeWatchConfigError(f"Invalid YAML in {path}: {exc}") from exc

        if not isinstance(raw, dict):
            raise SafeWatchConfigError(f"Config root must be a mapping: {path}")

        return _flatten_yaml(raw)


def _flatten_yaml(raw: Mapping[str, Any]) -> _ConfigMapping:
    """Map the nested ``configs/default.yaml`` shape onto Settings keys."""

    mapping: _ConfigMapping = {}

    if "environment" in raw:
        mapping["environment"] = raw["environment"]

    logging_cfg = raw.get("logging")
    if isinstance(logging_cfg, Mapping):
        mapping.setdefault("log_level", logging_cfg.get("level"))
        mapping.setdefault("log_format", logging_cfg.get("format"))

    detection_cfg = raw.get("detection")
    if isinstance(detection_cfg, Mapping):
        detection: dict[str, Any] = {}
        _copy_keys(
            source=detection_cfg,
            target=detection,
            keys=("weights", "device", "person_class"),
        )
        if "confidence_threshold" in detection_cfg:
            detection["conf_threshold"] = detection_cfg["confidence_threshold"]
        if detection:
            mapping["detection"] = detection

    pose_cfg = raw.get("pose")
    if isinstance(pose_cfg, Mapping):
        _copy_keys(
            source=pose_cfg,
            target=mapping,
            keys=("keypoint_confidence_threshold", "min_valid_keypoints"),
        )
        _rename_key(mapping, "min_valid_keypoints", "min_valid_keypoints_for_pose")
        pose: dict[str, Any] = {}
        _copy_keys(source=pose_cfg, target=pose, keys=("weights", "device"))
        if "confidence_threshold" in pose_cfg:
            pose["conf_threshold"] = pose_cfg["confidence_threshold"]
        if pose:
            mapping["pose"] = pose

    interaction_cfg = raw.get("interaction")
    if isinstance(interaction_cfg, Mapping):
        _copy_keys(
            source=interaction_cfg,
            target=mapping,
            keys=("window_frames", "contact_proximity_gate_h"),
        )
        _rename_key(mapping, "window_frames", "interaction_window_frames")
        zones = interaction_cfg.get("zones")
        if isinstance(zones, Mapping):
            _copy_keys(
                source=zones,
                target=mapping,
                keys=("intimate_h", "personal_h", "social_h"),
            )
            _rename_key(mapping, "intimate_h", "intimate_zone_h")
            _rename_key(mapping, "personal_h", "personal_zone_h")
            _rename_key(mapping, "social_h", "social_zone_h")

    smoothing_cfg = raw.get("smoothing")
    if isinstance(smoothing_cfg, Mapping):
        if "sigma_frames" in smoothing_cfg:
            mapping["smoothing_sigma_frames"] = smoothing_cfg["sigma_frames"]
        if "kernel_width" in smoothing_cfg:
            mapping["smoothing_kernel_width"] = smoothing_cfg["kernel_width"]

    tracking_cfg = raw.get("tracking")
    if isinstance(tracking_cfg, Mapping):
        tracking: dict[str, Any] = {}
        _copy_keys(
            source=tracking_cfg,
            target=tracking,
            keys=(
                "backend",
                "track_thresh",
                "match_thresh",
                "track_buffer",
                "min_hits",
            ),
        )
        if tracking:
            mapping["tracking"] = tracking

    return mapping


def _copy_keys(
    source: Mapping[str, Any],
    target: dict[str, Any],
    keys: tuple[str, ...],
) -> None:
    """Copy present scalar keys from ``source`` to ``target`` verbatim."""

    for key in keys:
        if key in source:
            target[key] = source[key]


def _rename_key(mapping: _ConfigMapping, old: str, new: str) -> None:
    """Rename key ``old`` to ``new`` when present in ``mapping``."""

    if old in mapping:
        mapping[new] = mapping.pop(old)


def load_settings(
    config_path: str = DEFAULT_CONFIG_PATH,
    overrides: Mapping[str, Any] | None = None,
    prefix: str = ENV_PREFIX,
) -> Settings:
    """Load runtime settings by merging defaults, YAML, env, and overrides.

    Precedence (lowest to highest): Pydantic defaults, ``config_path``,
    ``SAFEWATCH_*`` environment variables, then explicit ``overrides``.

    Raises :class:`SafeWatchConfigError` on unreadable/invalid YAML or when the
    merged result fails Pydantic validation.
    """

    loader = YamlConfigLoader()
    merged: _ConfigMapping = loader.load(config_path)
    env_values = _normalized_from_mapping_keys(
        {
            key[len(prefix) :]: value
            for key, value in os.environ.items()
            if key.startswith(prefix)
        }
    )
    merged = _deep_merge(merged, _fold_nested(env_values))
    if overrides:
        merged = _deep_merge(merged, _fold_nested(dict(overrides)))
    return Settings.from_mapping(merged)


def _normalized_from_mapping_keys(values: Mapping[str, Any]) -> _ConfigMapping:
    """Normalize env keys (dashes to underscores, lowercase)."""

    return {
        key.strip().lower().replace("-", "_"): value for key, value in values.items()
    }


def _fold_nested(values: Mapping[str, Any]) -> _ConfigMapping:
    """Fold ``<model>_<field>`` keys (e.g. ``detection_conf_threshold``) into
    their nested mapping so layered merges keep unrelated defaults."""

    folded: _ConfigMapping = dict(values)
    for model_name in _NESTED_MODEL_NAMES:
        nested: dict[str, Any] = {}
        for key, value in list(folded.items()):
            prefix, sep, suffix = key.partition("_")
            if (
                sep
                and prefix == model_name
                and suffix in _nested_field_names(model_name)
            ):
                nested[suffix] = value
                del folded[key]
        if nested:
            folded[model_name] = nested
    return folded


def _deep_merge(base: _ConfigMapping, overlay: Mapping[str, Any]) -> _ConfigMapping:
    """Dotted-free deep merge; nested ``detection`` keys are merged pairwise."""

    merged: _ConfigMapping = dict(base)
    for key, value in overlay.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


__all__ = [
    "DEFAULT_CONFIG_PATH",
    "ConfigLoader",
    "DetectionConfig",
    "PoseConfig",
    "Settings",
    "TrackingConfig",
    "YamlConfigLoader",
    "load_settings",
]
