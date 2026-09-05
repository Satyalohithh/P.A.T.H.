"""Runtime configuration: a frozen, environment-driven Settings object.

`Settings` is deliberately stdlib-only so the package imports cleanly before
pydantic / pydantic-settings are installed. The YAML-backed loaders
(`YamlConfigLoader`, `PydanticSettingsLoader`) are declared stubs to be
implemented once those dependencies are available.

Resolution order (lowest to highest precedence):
1. Class default values.
2. Explicit mapping via `Settings.from_mapping`.
3. Environment variables prefixed with `SAFEWATCH_` (e.g. `SAFEWATCH_LOG_LEVEL`).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from typing import Any, ClassVar

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


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable runtime settings with environment/mapping constructors."""

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
    extra: dict[str, Any] = field(default_factory=dict)
    """Unrecognized overrides are preserved here instead of rejected."""

    _FIELDS: ClassVar[dict[str, type]] = {
        "environment": str,
        "log_level": str,
        "log_format": str,
        "keypoint_confidence_threshold": float,
        "min_valid_keypoints_for_pose": int,
        "interaction_window_frames": int,
        "contact_proximity_gate_h": float,
        "intimate_zone_h": float,
        "personal_zone_h": float,
        "social_zone_h": float,
        "smoothing_sigma_frames": float,
        "smoothing_kernel_width": int,
    }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> Settings:
        """Build settings from a plain mapping, coercing scalar types.

        Unknown keys are silently moved into ``extra``; invalid values for a
        known key raise :class:`SafeWatchConfigError`.
        """

        coerced: dict[str, Any] = {}
        extra: dict[str, Any] = {}
        for key, raw in values.items():
            python_key = key.strip().lower().replace("-", "_")
            if python_key not in cls._FIELDS:
                extra[python_key] = raw
                continue
            try:
                coerced[python_key] = cls._coerce_value(cls._FIELDS[python_key], raw)
            except (TypeError, ValueError) as exc:
                raise SafeWatchConfigError(
                    f"Invalid value for {python_key}: {raw!r}"
                ) from exc
        return cls(**coerced, extra=extra)

    @classmethod
    def from_environment(cls, prefix: str = ENV_PREFIX) -> Settings:
        """Build settings from environment variables carrying ``prefix``.

        Variable ``SAFEWATCH_LOG_LEVEL=DEBUG`` maps to ``log_level="DEBUG"``.
        """

        values: dict[str, Any] = {}
        for key, raw in os.environ.items():
            if not key.startswith(prefix):
                continue
            values[key[len(prefix) :]] = raw
        return cls.from_mapping(values)

    @staticmethod
    def _coerce_value(expected: type, raw: Any) -> Any:
        if expected is bool:
            if isinstance(raw, bool):
                return raw
            normalized = str(raw).strip().lower()
            if normalized in {"1", "true", "yes", "on"}:
                return True
            if normalized in {"0", "false", "no", "off"}:
                return False
            raise ValueError(raw)
        return expected(raw)

    def as_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (excluding the ``extra`` bucket)."""

        return {
            field.name: getattr(self, field.name)
            for field in fields(self)
            if field.name != "extra"
        }


class ConfigLoader:
    """Base class for configuration-file loaders.

    Implementations must resolve relative references (e.g. a detection model
    config referenced from ``default.yaml``) and deep-merge results.
    """

    def load(self, path: str) -> dict[str, Any]:
        raise NotImplementedError("TODO(implementation): ConfigLoader.load")


class YamlConfigLoader(ConfigLoader):
    """Loads and validates YAML configuration skeletons.

    The merged output must contain every key with a declared default;
    resolution follows the precedence documented in ``configs/README.md``.
    """

    def load(self, path: str) -> dict[str, Any]:
        raise NotImplementedError("TODO(implementation): YamlConfigLoader.load")


class PydanticSettingsLoader(ConfigLoader):
    """Drop-in loader backed by pydantic-settings for strict validation."""

    def load(self, path: str) -> dict[str, Any]:
        raise NotImplementedError("TODO(implementation): PydanticSettingsLoader.load")


__all__ = ["ConfigLoader", "PydanticSettingsLoader", "Settings", "YamlConfigLoader"]
