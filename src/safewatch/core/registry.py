"""Registries for pipeline stages, feature extractors, and models.

Registries decouple configuration (which stage/extractor/model is active for
this deployment) from discovery (``import safewatch.features``). Lookup is by
string key so YAML configuration can reference names directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


class SafeWatchRegistry:
    """Name -> factory registry with duplicate detection."""

    def __init__(self, registry_name: str) -> None:
        self._registry_name = registry_name
        self._entries: dict[str, Callable[..., Any]] = {}

    def register(self, name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        raise NotImplementedError("TODO(implementation): SafeWatchRegistry.register")

    def resolve(self, name: str, **kwargs: Any) -> Any:
        raise NotImplementedError("TODO(implementation): SafeWatchRegistry.resolve")

    def available(self) -> tuple[str, ...]:
        raise NotImplementedError("TODO(implementation): SafeWatchRegistry.available")


class FeatureRegistry(SafeWatchRegistry):
    """Registry of named feature extractors (pose/motion/interaction)."""


class ModelRegistry(SafeWatchRegistry):
    """Registry of behavior classifiers keyed by model family."""


class PipelineStageRegistry(SafeWatchRegistry):
    """Registry of pipeline stage implementations keyed by stage name."""


__all__ = [
    "FeatureRegistry",
    "ModelRegistry",
    "PipelineStageRegistry",
    "SafeWatchRegistry",
]
