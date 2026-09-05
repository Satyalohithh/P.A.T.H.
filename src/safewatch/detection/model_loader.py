"""Model artifact loading and path resolution."""

from __future__ import annotations

from safewatch.core.exceptions import ModelRegistryError

ModelArtifact = object
"""Loaded model artifact (Ultralytics YOLO / PyTorch module at runtime)."""


class ModelLoader:
    """Resolves model config files and checkpoints to runtime artifacts."""

    def __init__(self, model_config_path: str) -> None:
        self.model_config_path = model_config_path

    def load(self) -> ModelArtifact:
        raise NotImplementedError("TODO(implementation): ModelLoader.load")

    def resolve_weights(self) -> str:
        """Return the absolute path/URI of the weights file."""

        raise NotImplementedError("TODO(implementation): ModelLoader.resolve_weights")

    def unload(self) -> None:
        raise NotImplementedError("TODO(implementation): ModelLoader.unload")


__all__ = ["ModelArtifact", "ModelLoader", "ModelRegistryError"]
