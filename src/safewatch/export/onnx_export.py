"""ONNX model export."""

from __future__ import annotations


class OnnxExporter:
    """Converts PyTorch/Ultralytics models to ONNX."""

    def __init__(self, family: str, model_path: str) -> None:
        self.family = family
        self.model_path = model_path

    def export(self, opset: int = 16) -> str:
        """Return the exported .onnx path."""

        raise NotImplementedError("TODO(implementation): OnnxExporter.export")

    def export_detection(self, dynamic_axes: bool = True) -> str:
        raise NotImplementedError("TODO(implementation): OnnxExporter.export_detection")


__all__ = ["OnnxExporter"]
