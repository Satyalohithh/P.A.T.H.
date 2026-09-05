"""Model quantization for edge deployment."""

from __future__ import annotations


class Quantizer:
    """Applied INT8/FP16 quantization."""

    def __init__(self, precision: str = "int8") -> None:
        self.precision = precision

    def quantize(self, exported_path: str) -> str:
        """Return the quantized model path."""

        raise NotImplementedError("TODO(implementation): Quantizer.quantize")


__all__ = ["Quantizer"]
