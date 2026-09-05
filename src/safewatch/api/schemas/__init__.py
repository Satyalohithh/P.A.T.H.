"""Pydantic request/response schemas."""

from __future__ import annotations

from typing import TypeAlias

JsonDict: TypeAlias = dict[str, object]


class ModelSchemas:
    """Placeholder namespace; pydantic BaseModel classes live here later."""


__all__ = ["JsonDict", "ModelSchemas"]
