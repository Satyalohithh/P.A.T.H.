"""Pydantic model definitions for the dashboard API.

FastAPI/pydantic are third-party; these classes are stubs resolved later.
"""

from __future__ import annotations

from safewatch.api.schemas import JsonDict


class AlertResponse(JsonDict):
    """Serialized alert as returned by /alerts endpoints."""

    id: str
    message: str
    status: str


class StreamRegisterRequest(JsonDict):
    """Request body for POST /streams."""

    uri: str
    stream_id: str
    source_type: str


__all__ = ["AlertResponse", "JsonDict", "StreamRegisterRequest"]
