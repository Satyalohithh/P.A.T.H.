"""FastAPI application factory for the SafeWatch dashboard backend.

FastAPI is a third-party dependency; create_app is defined but only
importable once fastapi is installed. Keep it dependency-hidden: importing
``safewatch.api`` does not import fastapi at module load.
"""

from __future__ import annotations

from typing import Any

T = Any
_app: T | None = None


def create_app() -> Any:
    """Return a configured FastAPI/Starlette application instance."""

    if _app is not None:
        return _app
    raise NotImplementedError("TODO(implementation): create_app")


__all__ = ["create_app"]
