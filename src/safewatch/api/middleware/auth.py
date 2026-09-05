"""Authentication / authorization middleware (JWT or API key)."""

from __future__ import annotations

from safewatch.core.types import StreamId


class AuthMiddleware:
    """Validates bearer tokens or API keys on protected routes."""

    def __init__(self, mode: str = "disabled") -> None:
        self.mode = mode

    def authorize(self, token: str | None, stream_id: StreamId | None = None) -> bool:
        raise NotImplementedError("TODO(implementation): AuthMiddleware.authorize")


__all__ = ["AuthMiddleware", "StreamId"]
