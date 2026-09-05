"""Structured request logging for API calls."""

from __future__ import annotations


class RequestLoggingMiddleware:
    """Emits one JSON log line per request (status, latency, route)."""

    def __init__(self, level: str = "INFO") -> None:
        self.level = level

    def log(self, method: str, path: str, status: int, latency_ms: float) -> None:
        raise NotImplementedError("TODO(implementation): RequestLoggingMiddleware.log")


__all__ = ["RequestLoggingMiddleware"]
