"""Structured JSON logging for SafeWatch AI.

Functional, stdlib-only logging bootstrap: configure a root logger with a
JSON formatter, plus a context-aware ``get_logger`` helper. Machine-readable
logs are expected by the alert/dashboard integrations.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, cast


class JsonFormatter(logging.Formatter):
    """emit each record as a single JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "safewatch_extra", None)
        if isinstance(extra, dict):
            payload.update(extra)
        if record.exc_info:
            exc_info = (
                record.exc_info
                if isinstance(record.exc_info, tuple)
                else sys.exc_info()
            )
            payload["exc_info"] = self.formatException(exc_info)
        return json.dumps(payload, default=str)


def configure_logging(
    level: str = "INFO",
    fmt: str = "json",
    stream: Any = None,
) -> None:
    """Configure the root logger. ``fmt`` is ``json`` or ``plain``."""

    handler = logging.StreamHandler(stream or sys.stderr)
    if fmt == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.handlers.clear()
    root.addHandler(handler)


def get_logger(name: str, extra: dict[str, Any] | None = None) -> logging.Logger:
    """Return namespace logger; optional static extra fields attached later."""

    logger = logging.getLogger(name)
    if extra:
        # Store on the factory side; merged at format time via safewatch_extra.
        return cast(logging.Logger, logging.LoggerAdapter(logger, extra))
    return logger


__all__ = ["JsonFormatter", "configure_logging", "get_logger"]
