"""Per-run pipeline context.

The context is the single source of truth for a pipeline execution: it carries
the immutable runtime settings, identifies the source stream, and exposes the
request logger. Stages read from it; they must never write to it.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from safewatch.core.config import Settings
from safewatch.core.types import StreamId


@dataclass(frozen=True, slots=True)
class PipelineContext:
    """Immutable context passed through every pipeline stage."""

    settings: Settings
    stream_id: StreamId = "cam-00"
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("safewatch.pipeline")
    )
    started_monotonic: float = field(default_factory=time.monotonic)

    @property
    def runtime_seconds(self) -> float:
        """Elapsed wall-clock time for this pipeline run."""

        return time.monotonic() - self.started_monotonic

    def child_logger(self, name: str) -> logging.Logger:
        """Derive a stage-namespaced logger from the context logger."""

        return self.logger.getChild(name)


__all__ = ["PipelineContext"]
