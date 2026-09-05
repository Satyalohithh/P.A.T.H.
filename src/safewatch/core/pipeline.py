"""Pipeline composition framework.

A pipeline is an ordered sequence of stages. Each stage receives the current
payload plus the :class:`PipelineContext` and returns an updated payload.

Signature contract (frozen):
    ``pipeline_process(ctx, payload) -> payload``

Implemented processing graph (input -> output):
    VideoStream -> Detection -> Pose -> Tracking -> FeatureExtraction
    -> BehavioralUnderstanding -> RiskAssessment -> AlertEngine -> Dashboard
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from safewatch.core.context import PipelineContext
from safewatch.core.exceptions import SafeWatchPipelineError


@runtime_checkable
class PipelineStage(Protocol):
    """A single processing stage in the SafeWatch pipeline."""

    name: str

    def process(self, ctx: PipelineContext, payload: Any) -> Any:
        """Transform ``payload``; return the updated payload."""
        ...


@dataclass
class Pipeline:
    """Ordered stage runner.

    Stages are executed in registration order. A stage may branch the payload
    (e.g. tracking validates pose existence) but must never mutate the
    context; the context is authoritative for the whole pipeline run.
    """

    stages: list[PipelineStage]

    def add_stage(self, stage: PipelineStage) -> None:
        raise NotImplementedError("TODO(implementation): Pipeline.add_stage")

    def run(self, ctx: PipelineContext, payload: Any) -> Any:
        raise NotImplementedError("TODO(implementation): Pipeline.run")

    def validate_graph(self) -> None:
        """Ensure the stage order matches the frozen processing graph.

        Raises :class:`SafeWatchPipelineError` on cycles or unknown stages.
        """

        raise NotImplementedError("TODO(implementation): Pipeline.validate_graph")


__all__ = ["Pipeline", "PipelineContext", "PipelineStage", "SafeWatchPipelineError"]
