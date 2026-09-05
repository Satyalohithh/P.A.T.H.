"""Integration smoke tests: end-to-end importability of pipeline layers.

These require the project installed (``uv sync --group dev``); dependencies
on third-party libs are only exercised at the boundary, not during import.
"""

from __future__ import annotations

import pytest

import safewatch  # noqa: F401
from safewatch.core.config import Settings
from safewatch.core.context import PipelineContext


def test_core_round_trip() -> None:
    settings = Settings.from_mapping({"environment": "ci"})
    ctx = PipelineContext(settings=settings, stream_id="cam-ci")
    assert ctx.settings.environment == "ci"
    assert ctx.stream_id == "cam-ci"
    assert ctx.runtime_seconds == pytest.approx(0.0, abs=0.1)


def test_pipeline_layers_import() -> None:
    from safewatch import (  # noqa: F401  # noqa: F401, E501
        alerts,
        api,
        behavior,
        detection,
        evaluation,
        export,
        features,
        ingestion,
        pose,
        risk,
        storage,
        tracking,
        training,
        utils,
    )

    assert features.FeaturePipeline is not None
