# AGENTS.md

Context for AI-assisted and human developers working in this repository.

## Project

SafeWatch AI is a production/research system classifying human interactions
(Normal / Playful / Suspicious / Aggressive) from surveillance video using
YOLO Pose + ByteTrack + engineered interaction features. The architecture,
feature catalog, and feature extraction specification are FROZEN. Do not
redesign the system; do not add features or change formulas without approval.

## Commands

```bash
uv sync --group dev
uv run pytest tests/unit -q
uv run ruff check src tests scripts
uv run ruff format src tests scripts
uv run mypy src
```

## Guidelines

- Python >= 3.11, src-layout, package `safewatch`.
- Future-style annotations (`from __future__ import annotations`) in every module.
- Infrastructure only in this scaffold. Business logic and ML algorithms are stubs
  raising `NotImplementedError("TODO(implementation): ...")`.
- Data contracts live in `src/safewatch/core/schemas/`.
- Runtime configuration is YAML under `configs/`; environment settings via
  `SAFEWATCH_`-prefixed variables.
- Do not commit checkpoints, model weights, raw datasets, `mlruns/`, or `.env`.

## Reading order

1. `docs/architecture.md`
2. `src/safewatch/core/schemas/`
3. `src/safewatch/features/`
4. `src/safewatch/pipeline` stages in ingestion → detection → pose → tracking order