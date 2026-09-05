# Contributing

See `../CONTRIBUTING.md` and `../AGENTS.md`.

## Workflow

1. Branch from `main` (`feat/`, `fix/`, `research/`).
2. `make check` must pass (ruff, mypy, unit tests).
3. Tests for new behavior go under `tests/unit/`; integration under
   `tests/integration/`.
4. Data/model artifacts are never committed (see `.gitignore`, use DVC/MLflow).

## Frozen contracts

Do not rename feature ids (`src/safewatch/features/feature_names.py`) or enum
values (`src/safewatch/core/constants.py`) — doing so corrupts saved datasets.
Changes require a documented migration.