#!/usr/bin/env bash
# Bootstrap a SafeWatch AI development environment.
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.12}"

echo "==> Installing project with uv"
uv sync --group dev

echo "==> Installing pre-commit hooks"
uv run pre-commit install

echo "==> Copying env template (safe to ignore if .env exists)"
cp -n .env.example .env || true

echo "==> Creating runtime directories"
mkdir -p models/checkpoints models/exported mlruns

echo "Done."
echo "Run: uv run pytest tests/unit -q"