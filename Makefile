.PHONY: install test test-all lint format typecheck check

install:
	uv sync --group dev

test:
	uv run pytest tests/unit -q

test-all:
	uv run pytest tests -q

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:
	uv run ruff format src tests scripts

typecheck:
	uv run mypy src

check: lint typecheck test