# SafeWatch AI

Context-aware aggression detection from surveillance video. Distinguishes
**Normal**, **Playful**, **Suspicious**, and **Aggressive** human interactions
using pose dynamics, multi-person tracking, and interpersonal movement analysis.

> Status: scaffolding. Interfaces and data contracts are defined; pipeline
> stages are declared but not yet implemented.

## Overview

```
Video Stream → Detection → Pose → Tracking → Features → Behavior → Risk → Alerts → Dashboard
```

## Repository layout

| Path | Purpose |
|---|---|
| `src/safewatch/` | Python package (src-layout) |
| `configs/` | YAML runtime configuration |
| `tests/` | Unit and integration smoke tests |
| `scripts/` | CLI entry points |
| `docs/` | Architecture and design documents |
| `research/` | Literature review and experiment planning |
| `datasets/` | Dataset acquisition and split management |
| `frontend/` | React dashboard (stubs) |
| `deployment/` | Docker and production configuration |
| `.github/` | CI/CD workflows |

## Development

```bash
uv sync --group dev        # install dependencies
uv run pytest tests/unit   # run unit tests
uv run ruff check src      # lint
uv run mypy src            # type check
```

## Documentation

- [Architecture](docs/architecture.md)
- [Roadmap](docs/roadmap.md)
- [Dataset strategy](docs/dataset_strategy.md)
- [Feature catalog](docs/feature_catalog.md)
- [Contributing](docs/CONTRIBUTING.md)

## License

MIT — see [LICENSE](LICENSE).