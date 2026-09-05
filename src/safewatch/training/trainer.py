"""Training orchestration for phase 1 (tabular) and phase 2 (temporal)."""

from __future__ import annotations

from safewatch.core.types import TrackId


class Trainer:
    """Runs a configured training job end-to-end."""

    def __init__(self, config_path: str) -> None:
        self.config_path = config_path

    def train(self) -> dict[str, float]:
        """Return the final tracked metrics."""

        raise NotImplementedError("TODO(implementation): Trainer.train")

    def fit_phase1(self, track_id: TrackId | None = None) -> dict[str, float]:
        raise NotImplementedError("TODO(implementation): Trainer.fit_phase1")

    def fit_phase2(self, track_id: TrackId | None = None) -> dict[str, float]:
        raise NotImplementedError("TODO(implementation): Trainer.fit_phase2")


__all__ = ["TrackId", "Trainer"]
