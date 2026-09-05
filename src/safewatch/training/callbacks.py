"""Training callbacks: early stopping and checkpointing."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Stops training when a monitored metric stalls."""

    patience: int = 10
    min_delta: float = 1e-4
    mode: str = "max"  # max | min

    def on_epoch_end(self, epoch: int, metric: float) -> bool:
        """Return True if training should stop."""

        raise NotImplementedError("TODO(implementation): EarlyStopping.on_epoch_end")


@dataclass
class ModelCheckpoint:
    """Saves the best K checkpoints by monitored metric."""

    directory: str
    monitor: str = "val_f1_macro"
    save_top_k: int = 3
    mode: str = "max"

    def on_epoch_end(
        self, epoch: int, metrics: dict[str, float], model: object
    ) -> None:
        raise NotImplementedError("TODO(implementation): ModelCheckpoint.on_epoch_end")


__all__ = ["EarlyStopping", "ModelCheckpoint"]
