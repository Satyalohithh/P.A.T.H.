"""Stratified cross-validation harness."""

from __future__ import annotations

from safewatch.core.schemas.features import FeatureVector


class CrossValidator:
    """Runs k-fold CV respecting stream-level splits."""

    def __init__(self, n_folds: int = 5, shuffle: bool = True, seed: int = 42) -> None:
        self.n_folds = n_folds
        self.shuffle = shuffle
        self.seed = seed

    def split(self, vectors: list[FeatureVector]) -> list[tuple[list[int], list[int]]]:
        """Return (train_idx, val_idx) pairs for each fold."""

        raise NotImplementedError("TODO(implementation): CrossValidator.split")


__all__ = ["CrossValidator", "FeatureVector"]
