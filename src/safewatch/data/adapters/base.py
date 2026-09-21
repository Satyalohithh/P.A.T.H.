"""Adapter base contract for external video datasets.

Each adapter knows the on-disk layout of one public dataset under
``datasets/raw/<name>/`` and translates its *native* labels onto the unified
:class:`~safewatch.data.labels.DatasetClass` vocabulary. Adapters never
download media; they only index what is already present.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from safewatch.data.labels import DatasetClass

logger = logging.getLogger("safewatch.data.adapters")


@dataclass(frozen=True, slots=True)
class Clip:
    """A single indexed video: location, native label, provenance."""

    video_path: Path
    native_label: str
    source: str
    video_id: str


class DatasetAdapter(ABC):
    """Indexes one public dataset into :class:`Clip` records.

    Subclasses pin ``source``, the accepted ``native_classes`` and the layout
    discovery under :meth:`iter_clips`.
    """

    source: str = ""
    native_classes: tuple[str, ...] = ()
    raw_dir_name: str = ""

    @property
    @abstractmethod
    def enabled(self) -> bool:
        """Whether the adapter produces manifest rows by default."""

    def unified_label(self, native: str) -> DatasetClass:
        """Translate a native (per-source) label onto the unified schema."""

        token = native.strip().lower()
        for normalized, unified in self._native_to_unified().items():
            if token in normalized:
                return unified
        raise ValueError(
            f"{self.source}: unknown native label {native!r}; "
            f"expected one of {self.native_classes}"
        )

    @abstractmethod
    def iter_clips(self, raw_root: Path) -> Iterator[Clip]:
        """Yield every clip present under ``raw_root/<raw_dir_name>/``."""

    def _native_to_unified(self) -> dict[tuple[str, ...], DatasetClass]:
        raise NotImplementedError  # pragma: no cover - overridden per adapter


def iter_videos(directory: Path) -> Iterator[Path]:
    """Walk ``directory`` for common video containers in stable order."""

    extensions = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
    for path in sorted(directory.rglob("*")):
        if path.suffix.lower() in extensions and path.is_file():
            yield path


def stable_id(source: str, rel: Path) -> str:
    """Deterministic per-clip identifier: ``<source>-<flattened path>``."""

    parts = [part for part in rel.parts if part not in {"", ".", ".."}]
    return f"{source}-" + "-".join(parts)
