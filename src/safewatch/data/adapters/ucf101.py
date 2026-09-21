"""UCF101 adapter scaffold.

UCF101 has 101 action labels (none aggression ground truth). Mapping requires
a curated class->unified table that has not been agreed yet, so this adapter
is disabled and intentionally raises until that map exists.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from safewatch.data.adapters.base import Clip, DatasetAdapter
from safewatch.data.labels import DatasetClass


class UCF101Adapter(DatasetAdapter):
    """Scaffold for curated UCF101 reading; disabled until mapped."""

    source = "ucf101"
    raw_dir_name = "ucf101"
    native_classes = ()

    @property
    def enabled(self) -> bool:
        return False

    def _native_to_unified(self) -> dict[tuple[str, ...], DatasetClass]:
        raise NotImplementedError(
            "TODO(implementation): ucf101 curated class->unified map"
        )

    def iter_clips(self, raw_root: Path) -> Iterator[Clip]:
        raise NotImplementedError(
            "TODO(implementation): ucf101 curated class->unified map"
        )
