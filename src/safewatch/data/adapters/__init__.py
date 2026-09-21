"""Dataset adapter registry.

Selecting sources via :func:`build_adapters` instantiates the requested
adapters; ``enabled_sources()`` lists the ones that produce manifest rows by
default (scaffolds such as UCF101 are excluded until their mapping exists).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeAlias

from safewatch.data.adapters.base import DatasetAdapter
from safewatch.data.adapters.hockey_fight import HockeyFightAdapter
from safewatch.data.adapters.rwf2000 import RWF2000Adapter
from safewatch.data.adapters.ucf101 import UCF101Adapter

AdapterFactory: TypeAlias = DatasetAdapter

REGISTRY: dict[str, type[DatasetAdapter]] = {
    RWF2000Adapter.source: RWF2000Adapter,
    HockeyFightAdapter.source: HockeyFightAdapter,
    UCF101Adapter.source: UCF101Adapter,
}
"""All known adapters keyed by ``source`` name."""


def build_adapters(sources: Iterable[str]) -> list[DatasetAdapter]:
    """Instantiate adapters for ``sources`` (names or ``"all"``)."""

    names = list(sources)
    if "all" in {name.lower() for name in names}:
        names = list(REGISTRY)
    adapters: list[DatasetAdapter] = []
    for name in names:
        normalized = name.lower()
        try:
            factory = REGISTRY[normalized]
        except KeyError:
            known = ", ".join(sorted(REGISTRY))
            raise ValueError(
                f"unknown dataset source {name!r}; known: {known}"
            ) from None
        adapters.append(factory())
    return adapters


def enabled_sources() -> list[str]:
    """Names of adapters that should run in default pipelines."""

    return [name for name, factory in REGISTRY.items() if factory().enabled]


def all_sources() -> list[str]:
    """Names of every registered adapter."""

    return sorted(REGISTRY)


__all__ = [
    "ALL_SOURCES",
    "REGISTRY",
    "AdapterFactory",
    "build_adapters",
    "enabled_sources",
]


ALL_SOURCES = all_sources()
