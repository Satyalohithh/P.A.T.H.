"""Dataset acquisition: unified labels, source adapters, manifest tooling."""

from __future__ import annotations

from safewatch.data.adapters import (
    ALL_SOURCES,
    REGISTRY,
    AdapterFactory,
    build_adapters,
    enabled_sources,
)
from safewatch.data.adapters.base import Clip, DatasetAdapter
from safewatch.data.labels import (
    UNIFIED_NAMES,
    UNIFIED_ORDER,
    DatasetClass,
    active_classes,
    from_behavior_class,
    label_name,
    label_value,
    missing_classes,
    parse_unified_label,
    to_behavior_class,
)

__all__ = [
    "ALL_SOURCES",
    "REGISTRY",
    "UNIFIED_NAMES",
    "UNIFIED_ORDER",
    "AdapterFactory",
    "Clip",
    "DatasetAdapter",
    "DatasetClass",
    "active_classes",
    "build_adapters",
    "enabled_sources",
    "from_behavior_class",
    "label_name",
    "label_value",
    "missing_classes",
    "parse_unified_label",
    "to_behavior_class",
]
