"""Unified external-dataset label schema (3 acquisition classes).

Public fighting datasets (RWF-2000, Hockey Fight) expose at most a coarse
binary distinction and UCF101 carries 101 action labels, none of which is an
aggression ground truth. Phase 6 therefore normalizes every clip onto the
three *acquisition* classes below.

These classes are intentionally distinct from the frozen four-way
:class:`~safewatch.core.constants.BehaviorClass` used for runtime inference:
acquisition labels no more expressive than the sources provide, while
:meth:`to_behavior_class` / :meth:`from_behavior_class` bridge to the frozen
taxonomy at training time.
"""

from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

from safewatch.core.constants import BehaviorClass


class DatasetClass(StrEnum):
    """Three-class vocabulary shared by all external dataset adapters."""

    NORMAL = "normal"
    PLAYFUL = "playful"
    AGGRESSIVE = "aggressive"


UNIFIED_ORDER = tuple(DatasetClass)
"""Frozen ordered set of unified acquisition classes."""

UNIFIED_NAMES: dict[DatasetClass, str] = {
    DatasetClass.NORMAL: "normal",
    DatasetClass.PLAYFUL: "playful",
    DatasetClass.AGGRESSIVE: "aggressive",
}
"""Display names for the unified acquisition classes."""


def parse_unified_label(value: object) -> DatasetClass:
    """Coerce ``value`` to a :class:`DatasetClass`.

    Accepts the enum member itself or a (case-insensitive) canonical name /
    value string. Raises :class:`TypeError` for booleans and
    :class:`ValueError` for anything else.
    """

    if isinstance(value, DatasetClass):
        return value
    if isinstance(value, bool):
        raise TypeError(f"invalid unified label: {value!r}")
    if isinstance(value, str):
        token = value.strip().lower()
        for cls in UNIFIED_ORDER:
            if cls.value == token or cls.name.lower() == token:
                return cls
        raise ValueError(f"invalid unified label: {value!r}")
    raise ValueError(f"invalid unified label: {value!r}")


def label_name(cls: DatasetClass) -> str:
    """Return the canonical display name for ``cls``."""

    return UNIFIED_NAMES[cls]


def label_value(cls: DatasetClass) -> str:
    """Return the canonical string value identifying ``cls`` in exports."""

    return cls.value


def to_behavior_class(cls: DatasetClass) -> BehaviorClass:
    """Map an acquisition class onto the frozen runtime taxonomy.

    ``suspicious`` has no acquisition equivalent and is therefore never
    produced by this mapping.
    """

    mapping: dict[DatasetClass, BehaviorClass] = {
        DatasetClass.NORMAL: BehaviorClass.NORMAL,
        DatasetClass.PLAYFUL: BehaviorClass.PLAYFUL,
        DatasetClass.AGGRESSIVE: BehaviorClass.AGGRESSIVE,
    }
    return mapping[cls]


def from_behavior_class(cls: BehaviorClass) -> DatasetClass | None:
    """Inverse of :meth:`to_behavior_class`; ``None`` for ``suspicious``."""

    mapping: dict[BehaviorClass, DatasetClass] = {
        BehaviorClass.NORMAL: DatasetClass.NORMAL,
        BehaviorClass.PLAYFUL: DatasetClass.PLAYFUL,
        BehaviorClass.AGGRESSIVE: DatasetClass.AGGRESSIVE,
    }
    return mapping.get(cls)


def active_classes(labels: Iterable[DatasetClass]) -> tuple[DatasetClass, ...]:
    """Unique acquisition classes present, in frozen ``UNIFIED_ORDER``."""

    present = set(labels)
    if any(not isinstance(cls, DatasetClass) for cls in present):
        raise TypeError("labels must be DatasetClass members")
    return tuple(cls for cls in UNIFIED_ORDER if cls in present)


def missing_classes(labels: Iterable[DatasetClass]) -> tuple[DatasetClass, ...]:
    """Acquisition classes absent from ``labels``, in ``UNIFIED_ORDER``."""

    present = set(labels)
    return tuple(cls for cls in UNIFIED_ORDER if cls not in present)


__all__ = [
    "UNIFIED_NAMES",
    "UNIFIED_ORDER",
    "DatasetClass",
    "active_classes",
    "from_behavior_class",
    "label_name",
    "label_value",
    "missing_classes",
    "parse_unified_label",
    "to_behavior_class",
]
