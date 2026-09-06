"""Label schema: mapping between BehaviorClass and display names."""

from __future__ import annotations

from safewatch.core.constants import BehaviorClass

LABELS: tuple[BehaviorClass, ...] = tuple(BehaviorClass)
"""Frozen ordered label set (index == label value)."""

LABEL_NAMES: dict[BehaviorClass, str] = {
    BehaviorClass.NORMAL: "normal",
    BehaviorClass.PLAYFUL: "playful",
    BehaviorClass.SUSPICIOUS: "suspicious",
    BehaviorClass.AGGRESSIVE: "aggressive",
}
"""Display names; used by dashboard and metric labels."""


def label_name(cls: BehaviorClass) -> str:
    """Return the canonical display name for ``cls``."""

    return LABEL_NAMES[cls]


def parse_label(value: object) -> BehaviorClass:
    """Coerce ``value`` to a :class:`BehaviorClass`.

    Accepts the canonical display name (case-insensitive), the integer label
    value, or the enum member itself. Raises :class:`ValueError` otherwise.
    """

    if isinstance(value, BehaviorClass):
        return value
    if isinstance(value, bool):
        raise TypeError(f"invalid behavior label: {value!r}")
    if isinstance(value, int):
        try:
            return BehaviorClass(value)
        except ValueError:
            raise ValueError(f"invalid behavior label: {value!r}") from None
    if isinstance(value, str):
        normalized = value.strip().lower()
        for cls, name in LABEL_NAMES.items():
            if name == normalized:
                return cls
        try:
            return BehaviorClass(int(normalized))
        except ValueError:
            raise ValueError(f"invalid behavior label: {value!r}") from None
    raise ValueError(f"invalid behavior label: {value!r}")


def label_value(value: object) -> int:
    """Return the integer value of a behavior label parsed via :func:`parse_label`."""

    return parse_label(value).value


__all__ = [
    "LABELS",
    "LABEL_NAMES",
    "BehaviorClass",
    "label_name",
    "label_value",
    "parse_label",
]
