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

    raise NotImplementedError("TODO(implementation): label_name")


__all__ = ["LABELS", "LABEL_NAMES", "BehaviorClass", "label_name"]
