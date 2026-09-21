"""Dataset statistics report.

Aggregates a manifest into the numbers consumed by ``reports/`` and the
experiment write-up: class counts, per-class availability on the unified
3-class catalog, source distribution, and (when probed) duration totals.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from safewatch.data.labels import (
    UNIFIED_ORDER,
    DatasetClass,
    parse_unified_label,
)

DURATION_BUCKETS = [(0.0, 1.0), (1.0, 3.0), (3.0, 5.0), (5.0, 10.0)]
"""Half-open duration spans in seconds; anything larger falls in a tail."""


def compute_statistics(manifest: pd.DataFrame) -> dict[str, Any]:
    """Compute aggregate statistics over a loaded manifest."""

    _validate_labels(manifest)
    total = len(manifest)

    class_counts: dict[str, int] = {}
    for cls in UNIFIED_ORDER:
        class_counts[cls.value] = int((manifest["label"] == cls.value).sum())

    available = [cls.value for cls in UNIFIED_ORDER if class_counts[cls.value] > 0]
    missing = [cls.value for cls in UNIFIED_ORDER if class_counts[cls.value] == 0]

    source_counts = {
        str(source): int(count)
        for source, count in manifest["source_dataset"].value_counts().items()
    }
    crosstab: dict[str, dict[str, int]] = {}
    table = pd.crosstab(manifest["label"], manifest["source_dataset"])
    for label in table.index:
        crosstab[str(label)] = {
            str(source): int(table.at[label, source]) for source in table.columns
        }

    report: dict[str, Any] = {
        "total_clips": total,
        "class_counts": class_counts,
        "available_classes": available,
        "missing_classes": missing,
        "source_distribution": source_counts,
        "class_source_crosstab": crosstab,
        "duration": _duration_summary(manifest),
    }
    return report


def _validate_labels(manifest: pd.DataFrame) -> None:
    for label in manifest["label"]:
        try:
            parse_unified_label(label)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"manifest contains invalid label: {label!r}") from exc


def _duration_summary(manifest: pd.DataFrame) -> dict[str, Any] | None:
    if "duration_s" not in manifest.columns:
        return None
    durations = pd.to_numeric(manifest["duration_s"], errors="coerce").dropna()
    if len(durations) == 0:
        return {"count": 0}
    summary: dict[str, Any] = {
        "count": len(durations),
        "missing": int(len(manifest) - len(durations)),
        "mean_s": _clean(float(durations.mean())),
        "min_s": _clean(float(durations.min())),
        "max_s": _clean(float(durations.max())),
        "p50_s": _clean(float(durations.quantile(0.5))),
        "p90_s": _clean(float(durations.quantile(0.9))),
        "histogram_s": _duration_histogram(durations),
    }
    return summary


def _clean(value: float) -> float | None:
    """Return ``None`` for NaN so the JSON report stays standard-clean."""

    return None if math.isnan(value) else value


def _duration_histogram(durations: pd.Series) -> dict[str, int]:
    histogram: dict[str, int] = {
        f"[{int(lo)},{int(hi)})": 0 for lo, hi in DURATION_BUCKETS
    }
    histogram["[10,inf)"] = 0
    for value in durations:
        bucket = _bucket(float(value))
        histogram[bucket] += 1
    return histogram


def _bucket(value: float) -> str:
    for lo, hi in DURATION_BUCKETS:
        if lo <= value < hi:
            return f"[{int(lo)},{int(hi)})"
    return "[10,inf)"


def write_statistics(manifest: pd.DataFrame, output: Path) -> dict[str, Any]:
    """Compute and persist a statistics report to ``output`` (JSON)."""

    report = compute_statistics(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, default=_json_default), encoding="utf-8"
    )
    return report


def _json_default(value: object) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    raise TypeError(f"cannot serialize {type(value).__name__}")


__all__ = [
    "DURATION_BUCKETS",
    "DatasetClass",
    "compute_statistics",
    "write_statistics",
]
