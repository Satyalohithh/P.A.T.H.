#!/usr/bin/env python3
"""Assign behavior labels to an unlabeled feature dataset.

Reads a CSV/Parquet dataset written by the Phase 5 schema (feature columns,
plus ``label``/``id``/``window_start``/``window_end``/... reserved columns).
Labels come either from a ``label`` column already present in the file or from
an annotations JSON file in the ``docs/labeling_guide.md`` format:

.. code-block:: json

    [{"id": "rwf-0001", "window_start": 42, "window_end": 72,
      "label": 1, "annotator": "u1", "confidence": 0.9}]

A mapping ``{"rwf-0001": "aggressive", ...}`` (video id -> label) is also
accepted. The labeled dataset is written back to CSV or Parquet.

Usage:
    uv run python scripts/label_dataset.py --input datasets/processed/features.parquet
    uv run python scripts/label_dataset.py --input features.csv \\
        --annotations datasets/annotations/labels.json --output labeled.parquet
"""

from __future__ import annotations

import argparse
import json

from safewatch.behavior.dataset import assign_labels, load_dataset, save_dataset
from safewatch.behavior.label_schema import LABEL_NAMES
from safewatch.core.schemas.dataset import Annotation, DatasetRecord
from safewatch.utils.logging import configure_logging, get_logger


def _load_annotations(
    path: str | None,
) -> list[Annotation] | dict[str, object] | None:
    if path is None:
        return None
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    if isinstance(raw, dict):
        return {str(key): value for key, value in raw.items()}
    if isinstance(raw, list):
        return [Annotation.from_dict(item) for item in raw]
    raise ValueError(
        f"unexpected annotations format in {path}; expected list or mapping"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="unlabeled CSV/Parquet dataset")
    parser.add_argument("--annotations", default=None)
    parser.add_argument("--output", required=True, help="labeled CSV/Parquet output")
    args = parser.parse_args()

    configure_logging(level="INFO", fmt="json")
    logger = get_logger("safewatch.label")

    records = load_dataset(args.input)
    if not records:
        logger.warning(
            "input dataset is empty", extra={"safewatch_extra": {"path": args.input}}
        )
    annotations = _load_annotations(args.annotations)
    if annotations is not None:
        records = assign_labels(records, annotations)
    labeled = sum(1 for record in records if record.label is not None)
    save_dataset(records, args.output)
    distribution = {
        LABEL_NAMES[label]: count
        for label, count in sorted(
            _label_counts(records).items(),
            key=lambda item: item[0].value,
        )
    }
    logger.info(
        "labeling complete",
        extra={
            "safewatch_extra": {
                "input": args.input,
                "output": args.output,
                "records": len(records),
                "labeled": labeled,
                "distribution": distribution,
            }
        },
    )


def _label_counts(records: list[DatasetRecord]) -> dict[object, int]:
    counts: dict[object, int] = {}
    for record in records:
        counts[record.label] = counts.get(record.label, 0) + 1
    return counts


if __name__ == "__main__":
    main()
