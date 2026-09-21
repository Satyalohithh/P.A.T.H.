#!/usr/bin/env python3
"""Write dataset statistics for a manifest to a JSON report.

Reports total clips, per-class counts, available/missing unified classes,
per-source distribution, a source-by-class crosstab, and (when probed) a
duration histogram. Mirrors ``safewatch.data.statistics.compute_statistics``.

Usage:
    uv run python scripts/dataset_statistics.py \\
        --manifest datasets/manifest.csv \\
        --output reports/dataset_statistics.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from safewatch.data.manifest import load_manifest
from safewatch.data.statistics import write_statistics
from safewatch.utils.logging import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="datasets/manifest.csv",
        help="manifest CSV (default: datasets/manifest.csv)",
    )
    parser.add_argument(
        "--output",
        default="reports/dataset_statistics.json",
        help="statistics JSON path (default: reports/dataset_statistics.json)",
    )
    args = parser.parse_args()

    configure_logging(level="INFO", fmt="json")
    logger = get_logger("safewatch.data.statistics")

    manifest: pd.DataFrame = load_manifest(args.manifest)
    stats = write_statistics(manifest, Path(args.output))
    logger.info(
        "dataset statistics written",
        extra={
            "safewatch_extra": {
                "clips": stats["total_clips"],
                "available_classes": stats["available_classes"],
                "missing_classes": stats["missing_classes"],
                "output": str(args.output),
            }
        },
    )


if __name__ == "__main__":
    main()
