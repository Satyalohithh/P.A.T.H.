#!/usr/bin/env python3
"""Split a manifest into stratified video-level train/val/test JSON.

Stratification is by unified label at the clip level (every clip lands in
exactly one split), per ``docs/dataset_strategy.md``. Output uses the existing
``datasets/splits/{train,val,test}.json`` schema.

Usage:
    uv run python scripts/split_dataset.py \\
        --manifest datasets/manifest.csv \\
        --output-dir datasets/splits \\
        --val-size 0.15 --test-size 0.15 --seed 42
"""

from __future__ import annotations

import argparse

import pandas as pd

from safewatch.data.manifest import load_manifest
from safewatch.data.split import split_manifest
from safewatch.utils.logging import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        default="datasets/manifest.csv",
        help="manifest CSV (default: datasets/manifest.csv)",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/splits",
        help="directory for train/val/test JSON (default: datasets/splits)",
    )
    parser.add_argument(
        "--val-size", type=float, default=0.15, help="validation fraction"
    )
    parser.add_argument("--test-size", type=float, default=0.15, help="test fraction")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    args = parser.parse_args()

    configure_logging(level="INFO", fmt="json")
    logger = get_logger("safewatch.data.split")

    manifest: pd.DataFrame = load_manifest(args.manifest)
    result = split_manifest(
        manifest,
        args.output_dir,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )
    logger.info(
        "video-level split written",
        extra={
            "safewatch_extra": {
                "counts": dict(result.counts),
                "output_dir": str(args.output_dir),
            }
        },
    )


if __name__ == "__main__":
    main()
