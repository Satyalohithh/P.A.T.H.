#!/usr/bin/env python3
"""Build the raw-dataset index table (``datasets/manifest.csv``).

Scans the provided raw sources under a raw root (default ``datasets/raw``),
maps each clip to its unified 3-class label, and writes one row per clip with
portable (relative) video paths. ``--no-probe`` skips the ffmpeg-free
OpenCV duration/fps probe (kept for metadata only, never required).

Usage:
    uv run python scripts/build_manifest.py \\
        --raw datasets/raw \\
        --output datasets/manifest.csv \\
        --sources rwf2000,hockey_fight
"""

from __future__ import annotations

import argparse

from safewatch.data.adapters import enabled_sources
from safewatch.data.manifest import build_manifest
from safewatch.utils.logging import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw",
        default="datasets/raw",
        help="raw dataset root (default: datasets/raw)",
    )
    parser.add_argument(
        "--output",
        default="datasets/manifest.csv",
        help="manifest output path (default: datasets/manifest.csv)",
    )
    parser.add_argument(
        "--sources",
        default=", ".join(enabled_sources()),
        help="comma-separated source names (default: all enabled sources)",
    )
    parser.add_argument(
        "--limit-per-source",
        type=int,
        default=None,
        help="cap the number of clips indexed per source",
    )
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="skip duration/fps probing (metadata only)",
    )
    args = parser.parse_args()

    configure_logging(level="INFO", fmt="json")
    logger = get_logger("safewatch.data.build_manifest")

    sources = [source.strip() for source in args.sources.split(",") if source.strip()]
    manifest = build_manifest(
        args.raw,
        sources,
        args.output,
        limit_per_source=args.limit_per_source,
        probe=not args.no_probe,
    )
    label_counts = manifest["label"].value_counts().to_dict()
    summary: dict[str, object] = {
        "clips": len(manifest),
        "sources": sorted(manifest["source_dataset"].unique().tolist()),
        "label_counts": {str(k): int(v) for k, v in label_counts.items()},
    }
    if not args.no_probe and "duration_s" in manifest.columns:
        summary["total_duration_s"] = float(manifest["duration_s"].sum(skipna=True))
    logger.info("manifest built", extra={"safewatch_extra": summary})


if __name__ == "__main__":
    main()
