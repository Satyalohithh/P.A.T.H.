#!/usr/bin/env python3
"""Run a Phase-6 end-to-end experiment from a YAML config.

Lives on top of ``safewatch.data.experiment.run_experiment``: feature
extraction (cached per clip under ``artifacts/features``), windowed dataset,
active-class XGBoost, evaluation snapshots under ``reports/<name>/``, model
under ``models/checkpoints/``, and a generated ``docs/<name>.md`` report.

Usage:
    uv run python scripts/run_experiment.py --config configs/experiment/experiment_v1.yaml
    uv run python scripts/run_experiment.py --config <file> --synthetic --limit-videos 3
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from safewatch.data.experiment import ExperimentConfig, run_experiment
from safewatch.utils.logging import configure_logging, get_logger


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/experiment/experiment_v1.yaml",
        help="experiment YAML config (default: configs/experiment/experiment_v1.yaml)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="run the deterministic synthetic smoke path (no datasets/raw needed)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-extract clips even when a feature cache already exists",
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="require (and reuse) existing feature caches; fail if missing",
    )
    parser.add_argument(
        "--limit-videos",
        type=int,
        default=None,
        help="cap the number of clips processed",
    )
    parser.add_argument(
        "--frames-cap",
        type=int,
        default=None,
        help="override the per-clip frame cap",
    )
    parser.add_argument(
        "--sources",
        default=None,
        help="comma-separated source override (default: config sources)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate inputs and report status without training",
    )
    args = parser.parse_args()

    configure_logging(level="INFO", fmt="json")
    logger = get_logger("safewatch.data.experiment")

    cfg = ExperimentConfig.from_file(args.config)
    if args.frames_cap is not None:
        cfg = replace(cfg, frames_cap=args.frames_cap)
    if args.sources:
        cfg = replace(
            cfg, sources=tuple(s.strip() for s in args.sources.split(",") if s.strip())
        )

    if args.check:
        _check_status(cfg, logger, synthetic=args.synthetic)
        return

    results = run_experiment(
        cfg,
        synthetic=args.synthetic,
        force=args.force,
        limit_videos=args.limit_videos,
        skip_extraction=args.skip_extraction,
    )
    logger.info(
        "experiment complete",
        extra={"safewatch_extra": {"summary": results.summary()}},
    )


def _check_status(cfg: ExperimentConfig, logger: Any, *, synthetic: bool) -> None:
    import pandas as pd

    manifest_path = Path(cfg.manifest)
    if synthetic:
        logger.info(
            "synthetic config OK", extra={"safewatch_extra": {"synthetic": True}}
        )
        return
    if not manifest_path.is_file():
        logger.warning(
            "manifest missing; run scripts/build_manifest.py",
            extra={"safewatch_extra": {"manifest": str(manifest_path)}},
        )
        return
    manifest: pd.DataFrame = pd.read_csv(manifest_path)
    features_dir = Path(cfg.artifacts_dir) / "features"
    cached = 0
    for video_id in manifest["video_path"]:
        safe = str(video_id).replace("\\", "_").replace("/", "__").replace(" ", "_")
        if (features_dir / f"{safe}.parquet").is_file():
            cached += 1
    logger.info(
        "input check",
        extra={
            "safewatch_extra": {
                "manifest": str(manifest_path),
                "clips": len(manifest),
                "feature_cache_count": cached,
            }
        },
    )


if __name__ == "__main__":
    main()
