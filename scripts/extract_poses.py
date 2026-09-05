#!/usr/bin/env python3
"""Extract per-track pose keypoints from raw video into dataset JSONL.

Usage: uv run python scripts/extract_poses.py --config configs/inference/inference_batch.yaml
"""

from __future__ import annotations

import argparse

from safewatch.core.schemas.pose import PoseResult


def extract(input_dir: str, output_path: str) -> list[PoseResult]:
    raise NotImplementedError("TODO(implementation): extract")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference/inference_batch.yaml")
    parser.add_argument("--output", default="datasets/processed/poses.jsonl")
    args = parser.parse_args()
    extract(args.config, args.output)


if __name__ == "__main__":
    main()
