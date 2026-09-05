#!/usr/bin/env python3
"""Compute feature vectors for a processed pose store.

Usage: uv run python scripts/prepare_features.py --poses datasets/processed/poses.jsonl
"""

from __future__ import annotations

import argparse


def prepare(poses_path: str, output_path: str) -> None:
    raise NotImplementedError("TODO(implementation): prepare")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poses", default="datasets/processed/poses.jsonl")
    parser.add_argument("--output", default="datasets/processed/features.parquet")
    args = parser.parse_args()
    prepare(args.poses, args.output)


if __name__ == "__main__":
    main()
