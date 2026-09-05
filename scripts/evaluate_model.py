#!/usr/bin/env python3
"""Evaluate a trained model on the held-out test split.

Usage: uv run python scripts/evaluate_model.py --checkpoint models/checkpoints/phase2/best.pt
"""

from __future__ import annotations

import argparse


def evaluate(checkpoint_path: str, test_split: str) -> dict[str, float]:
    raise NotImplementedError("TODO(implementation): evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test-split", default="datasets/splits/test.json")
    args = parser.parse_args()
    evaluate(args.checkpoint, args.test_split)


if __name__ == "__main__":
    main()
