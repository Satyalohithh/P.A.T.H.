#!/usr/bin/env python3
"""Train a behavior classifier.

Usage: uv run python scripts/train_model.py --config configs/training/train_phase1.yaml
"""

from __future__ import annotations

import argparse

from safewatch.training.trainer import Trainer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training/train_phase1.yaml")
    args = parser.parse_args()
    Trainer(args.config).train()


if __name__ == "__main__":
    main()
