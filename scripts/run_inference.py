#!/usr/bin/env python3
"""Run realtime or offline inference over a stream or file of video.

Usage: uv run python scripts/run_inference.py --source cam-01 --profile realtime
"""

from __future__ import annotations

import argparse


def run(source: str, profile: str) -> None:
    raise NotImplementedError("TODO(implementation): run")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="cam-01")
    parser.add_argument("--profile", default="realtime", choices=["realtime", "batch"])
    args = parser.parse_args()
    run(args.source, args.profile)


if __name__ == "__main__":
    main()
