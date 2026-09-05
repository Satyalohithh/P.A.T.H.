#!/usr/bin/env python3
"""Benchmark pipeline stage latencies against the realtime budget.

Usage: uv run python scripts/benchmark_pipeline.py --frames 300
"""

from __future__ import annotations

import argparse


def benchmark(num_frames: int) -> dict[str, float]:
    raise NotImplementedError("TODO(implementation): benchmark")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=300)
    args = parser.parse_args()
    benchmark(args.frames)


if __name__ == "__main__":
    main()
