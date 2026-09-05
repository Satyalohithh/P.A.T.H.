#!/usr/bin/env python3
"""Download public datasets (RWF-2000, RLVS, SCFD, DVD).

Usage: uv run python scripts/download_datasets.py --datasets rwf2000,rlvs
"""

from __future__ import annotations

import argparse
from pathlib import Path


def download(dataset: str, target_dir: Path) -> None:
    raise NotImplementedError("TODO(implementation): download")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="rwf2000",
        help="Comma-separated dataset names: rwf2000,rlvs,scfd,dvd",
    )
    parser.add_argument("--target", default="datasets/raw", help="Download target dir")
    args = parser.parse_args()
    for dataset in args.datasets.split(","):
        download(dataset, Path(args.target))


if __name__ == "__main__":
    main()
