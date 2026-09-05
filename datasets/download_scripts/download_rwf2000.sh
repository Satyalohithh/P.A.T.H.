#!/usr/bin/env bash
# Download RWF-2000 into datasets/raw/rwf2000.
# See https://github.com/mmaction2/mmaction2 / the RWF-2000 repository for the
# current direct download links.
set -euo pipefail
TARGET="${1:-datasets/raw/rwf2000}"
mkdir -p "$TARGET"
echo "TODO(implementation): download RWF-2000 into $TARGET"
echo "Register the download with DVC: dvc add $TARGET"