#!/usr/bin/env bash
# Download DVD (documentary violence dataset) into datasets/raw/dvd.
set -euo pipefail
TARGET="${1:-datasets/raw/dvd}"
mkdir -p "$TARGET"
echo "TODO(implementation): download DVD into $TARGET"
echo "Register the download with DVC: dvc add $TARGET"