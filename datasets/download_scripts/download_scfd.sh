#!/usr/bin/env bash
# Download SCFD (surveillance camera face dataset component) into datasets/raw/scfd.
set -euo pipefail
TARGET="${1:-datasets/raw/scfd}"
mkdir -p "$TARGET"
echo "TODO(implementation): download SCFD into $TARGET"
echo "Register the download with DVC: dvc add $TARGET"