#!/usr/bin/env bash
# Download RLVS (real-life violence sequences) into datasets/raw/rlvs.
set -euo pipefail
TARGET="${1:-datasets/raw/rlvs}"
mkdir -p "$TARGET"
echo "TODO(implementation): download RLVS into $TARGET"
echo "Register the download with DVC: dvc add $TARGET"