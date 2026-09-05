#!/usr/bin/env bash
# Health check for the SafeWatch stack.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"

for i in $(seq 1 30); do
  if curl -sf "$BASE_URL/api/v1/streams" > /dev/null; then
    echo "OK: API responds at $BASE_URL"
    exit 0
  fi
  echo "waiting for API... ($i/30)"
  sleep 2
done

echo "FAIL: API did not become healthy"
exit 1