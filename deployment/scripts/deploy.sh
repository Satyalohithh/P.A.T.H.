#!/usr/bin/env bash
# Deploy the full stack with production compose.
set -euo pipefail

COMPOSE_FILE="$(dirname "$0")/../docker-compose.prod.yaml"
ENV_FILE="$(dirname "$0")/../../.env"

echo "==> Building production images"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" build

echo "==> Starting stack"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d

echo "==> Running database migrations"
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" exec api \
  alembic -c src/safewatch/storage/migrations/alembic.ini upgrade head

echo "Done."