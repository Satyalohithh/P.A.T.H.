# Deployment

Status: draft.

## Topologies

- **Realtime**: single-stream edge node — YOLOv8n + ByteTrack + XGBoost; ONNX
  exported, optional TensorRT; budget 33.3 ms/frame.
- **Batch**: offline analysis of recorded video — YOLOv8m + BiLSTM-attention;
  higher accuracy, no latency constraint.
- **Central**: multiple streams terminating at the API + dashboard.

## Docker

`Dockerfile` builds an API image; `docker-compose.yml` adds Postgres and a
frontend dev container. Production compose in `deployment/docker-compose.prod.yaml`.

## Migration

Alembic migrations under `src/safewatch/storage/migrations/`; run via
`alembic upgrade head` with `SAFEWATCH_DATABASE_URL` set.

## Nginx

Reverse proxy config at `deployment/nginx/nginx.conf` (WebSocket upgrade
headers for `/ws`).