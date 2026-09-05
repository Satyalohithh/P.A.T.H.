# API reference

Status: draft.

Base path: `/api/v1`. WebSocket: `/ws`.

## REST

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/streams` | List registered streams |
| POST | `/api/v1/streams` | Register a stream |
| DELETE | `/api/v1/streams/{id}` | Stop a stream |
| GET | `/api/v1/alerts` | Alert history (filter by stream) |
| POST | `/api/v1/alerts/{id}/ack` | Acknowledge an alert |
| GET | `/api/v1/analytics/behavior-distribution` | Class histogram |
| GET | `/api/v1/analytics/alert-velocity` | Alerts per time window |

## WebSocket

Server pushes `AlertEvent` JSON on new alerts; client may subscribe to
`/ws?stream=<id>`.

## Auth

`configs/api/server_config.yaml` `auth.mode`: `disabled` (dev), `jwt`,
`api_key`. Protected routes require `Authorization: Bearer <token>`.