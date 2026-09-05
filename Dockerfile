# Multi-stage development/inference image for SafeWatch AI.
# The thin inference stage ships only the bundled model and API runtime.
# NOTE: final weights/checkpoints are mounted or downloaded at runtime.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml setup.cfg README.md ./
COPY src ./src

RUN pip install --no-cache-dir . --no-deps 2>/dev/null || true

COPY configs ./configs
COPY deployment/nginx/nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 8000

CMD ["uvicorn", "safewatch.api.app:app", "--host", "0.0.0.0", "--port", "8000"]