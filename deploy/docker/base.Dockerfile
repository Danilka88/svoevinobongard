# Базовый образ сервисов: общие python-зависимости без ML-тяжестей.
# Собирается отдельно: make images (или docker build -f deploy/docker/base.Dockerfile -t svoe-base:latest .)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# libgl1/libglib2.0-0 нужны opencv-headless; curl — для healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/docker/requirements-base.txt ./requirements-base.txt
RUN pip install -r requirements-base.txt

# non-root пользователь
RUN useradd --create-home --uid 10001 svoe
USER svoe

# PYTHONPATH=/app делает пакеты packages/* и services/* импортируемыми
ENV PYTHONPATH=/app
