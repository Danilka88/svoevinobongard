# Gateway — единственная публичная точка входа.
FROM svoe-base:latest

WORKDIR /app
COPY --chown=svoe:svoe packages ./packages
COPY --chown=svoe:svoe config ./config
COPY --chown=svoe:svoe services/__init__.py ./services/__init__.py
COPY --chown=svoe:svoe services/gateway ./services/gateway

ENV PYTHONPATH=/app \
    GATEWAY_PORT=8080
EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8080/v1/health || exit 1

CMD ["uvicorn", "services.gateway.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
