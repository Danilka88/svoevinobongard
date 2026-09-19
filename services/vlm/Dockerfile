# VLM — извлечение полей этикетки через LLM Router (провайдеры на хосте).
FROM svoe-base:latest

WORKDIR /app
COPY --chown=svoe:svoe packages ./packages
COPY --chown=svoe:svoe config ./config
COPY --chown=svoe:svoe services/__init__.py ./services/__init__.py
COPY --chown=svoe:svoe services/vlm ./services/vlm

ENV PYTHONPATH=/app
EXPOSE 8083

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8083/health || exit 1

CMD ["uvicorn", "services.vlm.app.main:app", "--host", "0.0.0.0", "--port", "8083"]
