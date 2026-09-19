# Retrieval — энкодер SigLIP 2 + геоверификация. Требует ML-базовый образ и volume с индексом.
FROM svoe-ml-base:latest

WORKDIR /app
COPY --chown=svoe:svoe packages ./packages
COPY --chown=svoe:svoe services/__init__.py ./services/__init__.py
COPY --chown=svoe:svoe services/retrieval ./services/retrieval

ENV PYTHONPATH=/app \
    HF_HOME=/models/hf \
    ENCODER_DEVICE=cpu
EXPOSE 8082

# start-period большой: модель и индекс грузятся при старте (прогрев).
HEALTHCHECK --interval=20s --timeout=5s --start-period=180s --retries=10 \
  CMD curl -fsS http://127.0.0.1:8082/health || exit 1

CMD ["uvicorn", "services.retrieval.app.main:app", "--host", "0.0.0.0", "--port", "8082"]
