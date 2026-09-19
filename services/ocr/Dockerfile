# OCR — дешёвый текстовый сигнал (RapidOCR, Apache-2.0).
FROM svoe-base:latest

WORKDIR /app
COPY --chown=svoe:svoe packages ./packages
COPY --chown=svoe:svoe services/__init__.py ./services/__init__.py
COPY --chown=svoe:svoe services/ocr ./services/ocr

ENV PYTHONPATH=/app
EXPOSE 8085

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=5 \
  CMD curl -fsS http://127.0.0.1:8085/health || exit 1

CMD ["uvicorn", "services.ocr.app.main:app", "--host", "0.0.0.0", "--port", "8085"]
