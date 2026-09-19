# ML-базовый образ: добавляет torch/transformers для retrieval (энкодер SigLIP 2 на MPS/CPU).
FROM svoe-base:latest

USER root
COPY deploy/docker/requirements-ml.txt ./requirements-ml.txt
RUN pip install -r requirements-ml.txt
USER svoe

ENV PYTHONPATH=/app
