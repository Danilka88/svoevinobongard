FROM python:3.11-slim

WORKDIR /app

COPY services/gateway/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY services/gateway/app ./app
COPY packages ./packages

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
