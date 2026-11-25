# CareFlow API + chatbot UI. Build: docker build -t careflow .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.lock.txt ./
RUN pip install -r requirements.lock.txt

COPY . .

# The index is rebuilt from the committed corpus on each start (idempotent; a few embedding calls).
# Render sets $PORT. Secrets come from environment variables, never from the image.
CMD ["sh", "-c", "python -m careflow.m4_rag.ingest && exec uvicorn careflow.m8_service.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
