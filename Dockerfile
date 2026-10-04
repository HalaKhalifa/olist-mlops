# Production Dockerfile for Olist late-delivery inference
# Multi-stage, no notebooks, no test/dev tooling

FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/requirements.txt requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt


FROM python:3.10-slim AS runner

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs /app/models /mlflow/artifacts && \
    chown -R appuser:appuser /app /mlflow

COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser data/ /app/data/
COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser src/ /app/src/

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
