# Production Dockerfile for Olist Late Delivery Inference Service
# Minimal, secure, multi-stage build without notebooks or heavy development tools

FROM python:3.9-slim as builder

WORKDIR /build

# Install build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/requirements.txt requirements.txt
RUN pip install --no-cache-dir --user -r requirements.txt


# Final runtime image
FROM python:3.9-slim as runner

WORKDIR /app

# Install curl for docker healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed wheels from builder to non-root location
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/logs && \
    chown -R appuser:appuser /app

# Copy only production application code and models (no notebooks, no dev files)
COPY --chown=appuser:appuser app/ /app/app/
COPY --chown=appuser:appuser config/ /app/config/
COPY --chown=appuser:appuser data/ /app/data/
COPY --chown=appuser:appuser models/ /app/models/
COPY --chown=appuser:appuser src/ /app/src/

USER appuser

# Expose FastAPI service port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=20s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
