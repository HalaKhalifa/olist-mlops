"""Prometheus monitoring metrics for Olist MLOps service."""

import time
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# 1. Service-level metrics: Request count, error rate, latency
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total count of HTTP requests processed by endpoint and status",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency histogram in seconds",
    ["method", "endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

# 2. Prediction-level metrics: Predictions count, distribution, latency
PREDICTION_REQUESTS_TOTAL = Counter(
    "prediction_requests_total",
    "Total count of prediction requests",
    ["type"],  # single vs batch
)

PREDICTIONS_BY_CLASS = Counter(
    "predictions_by_class_total",
    "Distribution of predicted classes (on_time vs late)",
    ["predicted_class"],
)

PREDICTION_ERRORS_TOTAL = Counter(
    "prediction_errors_total",
    "Total count of failed predictions or validation errors",
    ["error_type"],
)

PREDICTION_LATENCY_SECONDS = Histogram(
    "prediction_latency_seconds",
    "Inference latency for model execution in seconds",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.5, 1.0),
)

PREDICTED_PROBABILITY_HISTOGRAM = Histogram(
    "predicted_late_probability_distribution",
    "Distribution of predicted late delivery probabilities",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Middleware to track request duration and response status codes."""

    async def dispatch(self, request: Request, call_next):
        method = request.method
        path = request.url.path

        start_time = time.perf_counter()
        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        except Exception:
            status_code = "500"
            raise
        finally:
            duration = time.perf_counter() - start_time
            # Ignore /metrics endpoint itself to avoid self-monitoring loop
            if path != "/metrics":
                HTTP_REQUESTS_TOTAL.labels(
                    method=method, endpoint=path, status_code=status_code
                ).inc()
                HTTP_REQUEST_DURATION_SECONDS.labels(
                    method=method, endpoint=path
                ).observe(duration)

        return response


def get_metrics_response() -> Response:
    """Generate and return Prometheus metrics payload."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
