"""FastAPI route endpoints for Olist MLOps inference service."""

import time
from typing import Any, Dict
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import RedirectResponse, Response

from app.monitoring import (
    PREDICTED_PROBABILITY_HISTOGRAM,
    PREDICTION_ERRORS_TOTAL,
    PREDICTION_LATENCY_SECONDS,
    PREDICTION_REQUESTS_TOTAL,
    PREDICTIONS_BY_CLASS,
    get_metrics_response,
)
from app.schemas import (
    BatchOrderInput,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    MonitoringSummaryResponse,
    OrderInput,
    PredictionResponse,
)
from config.logging_config import logger
from config.settings import settings
from src.monitor import model_monitor
from src.predict import prediction_service
from src.preprocess import preprocessor_service
from src.utils import load_json
from src.validation import data_validator

router = APIRouter()


@router.get("/", include_in_schema=False)
def api_root() -> RedirectResponse:
    """Send browser visitors to the interactive API documentation."""
    return RedirectResponse(url="/docs")


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check endpoint",
    tags=["System"],
)
def health_check() -> HealthResponse:
    """Verify service readiness and model availability."""
    model_loaded = False
    try:
        _ = prediction_service.model
        model_loaded = True
    except Exception as e:
        logger.error(f"Health check failed to load model: {e}")

    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        service=settings.project.name,
        version=settings.project.version,
        model_loaded=model_loaded,
        environment=settings.environment,
    )


@router.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Model metadata and performance information",
    tags=["System"],
)
def get_model_info() -> ModelInfoResponse:
    """Return model type, hyperparameters, evaluation metrics, and feature count."""
    _ = prediction_service.model
    metrics_data: Dict[str, Any] = {}
    if settings.model.metrics_path.exists():
        metrics_data = load_json(settings.model.metrics_path)

    total_features = len(preprocessor_service.feature_names)

    return ModelInfoResponse(
        model_name=settings.model.name,
        model_version=prediction_service.model_version,
        model_stage=prediction_service.model_stage,
        model_type=metrics_data.get("model", "RandomForestClassifier"),
        total_features=total_features,
        hyperparameters=metrics_data.get("best_params", {}),
        validation_metrics=metrics_data.get("validation", {}),
        test_metrics=metrics_data.get("test", {}),
    )


@router.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Predict late delivery for a single order",
    tags=["Inference"],
)
def predict_single_order(order: OrderInput) -> PredictionResponse:
    """Predict whether a new order will be delivered late or on time.

    Validates data expectations with Great Expectations before running inference.
    Rejects bad payloads with HTTP 422.
    """
    order_dict = order.model_dump()

    # Step 4: Data Validation before reaching model
    is_valid, errors = data_validator.validate(order_dict)
    if not is_valid:
        PREDICTION_ERRORS_TOTAL.labels(error_type="validation").inc()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "message": "Incoming order failed Great Expectations validation checks.",
                "errors": errors,
            },
        )

    try:
        # Step 2 & 3: Run inference pipeline with latency tracking and logging
        result = prediction_service.predict_single(order_dict)

        # Step 10: Prometheus metrics recording
        PREDICTION_REQUESTS_TOTAL.labels(type="single").inc()
        PREDICTIONS_BY_CLASS.labels(predicted_class=result["label"]).inc()
        PREDICTION_LATENCY_SECONDS.observe(result["latency_ms"] / 1000.0)
        PREDICTED_PROBABILITY_HISTOGRAM.observe(result["late_probability"])

        return PredictionResponse(**result)

    except Exception as e:
        PREDICTION_ERRORS_TOTAL.labels(error_type="inference").inc()
        logger.error(f"Inference execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference execution error: {str(e)}",
        )


@router.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Batch predict late delivery for multiple orders",
    tags=["Inference"],
)
def predict_batch_orders(batch: BatchOrderInput) -> BatchPredictionResponse:
    """Predict late delivery for multiple orders in a single request."""
    orders = [order.model_dump() for order in batch.orders]

    # Validate each order in batch
    for i, order_dict in enumerate(orders):
        is_valid, errors = data_validator.validate(order_dict)
        if not is_valid:
            PREDICTION_ERRORS_TOTAL.labels(error_type="validation_batch").inc()
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "message": f"Order at index {i} failed validation checks.",
                    "errors": errors,
                },
            )

    start_time = time.perf_counter()
    try:
        results = prediction_service.predict_batch(orders)
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0

        PREDICTION_REQUESTS_TOTAL.labels(type="batch").inc()
        for r in results:
            PREDICTIONS_BY_CLASS.labels(predicted_class=r["label"]).inc()
            PREDICTED_PROBABILITY_HISTOGRAM.observe(r["late_probability"])
        PREDICTION_LATENCY_SECONDS.observe(total_latency_ms / 1000.0)

        return BatchPredictionResponse(
            predictions=[PredictionResponse(**r) for r in results],
            total_orders=len(orders),
            total_latency_ms=round(total_latency_ms, 2),
        )

    except Exception as e:
        PREDICTION_ERRORS_TOTAL.labels(error_type="inference_batch").inc()
        logger.error(f"Batch inference execution failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch inference error: {str(e)}",
        )


@router.get(
    "/monitoring",
    response_model=MonitoringSummaryResponse,
    summary="Prediction-log drift and latency summary",
    tags=["Monitoring"],
)
def monitoring_summary() -> MonitoringSummaryResponse:
    """Summarize stored prediction logs for drift and latency alerts."""
    return MonitoringSummaryResponse(**model_monitor.compute_summary_metrics())


@router.get(
    "/metrics",
    summary="Prometheus service and inference metrics",
    tags=["Monitoring"],
)
def prometheus_metrics() -> Response:
    """Expose Prometheus metrics endpoint."""
    return get_metrics_response()
