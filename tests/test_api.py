"""Integration tests for FastAPI REST API routes end to end."""

from starlette import status


def test_api_root_redirects_to_docs(api_client):
    """GET / should guide browser users to Swagger documentation."""
    response = api_client.get("/", follow_redirects=False)
    assert response.status_code == status.HTTP_307_TEMPORARY_REDIRECT
    assert response.headers["location"] == "/docs"


def test_api_health_check(api_client):
    """GET /health should return 200 OK with healthy status and model_loaded True."""
    response = api_client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "version" in data


def test_api_model_info(api_client):
    """GET /model-info should return 200 OK with model metadata and 145 features."""
    response = api_client.get("/model-info")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["model_name"] == "olist-late-delivery-model"
    assert data["model_version"] == "1"
    assert data["total_features"] == 145
    assert "validation_metrics" in data


def test_api_predict_single_success(api_client, sample_order_dict):
    """POST /predict with valid order returns 200 OK with prediction, probability, and version."""
    response = api_client.post("/predict", json=sample_order_dict)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["order_id"] == sample_order_dict["order_id"]
    assert data["prediction"] in (0, 1)
    assert data["label"] in ("late", "on_time")
    assert 0.0 <= data["late_probability"] <= 1.0
    assert data["model_version"] == "1"
    assert "latency_ms" in data


def test_api_predict_validation_failure_rejected(api_client, sample_order_dict):
    """POST /predict with invalid data (negative price) must be rejected with 422."""
    bad_order = sample_order_dict.copy()
    bad_order["total_price"] = -50.0

    response = api_client.post("/predict", json=bad_order)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()
    assert "detail" in data


def test_api_predict_leakage_rejected(api_client, sample_order_dict):
    """POST /predict with post-delivery leakage column must be rejected with 422."""
    leaky_order = sample_order_dict.copy()
    leaky_order["order_delivered_customer_date"] = "2017-10-31 21:47:42"

    response = api_client.post("/predict", json=leaky_order)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    data = response.json()
    assert "detail" in data


def test_api_predict_batch_success(api_client, sample_batch_dict):
    """POST /predict/batch should return 200 OK with array of predictions."""
    response = api_client.post("/predict/batch", json=sample_batch_dict)
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["total_orders"] == len(sample_batch_dict["orders"])
    assert len(data["predictions"]) == data["total_orders"]


def test_api_prometheus_metrics(api_client):
    """GET /metrics should expose Prometheus text metrics."""
    response = api_client.get("/metrics")
    assert response.status_code == status.HTTP_200_OK
    assert "http_requests_total" in response.text
    assert "prediction_requests_total" in response.text


def test_api_documentation_accessible(api_client):
    """GET /docs should return 200 OK for OpenAPI Swagger documentation."""
    response = api_client.get("/docs")
    assert response.status_code == status.HTTP_200_OK


def test_api_monitoring_summary(api_client):
    """GET /monitoring should return a drift/latency summary payload."""
    response = api_client.get("/monitoring")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert "total_predictions" in data
    assert "active_alerts" in data


def test_api_invalid_payment_type_rejected(api_client, sample_order_dict):
    """POST /predict with an unknown payment type is rejected by validation."""
    bad_order = sample_order_dict.copy()
    bad_order["dominant_payment_type"] = "bitcoin"

    response = api_client.post("/predict", json=bad_order)
    assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    assert "detail" in response.json()
