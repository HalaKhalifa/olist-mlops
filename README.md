# Olist MLOps

An end-to-end MLOps project built as part of the MLOps Training 2026/2027.

The project uses the Brazilian E-Commerce Public Dataset by Olist to predict whether an order will be delivered **late** or **on time**.

## Project Status

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Get the Data Into a Database | ✅ COMPLETED |
| Task 2 | Exploratory Analysis & Modelling Notebooks | ✅ COMPLETED |
| Task 3 | From Notebooks to Production | ✅ COMPLETED |

---

## Task 3: Production Inference Service

Training stays in the Task 2 notebooks. This service only runs **inference**: it loads the frozen preprocessor and model, validates a new order, and returns `late` / `on_time` with a probability.

### Architecture

```
olist-mlops/
├── app/                         # FastAPI application
│   ├── main.py                  #   App factory + lifespan handler
│   ├── routes.py                #   /predict, /health, /model-info, /metrics, /monitoring
│   ├── schemas.py               #   Pydantic request/response models
│   └── monitoring.py            #   Prometheus middleware
├── src/                         # Inference modules (no training)
│   ├── predict.py               #   CLI + PredictionService
│   ├── features.py              #   Haversine, lags, temporal features
│   ├── preprocess.py            #   Frozen ColumnTransformer (never re-fit)
│   ├── validation.py            #   Great Expectations firewall (reject on failure)
│   ├── data.py                  #   Payload → DataFrame
│   ├── registry.py              #   MLflow tracking + model registry
│   └── monitor.py               #   Drift / latency alerts from prediction logs
├── config/
│   ├── config.yaml              #   Paths, validation domains, alert thresholds
│   ├── settings.py              #   YAML + environment variables
│   └── logging_config.py        #   Console + rotating file logs
├── models/                      # DVC-versioned artifacts used at inference
├── data/                        # Sample payloads (DVC-versioned)
├── tests/                       # pytest: unit, data, model, API
├── monitoring/prometheus.yml    # Prometheus scrape config
├── requirements/                # Runtime vs development pins
├── Dockerfile
├── docker-compose.yml
└── .github/workflows/ci.yml
```

### Quick start (clean machine)

Prerequisites: Python 3.9+, Docker, Docker Compose.

```bash
git clone <repo-url>
cd olist-mlops
cp .env.example .env
# Set DB_PASSWORD in .env. Do not commit .env.

docker compose up --build
```

That single command starts PostgreSQL, MLflow artifact storage, the API, and Prometheus.

| Service | URL |
|---------|-----|
| API | http://localhost:8000 |
| Swagger docs | http://localhost:8000/docs |
| MLflow | http://localhost:5000 |
| Prometheus | http://localhost:9090 |

Optional: register the champion model into MLflow after the stack is up:

```bash
docker compose --profile register run --rm register-model
```

The API first tries `models:/olist-late-delivery-model/Production` and falls back to `models/best_model.joblib` inside the image / volume.

### Local Python (without Docker)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-dev.txt
pre-commit install

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

CLI inference:

```bash
PYTHONPATH=. python -m src.predict --input data/sample_order.json
PYTHONPATH=. python -m src.monitor
```

### API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness + whether the model loaded |
| `/model-info` | GET | Name, version, stage, metrics, feature count |
| `/predict` | POST | Single order |
| `/predict/batch` | POST | Batch of orders |
| `/monitoring` | GET | Drift / latency summary from prediction logs |
| `/metrics` | GET | Prometheus scrape endpoint |
| `/docs` | GET | OpenAPI / Swagger |

Validation policy: Great Expectations runs **before** the model. Failures are **rejected** with HTTP 422 (they are not imputed through to a silent default). Missing coordinates and optional timestamps are allowed; the frozen imputers handle them.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "9e8835f613d3d61b5c4aeae2550b893a",
    "order_purchase_timestamp": "2017-10-18 16:42:42",
    "order_approved_at": "2017-10-18 16:56:45",
    "order_delivered_carrier_date": "2017-10-24 19:22:45",
    "order_estimated_delivery_date": "2017-11-08 00:00:00",
    "customer_state": "RJ",
    "customer_city": "rio de janeiro",
    "customer_zip_code_prefix": "22430",
    "customer_lat": -22.98197,
    "customer_lng": -43.21999,
    "primary_seller_state": "SP",
    "primary_seller_city": "sao paulo",
    "primary_seller_zip_code": 8270.0,
    "seller_lat": -23.56128,
    "seller_lng": -46.46197,
    "primary_product_category": "telephony",
    "dominant_payment_type": "credit_card",
    "item_count": 1.0,
    "total_price": 70.9,
    "avg_item_price": 70.9,
    "total_freight": 14.25,
    "avg_item_freight": 14.25,
    "total_weight_g": 250.0,
    "total_volume_cm3": 1280.0,
    "num_sellers": 1.0,
    "total_payment_value": 85.15,
    "payment_installments_max": 3.0,
    "payment_transactions_count": 1.0
  }'
```

Expected shape of the response: `prediction`, `label`, `late_probability`, `model_version`, `latency_ms`.

On the sample order above the pipeline is checked against the notebook result: class `0` / `on_time`, late probability `0.4304`.

### Tests

```bash
pytest tests/ -v
pytest tests/ --cov=src --cov=app --cov-report=term-missing
```

A failing test stops CI. Break the payload (negative `total_price`, leakage column `is_late`, invalid state) and the service returns 422.

### DVC

Artifacts are versioned with DVC. A local remote is configured at `dvc-storage/` so you can run:

```bash
dvc add models/best_model.joblib models/preprocessing_pipeline.joblib
dvc add data/sample_order.json data/sample_batch.json
dvc push
dvc pull
```

Sample payloads and model files are also kept in the working tree so `pytest` and `docker compose` work without a cloud remote.

### Pre-commit and CI/CD

```bash
pre-commit install
pre-commit run --all-files
```

On every push/PR: Black format check → flake8 → pytest. On push to `main`, the production image is built and pushed to GHCR (`ghcr.io/<owner>/<repo>`).

### Monitoring

- Prometheus metrics: request count, latency, error rate, predicted-class distribution.
- Prediction audit log: `logs/predictions.jsonl` (input, output, latency, model version).
- Alert policy: `docs/monitoring_and_alerting.md`.

---

## Dataset

The raw Olist dump is not stored here. See `tasks/task-01-database` for database setup.

## Goal

Build an end-to-end machine learning and MLOps workflow around late delivery prediction.
