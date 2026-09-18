# Olist MLOps

An end-to-end MLOps project built as part of the MLOps Training 2026/2027.

The project uses the Brazilian E-Commerce Public Dataset by Olist to develop a machine learning system for predicting whether an order will be delivered **late** or **on time**.

## Project Status

| Task | Description | Status |
|------|-------------|--------|
| Task 1 | Get the Data Into a Database | ✅ COMPLETED |
| Task 2 | Exploratory Analysis & Modelling Notebooks | ✅ COMPLETED |
| Task 3 | From Notebooks to Production | ✅ COMPLETED |

---

## Task 3: Production Inference Service

A production-grade REST API inference service built from the Task 2 notebooks.

### Architecture

```
olist-mlops/
├── app/                         # FastAPI application
│   ├── main.py                  #   App factory + lifespan handler
│   ├── routes.py                #   /predict, /health, /model-info, /metrics
│   ├── schemas.py               #   Pydantic request/response models
│   └── monitoring.py            #   Prometheus middleware
├── src/                         # Business logic (pure Python)
│   ├── predict.py               #   PredictionService (load → engineer → preprocess → infer)
│   ├── features.py              #   Feature engineering (haversine, lags, temporal)
│   ├── preprocess.py            #   Preprocessor wrapper
│   ├── validation.py            #   Great Expectations input firewall
│   ├── data.py                  #   Batch CSV loader
│   ├── registry.py              #   MLflow model registry integration
│   └── monitor.py               #   Drift tracking & alert evaluation
├── config/
│   ├── config.yaml              #   All service configuration (no magic constants)
│   ├── settings.py              #   Pydantic settings loaded from config.yaml + .env
│   └── logging_config.py        #   Structured JSON logging
├── models/                      # DVC-tracked model artifacts
│   ├── best_model.joblib         #   Trained classifier
│   ├── preprocessing_pipeline.joblib
│   └── feature_names.txt
├── tests/                       # Pytest test suite (22 tests, 100% pass)
│   ├── conftest.py
│   ├── test_api.py              #   API integration tests
│   ├── test_data.py             #   Schema & leakage-firewall tests
│   ├── test_features.py         #   Feature engineering unit tests
│   ├── test_model.py            #   Model contract & reproducibility tests
│   └── test_preprocess.py       #   Preprocessor unit tests
├── notebooks/                   # Root-level notebook copies (from Task 2)
├── requirements/
│   ├── requirements.txt         #   Pinned production dependencies
│   └── requirements-dev.txt     #   Dev/test extras
├── docs/
│   └── monitoring_and_alerting.md  # Alerting policies & runbook
├── .github/workflows/ci.yml     # CI/CD: lint → test → Docker build
├── Dockerfile                   # Multi-stage production Docker image
├── docker-compose.yml           # Local service stack
└── .pre-commit-config.yaml      # black + flake8 pre-commit hooks
```

### Quick Start

#### Prerequisites

- Python 3.9+
- Docker & Docker Compose (for containerised deployment)

#### 1. Clone & Setup Environment

```bash
git clone <repo-url>
cd olist-mlops
cp .env.example .env
# Edit .env if needed (defaults work out-of-the-box for local dev)
```

#### 2. Install Dependencies

```bash
pip install -r requirements/requirements.txt
pip install -r requirements/requirements-dev.txt
```

#### 3. Run Locally (Python)

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Visit: [http://localhost:8000/docs](http://localhost:8000/docs)

#### 4. Run with Docker Compose

```bash
docker-compose up --build
```

Service: `http://localhost:8000` | Prometheus: `http://localhost:9090`

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (liveness + readiness) |
| `/model-info` | GET | Model version, features, training metrics |
| `/predict` | POST | Single order late-delivery prediction |
| `/predict/batch` | POST | Batch predictions (CSV upload) |
| `/metrics` | GET | Prometheus scrape endpoint |
| `/docs` | GET | Interactive Swagger UI |

### Single Prediction Example

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "order_id": "abc123",
    "order_purchase_timestamp": "2018-06-01T10:00:00",
    "payment_value": 150.0,
    "freight_value": 20.0,
    "price": 130.0,
    "product_weight_g": 500,
    "product_length_cm": 20,
    "product_height_cm": 10,
    "product_width_cm": 15,
    "customer_state": "SP",
    "seller_state": "SP",
    "product_category_name": "cama_mesa_banho",
    "payment_type": "credit_card",
    "payment_installments": 3,
    "review_score": 4,
    "customer_lat": -23.5489,
    "customer_lng": -46.6388,
    "seller_lat": -23.5505,
    "seller_lng": -46.6333,
    "estimated_delivery_days": 12.0
  }'
```

### Running Tests

```bash
# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov=app --cov-report=term-missing
```

### Data & Model Versioning (DVC)

Model artifacts are version-tracked with DVC:

```bash
dvc pull          # Pull latest model artifacts
dvc push          # Push updated artifacts to remote
```

### Pre-commit Hooks

```bash
pre-commit install        # Install hooks
pre-commit run --all-files  # Run manually
```

---

## Dataset

Brazilian E-Commerce Public Dataset by Olist.

The raw dataset is not stored in this repository. See `tasks/task-01-database` for setup instructions.

## Goal

Build an end-to-end machine learning and MLOps workflow around late delivery prediction.
