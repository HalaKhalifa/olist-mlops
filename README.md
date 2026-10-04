# Olist MLOps: Late Delivery Prediction

[![CI/CD Pipeline](https://github.com/HalaKhalifa/olist-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/HalaKhalifa/olist-mlops/actions/workflows/ci.yml)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED.svg?logo=docker)](https://www.docker.com)

---

## 🎯 Business Problem

In e-commerce, late deliveries directly damage customer satisfaction and increase support costs. Using the **Brazilian E-Commerce Public Dataset by Olist**, this project builds an end-to-end MLOps system to predict whether an order will arrive **late** or **on time** before it reaches the customer, enabling proactive operational decisions.

---

## 📋 Project Roadmap: The 3 Tasks

| Task | Title | Scope & Summary | Deliverables |
| :--- | :--- | :--- | :--- |
| **Task 1** | **Data Ingestion & Storage** | Designed relational database schema in PostgreSQL, set up primary/foreign keys, and loaded the multi-table Olist dataset. | [`tasks/task-01-database/`](tasks/task-01-database/) |
| **Task 2** | **EDA & Model Training** | Performed data cleaning, feature engineering (Haversine distance, temporal lags), handled class imbalance (~8.11% late rate), and trained/tuned ML models. | [`notebooks/`](notebooks/) |
| **Task 3** | **Production Inference Service** | Productionized the model into a containerized FastAPI service with validation firewalls, MLflow registry, Prometheus monitoring, CI/CD, and DVC artifact tracking. | [`app/`](app/), [`src/`](src/), [Task 3 Report](tasks/task-03-production/report/task_3_report.md) |

---

## 🏗️ Repository Structure

```
olist-mlops/
├── app/                         # FastAPI application (routes, schemas, middleware)
├── src/                         # Production inference pipeline (data, features, validation, predict)
├── config/                      # Dynamic configuration and settings (config.yaml, settings.py)
├── models/                      # Serialized models and preprocessor pipelines (DVC-tracked)
├── data/                        # Sample input payloads for testing (DVC-tracked)
├── notebooks/                   # Task 2 exploratory analysis & training notebooks
├── tests/                       # Pytest test suite (unit, data, model, API tests)
├── monitoring/                  # Prometheus scraping configuration
├── requirements/                # Pinned runtime and development dependencies
├── tasks/                       # Task delivery reports & documentation
├── Dockerfile                   # Multi-stage production container
└── docker-compose.yml           # Multi-service stack (Database, MLflow, API, Prometheus)
```

---

## 🚀 Quick Start (Docker)

Start the entire stack (PostgreSQL, MLflow, API, Prometheus) with a single command:

```bash
# 1. Clone repo & create environment file
git clone https://github.com/HalaKhalifa/olist-mlops.git
cd olist-mlops
cp .env.example .env

# 2. Build and launch all services
docker compose up --build
```

### Service Endpoints
- **API Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)
- **MLflow UI**: [http://localhost:5000](http://localhost:5000)
- **Prometheus UI**: [http://localhost:9090](http://localhost:9090)

---

## 💻 Local Setup (Without Docker)

```bash
# 1. Environment & Dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements/requirements.txt -r requirements/requirements-dev.txt
pre-commit install

# 2. Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 3. Start Interactive Showcase Dashboard (Streamlit)
streamlit run app/dashboard.py

# 4. CLI Prediction
PYTHONPATH=. python -m src.predict --input data/sample_order.json
```

---

## 📡 API Endpoints

| Endpoint | Method | Purpose |
| :--- | :---: | :--- |
| `/health` | `GET` | Health status and model readiness |
| `/model-info` | `GET` | Active model version, features, and metadata |
| `/predict` | `POST` | Single order prediction (`late` / `on_time`, probability) |
| `/predict/batch` | `POST` | Batch predictions |
| `/monitoring` | `GET` | Drift summary against baseline ($8.11\%$) |
| `/metrics` | `GET` | Prometheus scrape endpoint |

### Example Request (`POST /predict`)
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
    "primary_seller_state": "SP",
    "total_price": 70.9,
    "total_freight": 14.25,
    "item_count": 1.0
  }'
```

### Example Response
```json
{
  "order_id": "9e8835f613d3d61b5c4aeae2550b893a",
  "prediction": 0,
  "label": "on_time",
  "late_probability": 0.4595,
  "model_name": "olist-late-delivery-model",
  "model_version": "1",
  "latency_ms": 12.48
}
```

---

## 🧪 Testing & CI/CD

```bash
# Run pytest with test coverage
pytest tests/ -v --cov=src --cov=app

# Run linting and format checks
black --check app config src tests
flake8 app config src tests
```

- **CI/CD**: GitHub Actions runs format checks, linting, tests, container smoke tests, and publishes images to GHCR on `main`.

---

## 📊 Monitoring & Alerts

- **Drift Detection**: Compares live prediction late-rate against the $8.11\%$ baseline.
- **Audit Logs**: All requests recorded in `logs/predictions.jsonl` for delayed reconciliation when actual delivery dates arrive.
- **Alert Strategy**: Outlined in [`docs/monitoring_and_alerting.md`](docs/monitoring_and_alerting.md).
