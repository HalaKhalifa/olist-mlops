# Task 3 Report: From Notebooks to Production

**MLOps Engineering Training 2026/2027 · Olist Late Delivery Prediction**

---

## 1. Objective

Task 3 productionizes the Task 2 late-delivery model as an inference-only service. Training remains in the notebooks; the service accepts a new order and returns the predicted class (`late` or `on_time`), late-delivery probability, model version, and latency.

---

## 2. Comprehensive 10-Step Implementation Details

This section provides the complete step-by-step breakdown of how each required MLOps stage is architected and implemented in the repository.

### Step 1: Repository & Configuration
- **Clean Structure**: The repository is cleanly partitioned into modular packages: `app/` (FastAPI), `src/` (core pipeline), `config/` (configuration), `data/` (sample payloads), `models/` (serialized artifacts), `notebooks/` (Task 2 research), `tests/` (pytest suite), `requirements/` (dependency locks), and `monitoring/` (Prometheus setup).
- **Dynamic Settings**: No paths or operational thresholds are hardcoded. Centralized configuration in `config/config.yaml` is parsed dynamically via Pydantic in `config/settings.py` relative to `PROJECT_ROOT`, with runtime environment variable overrides via `.env`.
- **Pinned Dependencies**: Requirements are segregated into `requirements/requirements.txt` (production inference dependencies locked to exact versions: Python 3.10, scikit-learn 1.0.2, joblib 1.5.1, FastAPI 0.115.14) and `requirements/requirements-dev.txt` (development, test, and linting tools). Python and scikit-learn match the serialized model artifacts.
- **Zero-Setup Documentation**: `README.md` provides clear instructions for starting the full stack with Docker Compose or running locally with `uvicorn` and `pytest`.

### Step 2: Notebooks to Python Modules
- **Modular Refactoring**: The Task 2 notebooks were refactored into focused, single-responsibility modules under `src/`:
  - `src/data.py`: Handles raw dictionary-to-DataFrame conversions and defines schema constants.
  - `src/validation.py`: Data firewall and validation engine.
  - `src/features.py`: Deterministic feature derivations (Haversine distances, temporal components, lag features).
  - `src/preprocess.py`: Transforms features into the exact 145-dimensional matrix using the frozen pipeline.
  - `src/predict.py`: Inference service and prediction orchestration.
- **Frozen Fitted Objects**: Estimators and transformers (`preprocessing_pipeline.joblib`, `best_model.joblib`) are strictly loaded via `src/utils.py` and never re-fitted during inference.
- **Exact Reproducibility**: Implements deterministic parity matching the notebook outputs on the same inputs (verified on `data/sample_order.json`).

### Step 3: Logging & Error Handling
- **Structured Python Logging**: Replaces all `print` statements with standard Python logging (`config/logging_config.py`), outputting structured logs with timestamp, loglevel, logger name, and message.
- **Dual Destination & Log Rotation**: Simultaneous streaming to `stdout` (console) and rotating disk storage (`logs/app.log`, 10MB per file, 5 backup copies).
- **Prediction Audit Logging**: Every inference request logs input attributes, output prediction, latency, and model version to `logs/predictions.jsonl` for delayed evaluation.
- **Resilient Error Handling**: Missing optional values are imputed gracefully by the pipeline, and malformed inputs are intercepted with structured HTTP 422 errors without crashing the service.

### Step 4: Data Versioning & Validation
- **DVC Tracking**: Data payloads (`data/*.json.dvc`) and binary artifacts (`models/*.joblib.dvc`) are versioned with DVC using a local remote (`dvc-storage/`), ensuring complete lineage and auditability.
- **Great Expectations Firewall**: `src/validation.py` defines domain expectations checking numeric bounds (`total_price >= 0`, `total_freight >= 0`, `item_count >= 1`, latitude/longitude bounds), categorical sets (27 Brazilian states, recognized payment types), and geo missingness thresholds.
- **Target Leakage Prevention**: Automatically detects and blocks post-delivery fields (`is_late`, `order_delivered_customer_date`, `actual_delivery_days`, `delivery_delay_days`).
- **Strict Failure Policy**: Malformed or leaky requests are explicitly rejected with HTTP 422 before reaching the model.

### Step 5: Experiment Tracking & Model Registry
- **MLflow Tracking**: `src/registry.py` logs hyperparameters, baseline/validation/test metrics, and artifacts (`metadata/metrics.json`, `preprocessor`, `features`) to MLflow.
- **Model Registry & Promotion**: Model artifacts are registered under `olist-late-delivery-model` and automatically transitioned to the `Production` stage.
- **Decoupled Loading**: In production, `src/predict.py` queries `models:/olist-late-delivery-model/Production` from the MLflow registry, falling back to local versioned artifacts if offline.
- **Container-Accessible Storage**: MLflow runs with shared named volumes (`mlflow_artifacts`) across the Compose network.

### Step 6: Comprehensive Testing Suite
- **Pytest Suite**: Complete test suite organized under `tests/`:
  - Unit tests: `tests/test_features.py` (Haversine math, lags, timestamps), `tests/test_preprocess.py` (shape `(1, 145)`, NaN handling), `tests/test_utils.py`.
  - Data tests: `tests/test_data.py` (Great Expectations rules, schema bounds, target leakage prevention).
  - Model tests: `tests/test_model.py` (loading, contract format, batch array shapes, notebook parity).
  - Integration tests: `tests/test_api.py` (end-to-end API routes: `/health`, `/model-info`, `/predict`, `/predict/batch`, `/monitoring`, `/metrics`).
- **Single Command**: Executable via `pytest tests/` configured via `pytest.ini`.

### Step 7: FastAPI Inference Service
- **FastAPI Architecture**: Implemented in `app/main.py` using factory pattern with async `lifespan` handler preloading model artifacts into memory.
- **Schema Validation**: Strict request/response validation using Pydantic v2 (`app/schemas.py`) with rich examples for Swagger UI.
- **Standardized Endpoints**:
  - `GET /`: Redirects to `/docs`.
  - `GET /health`: Service health and model load state.
  - `GET /model-info`: Model architecture, hyperparameters, version, and performance metrics.
  - `POST /predict`: Single order inference with probability and latency.
  - `POST /predict/batch`: Batch order inference.
  - `GET /monitoring`: Real-time drift and latency summary.
  - `GET /metrics`: Prometheus metric scrape endpoint.

### Step 8: Docker & Docker Compose
- **Optimized Multi-Stage Dockerfile**: Multi-stage build with `python:3.11-slim`, non-root security user (`appuser`), healthcheck probe, and strict `.dockerignore` excluding notebooks, tests, and documentation.
- **Docker Compose Stack**: Orchestrates `db` (PostgreSQL 14), `mlflow` (tracking server & artifact storage), `api` (FastAPI inference engine), and `prometheus` (metrics collector) on an isolated bridge network (`olist_net`).
- **Environment Configuration**: Sensitive connection strings and credentials managed exclusively via `.env` (derived from `.env.example`) and never committed to version control.

### Step 9: CI/CD Pipeline & Code Quality
- **Pre-commit Hooks**: Configured in `.pre-commit-config.yaml` to enforce trailing whitespace removal, YAML/JSON validation, file size limits, Black formatting, and Flake8 linting before git commits.
- **GitHub Actions Workflow**: `.github/workflows/ci.yml` triggers on every push and PR:
  1. `lint`: Enforces Black and Flake8 compliance.
  2. `test`: Executes `pytest` with coverage reporting.
  3. `build-and-push`: Smoke-tests container health and pushes images to GitHub Container Registry (`ghcr.io`) upon merging to `main`.
- **Strict Quality Gate**: Any failing test or lint error immediately terminates the pipeline.

### Step 10: Monitoring, Drift Tracking & Alerting
- **Prometheus Metrics**: `app/monitoring.py` exposes request counters (`http_requests_total`, `prediction_requests_total`), latency histograms (`http_request_duration_seconds`, `prediction_latency_seconds`), error counters (`prediction_errors_total`), and class distribution counters (`predictions_by_class_total`).
- **Prediction Drift Detection**: `src/monitor.py` analyzes live prediction rates against the training baseline ($8.11\%$) and flags drift alerts if deviation exceeds $10\%$.
- **Audit Logging**: Structured JSONL persistence (`logs/predictions.jsonl`) records all payloads for delayed reconciliation against real delivery confirmations.
- **Alerting Framework**: Defined in `docs/monitoring_and_alerting.md` with operational SLAs and severity escalation tiers (P0 to P3).

---

## 3. Definition of Done — Summary Deliverables

| Requirement | Evidence |
|---|---|
| Structured repository, configuration, pinned requirements, README | Root layout, `config/config.yaml`, `requirements/requirements.txt`, `requirements/requirements-dev.txt`, and `README.md`. Runtime is pinned to Python 3.10, scikit-learn 1.0.2, and joblib 1.5.1 to match the persisted artifacts. |
| Inference pipeline loads fitted objects and registered model | `PreprocessingService` loads the frozen `ColumnTransformer`; `PredictionService` loads `models:/olist-late-delivery-model/Production` in production and falls back safely to the packaged artifact. |
| DVC, Great Expectations, MLflow | `.dvc` metadata and `.dvc` artifact files version sample data and model artifacts; `src/validation.py` rejects invalid input before inference; `src/registry.py` tracks and registers MLflow model versions. |
| Unit and integration tests | `tests/` covers utilities, data, features, preprocessing, model behavior, monitoring, and API routes. |
| FastAPI service | `/`, `/health`, `/model-info`, `/predict`, `/predict/batch`, `/monitoring`, `/metrics`, and `/docs` are provided. `/` redirects to Swagger documentation. |
| Docker Compose stack | PostgreSQL, MLflow artifact storage, FastAPI, and Prometheus start with `docker compose up --build`. |
| CI/CD and pre-commit | `.github/workflows/ci.yml` runs Black, Flake8, tests, image build, container health smoke test, and GHCR publication on `main`; `.pre-commit-config.yaml` runs local quality checks. |
| Logging and monitoring | Console and rotating-file logging, JSONL prediction audit records, Prometheus request/latency/error/prediction metrics, drift summary, and alert policy in `docs/monitoring_and_alerting.md`. |

---

## 4. Runtime Verification

The Docker image was built from the repository and the full stack was started successfully. During local verification, alternate host ports were used because ports 5000 and 5432 were already occupied by unrelated services. This does not affect the clean-machine default configuration.

| Service | Result |
|---|---|
| PostgreSQL | Healthy |
| MLflow | Healthy; experiment created and model version 1 registered and promoted to Production |
| FastAPI | Healthy; loaded the model successfully from the MLflow Production registry |
| Prometheus | Ready and scraping API metrics |

Live API checks passed:

- `GET /health` returned `healthy` with `model_loaded: true`.
- `GET /model-info` returned model version `1`, 145 features, tuned Random Forest parameters, and validation/test metrics.
- `POST /predict` with `data/sample_order.json` returned class `0`, label `on_time`, probability `0.4595`, model version `1`, and latency.
- `POST /predict/batch` returned two prediction records for `data/sample_batch.json`.
- A deliberately invalid request was rejected with HTTP 422.
- `/metrics` exposed `http_requests_total` and `prediction_requests_total`; `/monitoring` returned drift and latency summaries.

---

## 5. Notebook Parity

The exact saved production artifacts were executed against `data/sample_order.json`. The result is class `0` (`on_time`) with late probability `0.4595`. This value is asserted in the parity test and documented in the README, ensuring that inference uses the same frozen fitted objects as the notebook workflow.

---

## 6. Automated Verification

The final local quality checks completed successfully:

```text
pytest -q                 30 passed
black --check app config src tests   passed
flake8 app config src tests          passed
```

The test suite includes the negative-path proof required by the task: malformed or out-of-domain requests are rejected rather than reaching the model. CI repeats the same checks on every push and pull request.

---

## 7. Runbook

On a clean machine:

```bash
cp .env.example .env
docker compose up --build
```

Then open `http://localhost:8000` (redirects to `/docs`) or use `http://localhost:8000/health`. To register the packaged model in MLflow:

```bash
docker compose --profile register run --rm register-model
```

---

## 8. Conclusion

Task 3 delivers a tested, containerized, observable inference service. It preserves the trained notebook artifacts, validates requests before inference, records predictions for future evaluation, supports model registry loading, and can be run locally or through the Compose stack without retraining.
