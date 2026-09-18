# Task 3 Report: From Notebooks to Production

**MLOps Engineering Training 2026/2027 · Olist Late Delivery Prediction**

---

## 1. Objective

Task 3 productionizes the Task 2 late-delivery model as an inference-only service. Training remains in the notebooks; the service accepts a new order and returns the predicted class (`late` or `on_time`), late-delivery probability, model version, and latency.

## 2. Delivered architecture

The repository separates the inference service into `app/`, `src/`, `config/`, `data/`, `models/`, `tests/`, `requirements/`, and `monitoring/`.

- `src/data.py`, `src/validation.py`, `src/preprocess.py`, `src/features.py`, and `src/predict.py` implement discrete data, validation, preprocessing, feature-engineering, and inference responsibilities.
- `config/settings.py` loads paths and parameters from `config/config.yaml` and environment variables.
- The saved `preprocessing_pipeline.joblib` and `best_model.joblib` are loaded at inference. They are never fitted in the service.
- `app/` provides FastAPI routes and schemas; `Dockerfile` and `docker-compose.yml` define the production stack.

## 3. Definition of done — deliverables

| Requirement | Evidence |
|---|---|
| Structured repository, configuration, pinned requirements, README | Root layout, `config/config.yaml`, `requirements/requirements.txt`, `requirements/requirements-dev.txt`, and `README.md`. Runtime is pinned to Python 3.11, scikit-learn 1.7.0, and joblib 1.5.1 to match the persisted artifacts. |
| Inference pipeline loads fitted objects and registered model | `PreprocessingService` loads the frozen `ColumnTransformer`; `PredictionService` loads `models:/olist-late-delivery-model/Production` in production and falls back safely to the packaged artifact. |
| DVC, Great Expectations, MLflow | `.dvc` metadata and `.dvc` artifact files version sample data and model artifacts; `src/validation.py` rejects invalid input before inference; `src/registry.py` tracks and registers MLflow model versions. |
| Unit and integration tests | `tests/` covers utilities, data, features, preprocessing, model behavior, monitoring, and API routes. |
| FastAPI service | `/`, `/health`, `/model-info`, `/predict`, `/predict/batch`, `/monitoring`, `/metrics`, and `/docs` are provided. `/` redirects to Swagger documentation. |
| Docker Compose stack | PostgreSQL, MLflow artifact storage, FastAPI, and Prometheus start with `docker compose up --build`. |
| CI/CD and pre-commit | `.github/workflows/ci.yml` runs Black, Flake8, tests, image build, container health smoke test, and GHCR publication on `main`; `.pre-commit-config.yaml` runs local quality checks. |
| Logging and monitoring | Console and rotating-file logging, JSONL prediction audit records, Prometheus request/latency/error/prediction metrics, drift summary, and alert policy in `docs/monitoring_and_alerting.md`. |

## 4. Runtime verification

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

## 5. Notebook parity

The exact saved production artifacts were executed against `data/sample_order.json`. The result is class `0` (`on_time`) with late probability `0.4595`. This value is asserted in the parity test and documented in the README, ensuring that inference uses the same frozen fitted objects as the notebook workflow.

## 6. Automated verification

The final local quality checks completed successfully:

```text
pytest -q                 30 passed
black --check app config src tests   passed
flake8 app config src tests          passed
```

The test suite includes the negative-path proof required by the task: malformed or out-of-domain requests are rejected rather than reaching the model. CI repeats the same checks on every push and pull request.

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

## 8. Conclusion

Task 3 delivers a tested, containerized, observable inference service. It preserves the trained notebook artifacts, validates requests before inference, records predictions for future evaluation, supports model registry loading, and can be run locally or through the Compose stack without retraining.
