"""MLflow Experiment Tracking and Model Registry automation.

Logs runs, parameters, metrics, and artifacts into MLflow.
Registers the model in MLflow Model Registry with version and 'Production' stage.
"""

import json
import os
from typing import Optional
import joblib

from config.logging_config import logger
from config.settings import settings


def register_model_with_mlflow(
    tracking_uri: Optional[str] = None,
    experiment_name: Optional[str] = None,
) -> None:
    """Log model run, parameters, metrics, and register in MLflow."""
    import mlflow
    import mlflow.sklearn

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", settings.mlflow.tracking_uri)
    exp_name = experiment_name or settings.mlflow.experiment_name

    logger.info(f"Connecting to MLflow Tracking Server at: {uri}")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(exp_name)

    # 1. Load metrics and parameters
    metrics_file = settings.model.metrics_path
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")
    with open(metrics_file, "r") as f:
        metrics_data = json.load(f)

    # 2. Load model and preprocessor
    model_path = settings.model.model_path
    preprocessor_path = settings.model.preprocessor_path
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)

    with mlflow.start_run(run_name="production-tuned-random-forest") as run:
        run_id = run.info.run_id
        logger.info(f"Started MLflow Run: {run_id}")

        # Log parameters
        params = metrics_data.get("best_params", {})
        for k, v in params.items():
            mlflow.log_param(k, v)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("class_weight", "balanced")

        # Log metrics
        val_metrics = metrics_data.get("validation", {})
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", v)

        test_metrics = metrics_data.get("test", {})
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", v)

        baseline_metrics = metrics_data.get("baseline", {})
        for k, v in baseline_metrics.items():
            mlflow.log_metric(f"baseline_{k}", v)

        # Log artifacts
        mlflow.log_artifact(str(metrics_file), artifact_path="metadata")
        if preprocessor_path.exists():
            mlflow.log_artifact(str(preprocessor_path), artifact_path="preprocessor")
        if settings.model.feature_names_path.exists():
            mlflow.log_artifact(
                str(settings.model.feature_names_path), artifact_path="features"
            )

        # Log model with signature and input example if available
        model_name = settings.mlflow.model_registry_name
        logger.info(f"Logging and registering model '{model_name}'...")
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=model_name,
        )

        logger.info(f"Model logged with URI: {model_info.model_uri}")

        # Transition model version to Production stage
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient(tracking_uri=uri)
            latest_versions = client.get_latest_versions(model_name)
            if latest_versions:
                version = latest_versions[-1].version
                client.transition_model_version_stage(
                    name=model_name,
                    version=version,
                    stage="Production",
                    archive_existing_versions=True,
                )
                logger.info(
                    "Successfully transitioned model '%s' version %s to Production.",
                    model_name,
                    version,
                )
        except Exception as e:
            logger.warning(
                f"Could not transition model version stage via MlflowClient: {e}"
            )

    logger.info("MLflow model logging and registration process completed.")


if __name__ == "__main__":
    register_model_with_mlflow()
