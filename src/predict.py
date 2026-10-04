"""Inference pipeline for Olist Late Delivery prediction.

Supports both API-driven execution and Command Line Interface (CLI).
Loads the frozen model artifact (via MLflow Model Registry or local fallback),
executes feature transformation, produces binary class and late probability,
logs the request and latency, and records predictions.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from config.logging_config import logger
from config.settings import settings
from src.data import order_dict_to_dataframe
from src.preprocess import PreprocessingService, preprocessor_service
from src.utils import load_artifact


class PredictionService:
    """Production Inference Service."""

    def __init__(
        self,
        model_path: Optional[Path] = None,
        preprocessor: Optional[PreprocessingService] = None,
    ):
        self.model_path = model_path or settings.model.model_path
        self.preprocessor_service = preprocessor or preprocessor_service
        self._model: Optional[Any] = None
        self.model_name: str = settings.model.name
        self.model_version: str = settings.model.version
        self.model_stage: str = settings.model.stage

    @property
    def model(self) -> Any:
        """Load model with MLflow Model Registry preference and local fallback."""
        if self._model is None:
            self._model = self._load_model()
        return self._model

    def _load_model(self) -> Any:
        """Attempt loading from MLflow registry; fallback to local joblib file."""
        # The registry is a production dependency. Local development and tests
        # must remain runnable without a live MLflow server.
        if settings.environment != "production":
            logger.info("Development mode: loading local model artifact.")
            return load_artifact(self.model_path)

        # 1. Check MLflow Model Registry in production
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", settings.mlflow.tracking_uri)
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_tracking_uri(tracking_uri)
            registry_name = settings.mlflow.model_registry_name
            model_uri = f"models:/{registry_name}/{settings.mlflow.model_stage}"
            logger.info(
                f"Attempting to load model from MLflow Model Registry: {model_uri}"
            )
            model = mlflow.sklearn.load_model(model_uri)
            try:
                from mlflow.tracking import MlflowClient

                versions = MlflowClient(tracking_uri=tracking_uri).get_latest_versions(
                    registry_name, stages=[settings.mlflow.model_stage]
                )
                if versions:
                    self.model_version = str(
                        max(versions, key=lambda version: int(version.version)).version
                    )
            except Exception as e:
                logger.warning(
                    f"Could not determine the loaded MLflow model version: {e}"
                )
            logger.info("Successfully loaded model from MLflow Model Registry.")
            return model
        except Exception as e:
            logger.warning(
                f"Could not load model from MLflow ({e}). Falling back to local artifact: {self.model_path}"
            )

        # 2. Local artifact fallback
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model artifact not found at {self.model_path}")
        logger.info(f"Loading local model artifact from {self.model_path}")
        return load_artifact(self.model_path)

    def predict_dataframe(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Predict binary late label and probability for a DataFrame of orders."""
        # Transform inputs into feature matrix
        X = self.preprocessor_service.transform(df)

        # Run inference
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = self.model.predict(X)

        return predictions, probabilities

    def predict_single(self, order_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Predict for a single order payload, measure latency, and log."""
        start_time = time.perf_counter()

        # Convert to DataFrame and predict
        df = order_dict_to_dataframe(order_dict)
        predictions, probabilities = self.predict_dataframe(df)

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        pred_class = int(predictions[0])
        late_prob = float(probabilities[0])
        order_id = order_dict.get("order_id", "unknown")

        result = {
            "order_id": order_id,
            "prediction": pred_class,
            "label": "late" if pred_class == 1 else "on_time",
            "late_probability": round(late_prob, 4),
            "model_name": self.model_name,
            "model_version": self.model_version,
            "latency_ms": round(latency_ms, 2),
        }

        # Step 3 requirement: Log every prediction request: input, output, latency, model version
        logger.info(
            f"Prediction | order_id={order_id} | label={result['label']} | "
            f"prob={late_prob:.4f} | latency={latency_ms:.2f}ms | version={self.model_version}"
        )

        # Step 10 requirement: Store prediction logs
        self._log_prediction_record(order_dict, result)

        return result

    def predict_batch(self, orders: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict for a list of orders in batch, measure latency, and log."""
        start_time = time.perf_counter()

        df = order_dict_to_dataframe(orders)
        predictions, probabilities = self.predict_dataframe(df)

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        per_item_latency = latency_ms / max(len(orders), 1)

        results = []
        for i, order_dict in enumerate(orders):
            pred_class = int(predictions[i])
            late_prob = float(probabilities[i])
            order_id = order_dict.get("order_id", f"batch_item_{i}")

            item_result = {
                "order_id": order_id,
                "prediction": pred_class,
                "label": "late" if pred_class == 1 else "on_time",
                "late_probability": round(late_prob, 4),
                "model_name": self.model_name,
                "model_version": self.model_version,
                "latency_ms": round(per_item_latency, 2),
            }
            results.append(item_result)
            self._log_prediction_record(order_dict, item_result)

        logger.info(
            f"Batch Prediction | total_orders={len(orders)} | "
            f"total_latency={latency_ms:.2f}ms | avg_latency={per_item_latency:.2f}ms"
        )
        return results

    def _log_prediction_record(
        self, input_payload: Dict[str, Any], output_payload: Dict[str, Any]
    ) -> None:
        """Persist prediction audit records to structured JSONL file for drift & evaluation."""
        try:
            log_path = settings.logging.prediction_log_file
            log_path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "input": {
                    k: v for k, v in input_payload.items() if k not in ["order_id"]
                },
                "output": output_payload,
            }
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to record prediction log: {e}")


# Singleton prediction service instance
prediction_service = PredictionService()


def main():
    """Command-line interface for running predictions."""
    parser = argparse.ArgumentParser(description="Olist Late Delivery Prediction CLI")
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="data/sample_order.json",
        help="Path to JSON file containing a single order or batch of orders",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found at {input_path}")
        sys.exit(1)

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "orders" in data:
        results = prediction_service.predict_batch(data["orders"])
        sys.stdout.write(json.dumps(results, indent=2) + "\n")
    elif isinstance(data, list):
        results = prediction_service.predict_batch(data)
        sys.stdout.write(json.dumps(results, indent=2) + "\n")
    elif isinstance(data, dict):
        result = prediction_service.predict_single(data)
        sys.stdout.write(json.dumps(result, indent=2) + "\n")
    else:
        logger.error("Invalid JSON structure in input file.")
        sys.exit(1)


if __name__ == "__main__":
    main()
