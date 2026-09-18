"""Centralized settings and configuration loader for Olist MLOps service."""

import os
from pathlib import Path
from typing import Any, Dict, List
import yaml
from pydantic import BaseModel, Field

# Determine project root directory dynamically
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class ProjectConfig(BaseModel):
    name: str = "olist-late-delivery"
    version: str = "1.0.0"
    description: str = "Production inference service for predicting late order deliveries"


class ModelConfig(BaseModel):
    name: str = "olist-late-delivery-model"
    version: str = "1"
    stage: str = "Production"
    alias: str = "champion"
    model_path: Path = PROJECT_ROOT / "models" / "best_model.joblib"
    preprocessor_path: Path = PROJECT_ROOT / "models" / "preprocessing_pipeline.joblib"
    feature_names_path: Path = PROJECT_ROOT / "models" / "feature_names.txt"
    metrics_path: Path = PROJECT_ROOT / "models" / "metrics.json"


class MLflowConfig(BaseModel):
    tracking_uri: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    )
    experiment_name: str = Field(
        default_factory=lambda: os.getenv("MLFLOW_EXPERIMENT_NAME", "olist-late-delivery")
    )
    model_registry_name: str = "olist-late-delivery-model"
    model_stage: str = "Production"


class LoggingConfig(BaseModel):
    level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    format: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    log_file: Path = PROJECT_ROOT / "logs" / "app.log"
    prediction_log_file: Path = PROJECT_ROOT / "logs" / "predictions.jsonl"
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 5


class ValidationConfig(BaseModel):
    min_price: float = 0.0
    min_freight: float = 0.0
    min_items: int = 1
    min_weight_g: float = 0.0
    min_volume_cm3: float = 0.0
    min_installments: int = 1
    allowed_payment_types: List[str] = [
        "credit_card", "boleto", "voucher", "debit_card", "not_defined"
    ]
    allowed_brazilian_states: List[str] = [
        "AC", "AL", "AM", "AP", "BA", "CE", "DF", "ES", "GO", "MA", "MG",
        "MS", "MT", "PA", "PB", "PE", "PI", "PR", "RJ", "RN", "RO", "RR",
        "RS", "SC", "SE", "SP", "TO"
    ]


class MonitoringConfig(BaseModel):
    max_latency_ms: float = 500.0
    max_error_rate: float = 0.01
    baseline_late_rate: float = 0.0811
    max_drift_rate_deviation: float = 0.10


class AppSettings(BaseModel):
    project_root: Path = PROJECT_ROOT
    host: str = Field(default_factory=lambda: os.getenv("API_HOST", "0.0.0.0"))
    port: int = Field(default_factory=lambda: int(os.getenv("API_PORT", "8000")))
    environment: str = Field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    mlflow: MLflowConfig = Field(default_factory=MLflowConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)


def load_settings(yaml_path: Path = PROJECT_ROOT / "config" / "config.yaml") -> AppSettings:
    """Load configuration from config.yaml and environment variables."""
    overrides: Dict[str, Any] = {}
    if yaml_path.exists():
        with open(yaml_path, "r") as f:
            raw_yaml = yaml.safe_load(f) or {}

        if "project" in raw_yaml:
            overrides["project"] = ProjectConfig(**raw_yaml["project"])
        if "model" in raw_yaml:
            model_dict = raw_yaml["model"].copy()
            for key in ["model_path", "preprocessor_path", "feature_names_path", "metrics_path"]:
                if key in model_dict:
                    model_dict[key] = PROJECT_ROOT / model_dict[key]
            overrides["model"] = ModelConfig(**model_dict)
        if "mlflow" in raw_yaml:
            overrides["mlflow"] = MLflowConfig(**raw_yaml["mlflow"])
        if "logging" in raw_yaml:
            log_dict = raw_yaml["logging"].copy()
            if "log_file" in log_dict:
                log_dict["log_file"] = PROJECT_ROOT / log_dict["log_file"]
            if "prediction_log_file" in log_dict:
                log_dict["prediction_log_file"] = PROJECT_ROOT / log_dict["prediction_log_file"]
            overrides["logging"] = LoggingConfig(**log_dict)
        if "validation" in raw_yaml:
            overrides["validation"] = ValidationConfig(**raw_yaml["validation"])
        if "monitoring" in raw_yaml and "alert_thresholds" in raw_yaml["monitoring"]:
            overrides["monitoring"] = MonitoringConfig(**raw_yaml["monitoring"]["alert_thresholds"])

    return AppSettings(**overrides)


# Global settings singleton
settings = load_settings()
