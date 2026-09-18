"""Unit tests for shared utilities."""

from config.settings import settings
from src.utils import load_artifact, load_json


def test_load_json_metrics():
    metrics = load_json(settings.model.metrics_path)
    assert "model" in metrics or "test" in metrics or "validation" in metrics


def test_load_artifact_preprocessor():
    preprocessor = load_artifact(settings.model.preprocessor_path)
    assert hasattr(preprocessor, "transform")
