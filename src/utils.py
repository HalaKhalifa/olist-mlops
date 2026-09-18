"""Common utility functions for Olist MLOps pipeline."""

import json
from pathlib import Path
from typing import Any, Dict
import joblib

from config.logging_config import logger


def load_json(file_path: Path) -> Dict[str, Any]:
    """Load JSON file safely."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], file_path: Path, indent: int = 2) -> None:
    """Save data to JSON file safely."""
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent)


def load_artifact(artifact_path: Path) -> Any:
    """Load a serialized model or transformer artifact via joblib."""
    if not artifact_path.exists():
        raise FileNotFoundError(f"Artifact not found at {artifact_path}")
    logger.debug(f"Loading artifact from {artifact_path}")
    return joblib.load(artifact_path)
