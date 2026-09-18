"""Shared pytest fixtures for unit and integration testing."""

import json
import pytest
from starlette.testclient import TestClient

from app.main import app
from config.settings import PROJECT_ROOT


@pytest.fixture(scope="session")
def sample_order_dict():
    """Load valid single order payload fixture."""
    order_file = PROJECT_ROOT / "data" / "sample_order.json"
    with open(order_file, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def sample_batch_dict():
    """Load valid batch order payload fixture."""
    batch_file = PROJECT_ROOT / "data" / "sample_batch.json"
    with open(batch_file, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def api_client():
    """Create FastAPI test client."""
    with TestClient(app) as client:
        yield client
