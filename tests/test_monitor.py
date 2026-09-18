"""Monitoring module tests using a temporary prediction log."""

import json
from pathlib import Path

from src.monitor import ModelMonitor


def test_monitor_detects_drift(tmp_path: Path):
    log_path = tmp_path / "predictions.jsonl"
    records = []
    for i in range(20):
        records.append(
            {
                "timestamp": "2018-01-01T00:00:00Z",
                "output": {
                    "order_id": f"o{i}",
                    "prediction": 1,
                    "label": "late",
                    "late_probability": 0.9,
                    "latency_ms": 12.0,
                    "model_version": "1",
                },
            }
        )
    log_path.write_text(
        "\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8"
    )

    summary = ModelMonitor(log_path=log_path).compute_summary_metrics()
    assert summary["total_predictions"] == 20
    assert summary["drift_detected"] is True
    assert any("DRIFT" in alert for alert in summary["active_alerts"])


def test_monitor_empty_log_has_schema(tmp_path: Path):
    log_path = tmp_path / "predictions.jsonl"
    summary = ModelMonitor(log_path=log_path).compute_summary_metrics()
    assert summary["total_predictions"] == 0
    assert "timestamp" in summary
    assert summary["active_alerts"] == []
