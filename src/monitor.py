"""Monitoring, drift tracking, and prediction log evaluation module.

Analyzes stored prediction logs, computes drift metrics against baseline,
monitors latency distributions, and generates alert reports.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd

from config.settings import settings


class ModelMonitor:
    """Monitors live predictions, latency, distribution drift, and alerts."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path or settings.logging.prediction_log_file
        self.monitoring_config = settings.monitoring

    def load_prediction_logs(self) -> pd.DataFrame:
        """Load prediction audit logs from structured JSONL file."""
        if not self.log_path.exists():
            return pd.DataFrame()

        records = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line)
                        flat_record = {
                            "timestamp": record.get("timestamp"),
                            "order_id": record.get("output", {}).get("order_id"),
                            "prediction": record.get("output", {}).get("prediction"),
                            "label": record.get("output", {}).get("label"),
                            "late_probability": record.get("output", {}).get("late_probability"),
                            "latency_ms": record.get("output", {}).get("latency_ms"),
                            "model_version": record.get("output", {}).get("model_version"),
                        }
                        records.append(flat_record)
                    except json.JSONDecodeError:
                        continue

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df

    def compute_summary_metrics(self) -> Dict[str, Any]:
        """Compute live operational and distribution metrics from logs."""
        df = self.load_prediction_logs()
        if df.empty:
            return {
                "total_predictions": 0,
                "message": "No prediction logs recorded yet.",
            }

        total_predictions = len(df)
        late_count = int((df["prediction"] == 1).sum())
        on_time_count = int((df["prediction"] == 0).sum())
        current_late_rate = float(df["prediction"].mean())
        mean_latency = float(df["latency_ms"].mean())
        p95_latency = float(np.percentile(df["latency_ms"].dropna(), 95))
        mean_probability = float(df["late_probability"].mean())

        baseline_late_rate = self.monitoring_config.baseline_late_rate
        rate_diff = abs(current_late_rate - baseline_late_rate)
        drift_detected = rate_diff > self.monitoring_config.max_drift_rate_deviation

        # Alert evaluation
        alerts = []
        if p95_latency > self.monitoring_config.max_latency_ms:
            alerts.append(
                f"HIGH LATENCY ALERT: p95 latency is {p95_latency:.2f}ms "
                f"(threshold: {self.monitoring_config.max_latency_ms}ms)"
            )
        if drift_detected:
            alerts.append(
                f"PREDICTION DRIFT ALERT: Current late rate {current_late_rate*100:.2f}% "
                f"deviates by {rate_diff*100:.2f}% from baseline {baseline_late_rate*100:.2f}%"
            )

        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "total_predictions": total_predictions,
            "late_count": late_count,
            "on_time_count": on_time_count,
            "current_late_rate": round(current_late_rate, 4),
            "baseline_late_rate": round(baseline_late_rate, 4),
            "rate_difference": round(rate_diff, 4),
            "drift_detected": drift_detected,
            "mean_probability": round(mean_probability, 4),
            "mean_latency_ms": round(mean_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "active_alerts": alerts,
        }


# Singleton monitor instance
model_monitor = ModelMonitor()


if __name__ == "__main__":
    metrics = model_monitor.compute_summary_metrics()
    print(json.dumps(metrics, indent=2))
