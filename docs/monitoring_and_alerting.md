# Monitoring & Alerting Strategy

MLOps Training 2026/2027 · Task 3: From Notebooks to Production

This document outlines the monitoring, drift detection, and alerting framework for the Olist Late Delivery Inference Service.

---

## 1. Core Service Health & Latency Metrics (SLAs)

| Metric | Measurement Mechanism | Normal Threshold | Alert Threshold | Severity |
| :--- | :--- | :--- | :--- | :--- |
| **HTTP Error Rate** | Prometheus `http_requests_total{status_code=~"5.."}` | $< 0.1\%$ | $> 1.0\%$ over 5 min window | **Critical** (P1) |
| **Inference Latency (p95)** | Prometheus `http_request_duration_seconds` / `latency_ms` | $< 150\text{ ms}$ | $> 500\text{ ms}$ over 5 min window | **Warning** (P2) |
| **Inference Latency (p99)** | Prometheus `http_request_duration_seconds` | $< 300\text{ ms}$ | $> 1000\text{ ms}$ over 5 min window | **Critical** (P1) |
| **Service Availability** | `GET /health` periodic ping (every 15s) | $100\%$ | 2 consecutive non-200 responses | **Critical** (P0) |
| **Validation Error Spike** | `prediction_errors_total{error_type="validation"}` | $< 2\%$ | $> 10\%$ of incoming traffic | **Warning** (P2) |

---

## 2. Model & Data Drift Monitoring

Because consumer and logistical behaviors change, the inference service logs every prediction request to `logs/predictions.jsonl` with timestamp, input attributes, predicted class, probability, and model version.

### A. Prediction Distribution Drift (Prior Probability Shift)
- **Baseline Late Delivery Rate**: **$8.11\%$** (established on the 67,529 orders in Task 2 training split).
- **Drift Detection**:
  - A rolling window of 1,000 predictions (or daily batch) is evaluated.
  - If the proportion of orders predicted as late exceeds **$18\%$** ($+10\%$ absolute shift from baseline) or drops below **$2\%$**, a **Drift Alert** is triggered.
  - *Context*: During the March 2018 Brazilian postal strikes, late deliveries surged to 21.36%. Tracking this metric allows proactive logistics intervention before customers complain.

### B. Input Covariate Shift (Feature Drift)
- **Financial Features**: Monitor `total_price` and `total_freight` distribution medians against baseline training medians ($R\$ 74.90$ and $R\$ 16.29$). A Kolmogorov-Smirnov test $p < 0.01$ indicates macroeconomic shift.
- **Geographic Coverage**: Spikes in orders originating from remote northern/northeastern states (`BA`, `MA`, `RJ`) trigger routing alerts, as these regions have 2.4x higher delay friction.
- **Feature Missingness**: If missing rate on coordinates (`customer_lat`, `seller_lat`) exceeds $5\%$ (baseline is $0.3\%$), alert the data ingestion engineering team.

---

## 3. Ground Truth Delayed Evaluation

In late-delivery prediction, ground truth delivery dates (`order_delivered_customer_date`) arrive days or weeks after the purchase timestamp:

1. **Audit Logs**: All requests are appended to `logs/predictions.jsonl` with unique `order_id`.
2. **Delayed Reconciliation Job**:
   - A weekly batch cron joins `logs/predictions.jsonl` with incoming delivery confirmations from the PostgreSQL `orders` table.
   - Computes empirical **ROC-AUC**, **PR-AUC**, and **Late Class F1-Score**.
   - If ROC-AUC drops below **$0.75$** (baseline: $0.8156$), trigger **Model Retraining Alert**.

---

## 4. Alert Routing & Escalation Policy

- **P0 / P1 Alerts (Service Outage, Error Rate > 1%, p99 > 1s)**:
  - Sent via PagerDuty / OpsGenie to on-call MLOps engineer.
  - Automatic restart via container orchestration (Docker restart policy).
- **P2 Alerts (Validation errors, p95 latency > 500ms)**:
  - Slack `#mlops-alerts` channel notification.
- **P3 Alerts (Model drift, distribution shift, missingness)**:
  - Notification to Data Science team ticket backlog for investigation and retraining pipeline trigger.
