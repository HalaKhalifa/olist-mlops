# LinkedIn Post & Portfolio Case Study

Use this structured write-up, demo flow, and metrics comparison to showcase the project on LinkedIn, your portfolio, or GitHub!

---

## 📱 Ready-to-Post LinkedIn Write-up

```markdown
🚀 Excited to share my latest end-to-end MLOps project: Predicting E-Commerce Late Deliveries at Scale!

In e-commerce, late deliveries directly damage customer retention and inflate support costs. Working with the Brazilian E-Commerce dataset (Olist), I built a production-grade, reproducible MLOps system from database architecture to a containerized inference service with real-time drift monitoring.

Key Highlights of the System:
🔹 Relational Data Architecture: Multi-table PostgreSQL schema with grain alignment and zero-leakage pre-aggregations.
🔹 Feature Engineering & Parity: Built domain features (Haversine logistics distance, interstate delivery friction, warehouse dispatch lag ratios, and transit pressure).
🔹 Model Optimization (+31% PR-AUC): Handled severe class imbalance (~8.11% minority late rate) using regularized HistGradientBoosting and optimal decision boundary thresholding, boosting PR-AUC from 0.3528 ➡️ 0.4620 and F1-score from 0.3913 ➡️ 0.4341.
🔹 Production Architecture (FastAPI & Docker):
  • Great Expectations data validation firewall rejecting bad/leaky payloads before model execution (HTTP 422).
  • MLflow Model Registry integration with champion stage promotion.
  • DVC artifact versioning for full model & data lineage.
🔹 CI/CD & Observability:
  • Automated GitHub Actions pipeline (Black, Flake8, Pytest suite, GHCR image builds).
  • Prometheus metrics & live drift monitoring tracking latency SLAs and distribution shifts against baseline.
🔹 Interactive UI: Built a Streamlit Showcase Dashboard for live order risk simulation and batch analytics.

🔗 GitHub Repository: https://github.com/HalaKhalifa/olist-mlops

#MLOps #MachineLearning #FastAPI #Docker #DataScience #MLflow #DVC #Python #Streamlit #Prometheus
```

---

## 📸 Recommended Screenshots / Media for Your Post

To maximize engagement on LinkedIn, capture 2 to 4 screenshots or a short GIF/screen recording showing:

1. **Interactive Streamlit Dashboard** (`streamlit run app/dashboard.py`):
   - **Tab 1: Live Order Risk Simulator**: Adjusting sliders (e.g. carrier delay or interstate delivery) and showing the dynamic probability gauge change.
   - **Tab 3: MLOps Drift & Monitoring**: Showing the live late delivery rate vs. the $8.11\%$ training baseline.
2. **FastAPI Swagger Documentation** (`http://localhost:8000/docs`):
   - Demonstrating the `/predict` endpoint with schema validation and example payloads.
3. **MLflow Model Registry UI** (`http://localhost:5000`):
   - Showing experiment runs, logged parameters, artifacts, and the registered model in the `Production` stage.
4. **CI/CD Pipeline Run**:
   - Showing green passing checkmarks on GitHub Actions (linting $\rightarrow$ testing $\rightarrow$ Docker build/push).

---

## 📊 Summary Metrics Table (For Portfolio / README)

| Dimension | Initial Baseline | Production Champion | Impact |
| :--- | :---: | :---: | :---: |
| **Model** | Random Forest | Regularized HistGradientBoosting | Non-linear tabular boosting |
| **PR-AUC (Test)** | $0.3528$ | **$0.4620$** | **$+30.95\%$ relative gain** |
| **ROC-AUC (Test)** | $0.8156$ | **$0.8481$** | $+3.98\%$ gain across thresholds |
| **Late Class Precision** | $31.00\%$ | **$44.64\%$** | Fewer false alarms on deliveries |
| **Late Class F1-Score** | $0.3913$ | **$0.4341$** | $+10.94\%$ overall balance |
| **Feature Count** | 145 | **150** | Added transit speed & lag ratios |
| **Inference Latency** | $< 150\text{ ms}$ | $< 50\text{ ms}$ | High-throughput serving |
| **Data Quality** | Manual checks | **Great Expectations Firewall** | Auto-rejects bad payloads (HTTP 422) |
