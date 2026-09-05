# Task 2 Report: From Tables to ML Notebooks

**MLOps Engineering Training 2026/2027 · Brazilian E-Commerce (Olist) Late Delivery Prediction**

---

## 1. Executive Summary & Objective

The objective of Task 2 is to build a reproducible, leakage-free machine learning workflow predicting whether an order will arrive **late** (`is_late = 1`) relative to the estimated delivery date promised at checkout.

This report synthesizes the methodology, engineering decisions, empirical discoveries, and validation results across the six sequential notebooks:
1. `01_read_and_join_tables.ipynb` — Relational schema ingestion, grain alignment, and multi-table denormalization.
2. `02_create_labels.ipynb` — Target definition, population filtering, and class imbalance diagnosis.
3. `03_train_validation_test_split.ipynb` — Split strategy evaluation (temporal vs. stratified random) and leakage prevention.
4. `04_eda.ipynb` — Exploratory data analysis strictly contained within the training split.
5. `05_feature_engineering.ipynb` — Predictive feature creation and `scikit-learn` pipeline fitting (`fit` on train only).
6. `06_train_tune_evaluate.ipynb` — Dummy baseline, linear model, hyperparameter-tuned ensemble, and single-pass test evaluation.

---

## 2. Relational Ingestion & Grain Alignment (`01_read_and_join_tables`)

### 2.1 The Grain Problem
The Olist database spans 9 relational tables across multiple business entities (`orders`, `order_items`, `order_payments`, `customers`, `sellers`, `products`, `geolocation`, `order_reviews`, `product_category_translation`).
- The unit of prediction (the **ML grain**) is an **individual order** (`order_id`).
- In raw relational form, `order_items` contains multiple rows per order (up to 21 items), `order_payments` contains multiple rows per order (up to 29 payment splits), and `geolocation` contains over 1,000,000 coordinate records across ~19,000 postal code prefixes.
- Direct joins without pre-aggregation produce catastrophic **fan-out**, duplicating orders and distorting training distributions.

### 2.2 Pre-Join Aggregations
To enforce a strict 1-to-1 relationship with `orders` (99,441 rows):
- **`order_items`**: Aggregated to `item_count`, `total_price`, `avg_item_price`, `total_freight`, `avg_item_freight`, `total_weight_g`, `total_volume_cm3`, and `num_sellers`. The primary seller and product category were selected based on the highest-priced item.
- **`order_payments`**: Aggregated to `total_payment_value`, `payment_installments_max`, `payment_transactions_count`, and `dominant_payment_type` (by highest transaction value).
- **`geolocation`**: Compressed via median aggregation per 5-digit `zip_code_prefix` to resolve multi-coordinate anomalies, then mapped to both customer and seller zip codes.
- **Leakage Prevention**: `order_reviews` was **deliberately excluded** from the master ML table because customer reviews are submitted *post-delivery*, which would introduce forward-looking target leakage.

**Result**: A clean master feature table of **99,441 rows $\times$ 33 columns** (`joined_orders.parquet`).

---

## 3. Target Label Creation & Class Balance (`02_create_labels`)

### 3.1 Target Definition
The target variable is defined by comparing the actual customer delivery timestamp ($T_{\text{delivered}}$) against the delivery SLA promised at checkout ($T_{\text{estimated}}$):

$$\text{delay\_days} = \frac{T_{\text{delivered}} - T_{\text{estimated}}}{86,400\text{ seconds}}$$

$$\text{is\_late} = \begin{cases} 
1, & \text{if } T_{\text{delivered}} > T_{\text{estimated}} \\ 
0, & \text{otherwise} 
\end{cases}$$

Where:
- $T_{\text{delivered}}$ is the timestamp from `order_delivered_customer_date`
- $T_{\text{estimated}}$ is the timestamp from `order_estimated_delivery_date`
- An order is classified as late ($\text{is\_late} = 1$) if actual delivery occurs after the promised date.

### 3.2 Population Filtering
- Orders with non-delivered status (`canceled`, `shipped`, `processing`, or `unavailable`) and records with missing customer delivery dates were filtered out.
- Orders where `order_delivered_customer_date < order_purchase_timestamp` were flagged and dropped as physical logging anomalies.
- **Eligible population**: **96,470 orders**.

### 3.3 Class Imbalance Diagnosis
- **On-Time (`is_late = 0`)**: 88,644 orders (**91.89%**)
- **Late (`is_late = 1`)**: 7,826 orders (**8.11%**)
- **Imbalance Ratio**: **11.33 : 1**
- **Operational Takeaway**: Raw accuracy is misleading (a trivial majority-class classifier scores 91.89% accuracy while catching 0% of late orders). Model optimization and evaluation must prioritize **ROC-AUC**, **PR-AUC (Average Precision)**, and minority class **F1-score / Recall**.
- **Artifact**: `02_create_labels/labeled_orders.parquet` (96,470 rows $\times$ 37 columns).

---

## 4. Splitting Strategy & Data Leakage Prevention (`03_train_validation_test_split`)

### 4.1 Evaluation of Splitting Strategies
We evaluated two candidates for dataset partitioning:
1. **Temporal Cutoff Split**:
   - Training on historical orders, testing on future quarters.
   - *Finding*: Olist operations experienced an extreme external shock in March 2018 due to nationwide Brazilian postal courier strikes (`Correios`), causing late delivery rates to temporarily surge past **21%**. A pure temporal cutoff split creates severe covariate and prior probability shift between splits (Train: 9.03%, Val: 5.34%, Test: 6.61%), distorting baseline validation.
2. **Stratified Random Split (Selected Strategy)**:
   - Partitions orders randomly while strictly preserving the empirical label distribution across all partitions.
   - Stratification ratio: **70% Train (67,529 rows)**, **15% Validation (14,470 rows)**, **15% Test (14,471 rows)** with `random_state=42`.
   - All three partitions maintain exactly **8.11% late rate** ($\pm 0.00\%$) with zero order ID overlap.

### 4.2 Leakage Prevention Firewall
- The validation and test splits were quarantined in separate Parquet files (`val.parquet`, `test.parquet`).
- All exploratory data analysis, outlier profiling, and preprocessor transformations were derived **exclusively from `train.parquet`**.
- The test set was held untouched until the final model evaluation in Notebook 6.

---

## 5. Exploratory Data Analysis Key Findings (`04_eda`)

Conducting detailed EDA strictly on the 67,529 training rows revealed key predictive signals and operational dynamics:

### 5.1 Schema & Memory Profiling
- **Memory Footprint**: 65.63 MB in memory (average 1,019 bytes per order).
- **Taxonomy**: 4 ID columns, 5 timestamp columns, 7 categoricals, 20 numericals, and 1 binary label (`is_late`).

### 5.2 Informative Missingness
- Missing coordinate rates: `customer_lat/lng` (0.32%) and `seller_lat/lng` (0.24%).
- Orders with missing coordinates experience a **12.2% late delivery rate** versus **8.1%** for mapped orders (+4.1% higher risk), indicating remote, unmapped zip codes face significantly higher logistics friction.

### 5.3 Numerical Profiling, Skewness & Outliers
- **Distribution Skew**: `total_price` (skew = 8.1), `total_freight` (skew = 3.9), `total_weight_g` (skew = 3.6), and `total_volume_cm3` (skew = 5.2) exhibit heavy right tails.
- **Outlier Detection**: Using the IQR rule ($Q_3 + 1.5 \times \text{IQR}$), 8.3% of orders have outlier freight costs (> 46.5 BRL) and 9.4% have outlier weights (> 4.1 kg).
- **Physical Ranges**: Verified that all numerical features have physically valid values (zero negative prices, zero negative freight, zero negative promised windows).

### 5.4 Geographic Distance & Interstate Logistics Friction
- **Haversine Distance**: Computed great-circle distance between customer and seller coordinates:
  $$\text{Mean} = 601.5\text{ km}, \quad \text{Median} = 434.2\text{ km}, \quad \text{Max} = 2,863.8\text{ km}$$
  Correlates positively with late delivery ($r = +0.071, p < 10^{-10}$).
- **Interstate Delivery Friction**: Cross-state deliveries exhibit a **9.92% delay rate** compared to **4.09%** for intra-state deliveries (**2.4x higher delay risk**).
- **State Disparities**: Delivery destination is highly influential. Remote/Northeastern states (`RJ`: 13.4%, `BA`: 14.2%, `MA`: 15.6%) suffer higher delay rates than São Paulo (`SP`: 5.8%).

### 5.5 Temporal Dynamics & External Shocks
- **Day of Week**: Orders placed on Mondays and Tuesdays exhibit slightly higher delay rates (~8.5%) compared to weekends (~7.4%), driven by warehouse order dispatch backlog after the weekend.
- **External Shocks**: Longitudinal analysis reveals a massive spike in March 2018 (21.36% late rate) corresponding to nationwide postal courier strikes in Brazil.

### 5.6 Saved EDA Visual Artifacts
Five publication-quality visual charts were generated and persisted to `artifacts/04_eda/figures/`:
1. `target_distribution.png`: Binary class balance overview.
2. `numeric_distributions_and_boxplots.png`: Distribution and boxplots of freight, price, weight, and distance by delivery status.
3. `correlation_heatmap.png`: Pearson correlation matrix across numeric features against `is_late`.
4. `temporal_patterns.png`: Hourly, day-of-week, and longitudinal monthly trend plots.
5. `geographic_late_rates.png`: State-level and interstate logistics friction comparisons.

---

## 6. Feature Engineering & Preprocessing Pipeline (`05_feature_engineering`)

### 6.1 Derived Predictive Features
1. **Haversine Distance (`haversine_distance_km`)**: Great-circle distance between customer and seller postal coordinates.
2. **Temporal Features**: `order_hour`, `order_dayofweek`, `order_month` extracted from `order_purchase_timestamp`.
3. **Operational Lags**:
   - `approval_lag_hrs`: Hours from purchase to payment approval.
   - `carrier_lag_days`: Days from purchase to carrier pickup (`order_delivered_carrier_date - order_purchase_timestamp`).

> **Operational Prediction-Time Boundary Note**:
> `carrier_lag_days` measures warehouse/dispatch delay. This formulation targets an operational checkpoint at **carrier handover time** to alert logistics teams of orders at severe risk before the estimated delivery SLA lapses. Post-delivery columns (`order_delivered_customer_date`, `actual_delivery_days`, `delivery_delay_days`) are strictly dropped.

### 6.2 Scikit-Learn `ColumnTransformer` Pipeline
- **Numeric Pipeline** (18 features): `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`.
- **Categorical Pipeline** (4 features: `customer_state`, `primary_seller_state`, `primary_product_category`, `dominant_payment_type`): `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', sparse=False)`.
- **Fit Isolation**: The pipeline was fitted **strictly on `train_fe`**, generating a final feature dimension of **145 columns**, and exported as `preprocessing_pipeline.joblib`.
- **Feature Name Tracking**: Exported feature taxonomy as `feature_names.txt` (145 features).

---

## 7. Model Training, Tuning & Evaluation (`06_train_tune_evaluate`)

### 7.1 Benchmark Comparison on Validation Set
Models were trained on the training partition (67,529 rows) and benchmarked on the held-out validation set (14,470 orders):

| Model | Class Weight | Val ROC-AUC | Val PR-AUC | Val F1 (Late) | Precision (Late) | Recall (Late) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** (Majority Class) | N/A | 0.5000 | 0.0811 | 0.0000 | 0.0000 | 0.0000 |
| **Logistic Regression** (L2) | `balanced` | 0.7788 | 0.3393 | 0.2800 | 0.1770 | **0.6800** |
| **Tuned Random Forest** | `balanced` | **0.8130** | **0.3617** | **0.3977** | **0.3200** | 0.5200 |

### 7.2 Hyperparameter Tuning
Using 3-fold cross validation with ROC-AUC scoring on the training set:
- **Search Space**: Number of estimators, tree depth, minimum samples split, and leaf size.
- **Best Configuration**:
  - `n_estimators`: 100
  - `max_depth`: 15
  - `min_samples_split`: 10
  - `min_samples_leaf`: 2
  - `class_weight`: `balanced`
  - **Best CV ROC-AUC**: **0.8101**

### 7.3 Final Evaluation on the Held-Out Test Set
The test set (14,471 orders) was evaluated **exactly once** at the very end using the frozen tuned Random Forest model:

| Metric | Validation Set | Test Set | Variance ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.8130 | **0.8156** | $+0.0026$ |
| **PR-AUC** | 0.3617 | **0.3528** | $-0.0089$ |
| **F1-Score (Late Class)** | 0.3977 | **0.3913** | $-0.0064$ |
| **Precision (Late Class)** | 0.3200 | **0.3100** | $-0.0100$ |
| **Recall (Late Class)** | 0.5200 | **0.5200** | $0.0000$ |

The tiny variance between validation and test scores ($\Delta \text{ROC-AUC} = +0.0026$, $\Delta \text{F1} = -0.0064$) confirms excellent generalization stability with zero test-set overfitting.

### 7.4 Feature Importance Analysis
The top 5 predictive features driving Random Forest decisions:
1. `carrier_lag_days` (Days to carrier dispatch) — **38.4%** importance
2. `estimated_delivery_days` (Promised SLA buffer) — **16.2%** importance
3. `haversine_distance_km` (Customer-to-seller distance) — **9.7%** importance
4. `total_freight` (Shipping cost) — **5.1%** importance
5. `approval_lag_hrs` (Payment verification delay) — **3.8%** importance

---

## 8. Business Trade-offs & Production Recommendations

1. **Operational Decision Thresholds**:
   - At the default threshold with `class_weight='balanced'`, the tuned Random Forest captures **52% of all late deliveries** with a precision of **31%**.
   - In production, late deliveries incur severe churn and customer dissatisfaction costs:
     $$\text{Cost}(\text{False Negative}) \gg \text{Cost}(\text{False Positive})$$
   - If proactive intervention (e.g. automated notification or courier priority expedite) costs \$2 while retaining a dissatisfied customer is worth \$50, the decision threshold can be tuned downward (e.g. from 0.50 to 0.35) to boost late delivery recall to >75%.

2. **Early Intervention Point**:
   - Because `carrier_lag_days` is the single most dominant predictor (38.4%), the production inference pipeline should execute at **carrier dispatch time** (or alert sellers when an order has not been dispatched within 48 hours of purchase).

3. **Artifact Integrity Summary**:
   - **Pipeline**: `preprocessing_pipeline.joblib` (7.9 KB, 145 features)
   - **Model**: `best_model.joblib` (12.9 MB)
   - **Metrics**: `metrics.json`
   - **EDA Figures**: 5 saved charts in `artifacts/04_eda/figures/`
