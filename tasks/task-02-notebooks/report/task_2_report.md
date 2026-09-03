# Task 2 Report: From Tables to ML Notebooks

**MLOps Engineering Training 2026/2027 · Brazilian E-Commerce (Olist) Late Delivery Prediction**

---

## 1. Executive Summary & Objective

The objective of Task 2 is to build a reproducible, leakage-free machine learning workflow predicting whether a customer's order will arrive **late** (`is_late = 1`) relative to the estimated delivery date promised at checkout. 

This report synthesizes the methodology, engineering decisions, empirical findings, and validation results across the six sequential notebooks:
1. `01_read_and_join_tables.ipynb` — Relational schema ingestion, grain alignment, and multi-table denormalization.
2. `02_create_labels.ipynb` — Target definition, population filtering, and class imbalance diagnosis.
3. `03_train_validation_test_split.ipynb` — Split strategy evaluation (temporal vs. stratified random) and leakage prevention.
4. `04_eda.ipynb` — Exploratory data analysis strictly contained within the training split.
5. `05_feature_engineering.ipynb` — Predictive feature creation and `scikit-learn` pipeline fitting (`fit` on train only).
6. `06_train_tune_evaluate.ipynb` — Dummy baseline, linear model, hyperparameter-tuned ensemble, and single-pass test evaluation.

---

## 2. Relational Ingestion & Grain Alignment (`01_read_and_join_tables`)

### 2.1 The Grain Problem
The Olist database spans 9 relational tables across multiple entities (`orders`, `order_items`, `order_payments`, `customers`, `sellers`, `products`, `geolocation`, `order_reviews`).
- The unit of prediction (the **ML grain**) is an **individual order** (`order_id`).
- In raw relational form, `order_items` contains multiple rows per order (up to 21 items), `order_payments` contains multiple rows per order (up to 29 payment splits), and `geolocation` contains over 1,000,000 coordinate records.
- Direct joins without pre-aggregation produce catastrophic **fan-out**, creating duplicate order rows and distorting training sample distributions.

### 2.2 Pre-Join Aggregations
To enforce a strict 1-to-1 relationship with `orders` (99,441 rows):
- **`order_items`**: Aggregated to `item_count`, `total_price`, `avg_item_price`, `total_freight`, `avg_item_freight`, `total_weight_g`, `total_volume_cm3`, and `num_sellers`. The primary seller and product category were selected based on the highest-priced item.
- **`order_payments`**: Aggregated to `total_payment_value`, `payment_installments_max`, `payment_transactions_count`, and `dominant_payment_type` (by highest transaction value).
- **`geolocation`**: Compressed via median aggregation per 5-digit `zip_code_prefix` to resolve multi-coordinate anomalies, then mapped to both customer and seller zip codes.
- **Leakage Prevention**: `order_reviews` was **deliberately excluded** from the master ML table because customer satisfaction reviews are submitted *post-delivery*, which would introduce severe forward-looking target leakage.

**Result**: A clean master feature table of **99,441 rows $\times$ 33 columns**.

---

## 3. Target Label Creation & Class Balance (`02_create_labels`)

### 3.1 Target Definition
The target variable is defined as:
$$\text{delivery\_delay\_days} = \frac{\text{order\_delivered\_customer\_date} - \text{order\_estimated\_delivery\_date}}{86,400\text{ seconds}}$$

$$\text{is\_late} = \begin{cases} 1 & \text{if } \text{order\_delivered\_customer\_date} > \text{order\_estimated\_delivery\_date} \\ 0 & \text{otherwise} \end{cases}$$

### 3.2 Population Filtering
- Orders with missing delivery dates (`canceled`, `shipped`, `processing`, or `unavailable`) cannot be evaluated against customer delivery reality and were filtered out.
- Orders where `order_delivered_customer_date < order_purchase_timestamp` were flagged and dropped as physical impossibilities / logging corruptions.
- **Eligible population**: **96,470 orders**.

### 3.3 Class Imbalance Diagnosis
- **On-Time (`is_late = 0`)**: 88,644 orders (**91.89%**)
- **Late (`is_late = 1`)**: 7,826 orders (**8.11%**)
- **Imbalance Ratio**: $\approx 11.3 : 1$.
- **Operational Takeaway**: Raw accuracy is misleading (a trivial majority-class classifier scores 91.89% accuracy while catching 0% of late orders). Model optimization and evaluation must prioritize **ROC-AUC**, **PR-AUC**, and minority class **F1-score / Recall**.

---

## 4. Splitting Strategy & Data Leakage Prevention (`03_train_validation_test_split`)

### 4.1 Evaluation of Splitting Strategies
We compared two candidates for dataset partitioning:
1. **Temporal Cutoff Split**:
   - Training on historical orders, testing on future quarters.
   - *Finding*: Olist operations experienced an extreme external shock in March 2018 due to nationwide Brazilian postal courier strikes (`Correios`), causing late delivery rates to temporarily surge past 21%. A pure temporal cutoff split creates severe covariate and prior probability shift between splits, distorting baseline evaluation.
2. **Stratified Random Split (Selected Strategy)**:
   - Partitions orders randomly while strictly preserving the empirical label distribution across all subsets.
   - Stratification ratio: **70% Train (67,529 rows)**, **15% Validation (14,470 rows)**, **15% Test (14,471 rows)**.
   - All three partitions maintain exactly **8.11% late rate** ($\pm 0.00\%$).

### 4.2 Leakage Prevention Firewall
- The validation and test splits were quarantined in separate Parquet files (`val.parquet`, `test.parquet`).
- All exploratory data analysis, outlier trimming decisions, and preprocessor transformations were derived **exclusively from `train.parquet`**.
- The test set was held untouched until the final model evaluation in Notebook 6.

---

## 5. Exploratory Data Analysis Key Findings (`04_eda`)

Conducting EDA strictly on the 67,529 training rows revealed key predictive signals:
1. **Carrier Dispatch Lag (`carrier_lag_days`)**:
   - The time elapsed between order purchase and handover to the freight carrier exhibited the strongest linear and rank correlation with late delivery ($r = +0.216$). Delays in warehouse fulfillment or initial dispatch propagate directly into final delivery failure.
2. **Geographical Distance (`haversine_distance_km`)**:
   - Haversine distance between customer coordinates and primary seller coordinates showed positive correlation ($r = +0.071$).
   - Inter-state deliveries (e.g., Southeast seller to North / Northeast customer) exhibit substantially higher late rates (13–15% in states like `RJ`, `BA`, `MA`, `AL`) compared to local São Paulo deliveries (`SP`: 5.8%).
3. **Delivery Estimate Generosity (`estimated_delivery_days`)**:
   - Olist's estimated delivery buffer varies by region. Orders with tight promised windows have higher probability of breaching SLA.
4. **Volume & Weight**:
   - Bulky freight (`total_weight_g`, `total_volume_cm3`) had slight positive correlations with delays due to specialized carrier routing.

---

## 6. Feature Engineering & Preprocessing Pipeline (`05_feature_engineering`)

### 6.1 Derived Features
1. **Haversine Distance**: Great-circle distance in kilometers calculated from customer and seller coordinates:
   $$d = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos \phi_1 \cos \phi_2 \sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
2. **Temporal Features**: `order_hour`, `order_dayofweek`, `order_month` extracted from `order_purchase_timestamp`.
3. **Operational Lags**:
   - `approval_lag_hrs`: Hours from purchase to payment approval.
   - `carrier_lag_days`: Days from purchase to carrier pickup.

### 6.2 Feature Pruning & Leakage Elimination
Columns dropped prior to pipeline transformation:
- Identifiers: `order_id`, `customer_id`, `customer_unique_id`, `primary_seller_id`.
- Leaky post-delivery timestamps and targets: `order_delivered_customer_date`, `actual_delivery_days`, `delivery_delay_days`, `order_status`.
- Raw coordinate duplicates replaced by haversine distance: `customer_lat`, `customer_lng`, `seller_lat`, `seller_lng`.

### 6.3 Scikit-Learn `ColumnTransformer` Pipeline
- **Numeric Pipeline** (18 features): `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`.
- **Categorical Pipeline** (4 features: `customer_state`, `primary_seller_state`, `primary_product_category`, `dominant_payment_type`): `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore', sparse=False)`.
- **Fit Isolation**: The pipeline was fitted **only on `train_fe`**, generating a final feature dimension of **123 columns**, and exported as `preprocessing_pipeline.joblib`.

---

## 7. Model Training, Tuning & Evaluation (`06_train_tune_evaluate`)

### 7.1 Benchmark Comparison on Validation Set
Models were evaluated on the held-out validation set (14,470 orders):

| Model | Class Weight | Val ROC-AUC | Val PR-AUC | Val F1 (Late) | Precision (Late) | Recall (Late) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** (Majority Class) | N/A | 0.5000 | 0.0811 | 0.0000 | 0.0000 | 0.0000 |
| **Logistic Regression** (L2) | `balanced` | 0.7788 | 0.2814 | 0.3392 | 0.2215 | 0.7289 |
| **Tuned Random Forest** | `balanced` | **0.8130** | **0.3617** | **0.3977** | **0.2941** | **0.6141** |

### 7.2 Hyperparameter Tuning
Using 3-fold cross validation with ROC-AUC scoring on the training set:
- **Best Configuration**:
  - `n_estimators`: 100
  - `max_depth`: 15
  - `min_samples_split`: 10
  - `min_samples_leaf`: 2
  - `class_weight`: `balanced`

### 7.3 Final Evaluation on the Held-Out Test Set
The test set (14,471 orders) was evaluated exactly once using the frozen `Tuned Random Forest` model:

| Metric | Validation Set | Test Set | Variance ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.8130 | **0.8156** | $+0.0026$ |
| **PR-AUC** | 0.3617 | **0.3528** | $-0.0089$ |
| **F1-Score (Late Class)** | 0.3977 | **0.3913** | $-0.0064$ |
| **Precision (Late Class)** | 0.2941 | **0.2882** | $-0.0059$ |
| **Recall (Late Class)** | 0.6141 | **0.6099** | $-0.0042$ |

### 7.4 Feature Importance Analysis
The top 5 predictive features driving Random Forest decisions:
1. `carrier_lag_days` (Days to carrier dispatch) — 38.4% importance
2. `estimated_delivery_days` (Promised SLA buffer) — 16.2% importance
3. `haversine_distance_km` (Customer-to-seller distance) — 9.7% importance
4. `total_freight` (Shipping cost) — 5.1% importance
5. `approval_lag_hrs` (Payment verification delay) — 3.8% importance

---

## 8. Business Trade-offs & Production Recommendations

1. **Threshold Tuning for Operational Intervention**:
   - With `class_weight='balanced'`, the default classification threshold catches **61% of all late deliveries** (Recall = 0.61) with a precision of ~29%.
   - In production, proactive customer notifications or courier expedites incur operational costs. The decision threshold should be calibrated against cost asymmetries:
     $$\text{Cost}(\text{False Negative}) \gg \text{Cost}(\text{False Positive})$$
     If customer churn from a surprise delay costs \$50 while a proactive SMS / coupon costs \$2, the threshold can be lowered to capture >80% of at-risk orders.
2. **Early Intervention Point**:
   - Because `carrier_lag_days` is the single dominant feature, the inference pipeline should run at **carrier dispatch time** (or alert sellers when an order has not been dispatched within 48 hours of purchase).
3. **Artifact Integrity**:
   - Preprocessing pipeline: `preprocessing_pipeline.joblib` (7.9 KB)
   - Model artifact: `best_model.joblib` (13 MB)
   - Performance metrics: `metrics.json`
