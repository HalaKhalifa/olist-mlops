# Task 2 — From Tables to Notebooks

MLOps Training 2026/2027 · Olist MLOps

## Objective

Task 2 moves the Olist data from the PostgreSQL database into a reproducible notebook-based machine-learning workflow.

The goal is to prepare the dataset, engineer domain-specific features, and build an initial machine-learning model for the **late-delivery classification problem** established in Task 1.

Task 2 consists of six numbered notebooks. Each notebook has a single responsibility, executes sequentially, consumes artifacts produced by the previous step, and persists artifacts for subsequent steps.

---

## Starting Point & Database Foundation

Task 1 established the database foundation for this stage:

* **Database Engine**: PostgreSQL `olist`
* **Host / Port**: `localhost:5432` (configured via `.env`)
* **Tables Ingested**: 9 relational tables (`customers`, `orders`, `order_items`, `products`, `sellers`, `order_payments`, `order_reviews`, `geolocation`, `product_category_translation`)
* **Database Schema Reference**: [`docs/erd/olist_erd.png`](../../docs/erd/olist_erd.png)

---

## What Is an Artifact?

An **artifact** is an immutable data, feature, or model file saved by a notebook so that subsequent steps can consume its output without re-running earlier computation or re-querying the database.

Artifacts ensure:
1. **Decoupled execution**: You can work on EDA, feature engineering, or modeling without re-querying the database each time.
2. **Reproducibility**: Exact intermediate states are saved and tracked.
3. **Data consistency**: Schema and data types are preserved across steps using Parquet format.

---

## Pipeline Overview & Status

| Notebook | Purpose | Input | Output Artifacts | Status |
| :--- | :--- | :--- | :--- | :--- |
| **01 — Read & Join Tables** | Connect to PostgreSQL, inspect all 9 tables, aggregate 1-to-many tables, and join into 1 row per order | PostgreSQL `olist` DB | `01_read_and_join/joined_orders.parquet`<br>`01_read_and_join/joined_orders.csv` | **Completed** |
| **02 — Create Labels** | Define binary late-delivery target (`is_late`), analyze label distribution, and filter invalid records | `01_read_and_join/` | `02_create_labels/labeled_orders.parquet` | **Completed** |
| **03 — Split** | Temporal vs. stratified analysis, 70/15/15 stratified random split preserving 8.11% class balance | `02_create_labels/` | `03_split/train.parquet`<br>`03_split/val.parquet`<br>`03_split/test.parquet` | **Completed** |
| **04 — EDA** | Exploratory data analysis strictly on the training set; identify signals, correlations, and anomalies | `03_split/train.parquet` | `04_eda/eda_summary.md`<br>`04_eda/figures/` | **Completed** |
| **05 — Feature Engineering** | Create domain features, encode categoricals, handle scaling & missing values (fit on train only) | `03_split/` + EDA findings | `05_feature_engineering/X_train.parquet`, etc.<br>`05_feature_engineering/preprocessing_pipeline.joblib` | **Completed** |
| **06 — Train, Tune & Evaluate** | Train baseline and ML models (Logistic Regression, Random Forest), hyperparameter tuning, test set evaluation | `05_feature_engineering/` | `06_train_tune_evaluate/best_model.joblib`<br>`06_train_tune_evaluate/metrics.json` | **Completed** |

---

## Directory Structure

```text
tasks/task-02-notebooks/
├── README.md
├── notebooks/
│   ├── 01_read_and_join_tables.ipynb        # [Completed] Ingestion, grain alignment & join
│   ├── 02_create_labels.ipynb               # Target creation (is_late)
│   ├── 03_train_validation_test_split.ipynb # Dataset splitting strategy
│   ├── 04_eda.ipynb                         # Exploratory data analysis on train set
│   ├── 05_feature_engineering.ipynb         # Feature transformers & pipeline
│   └── 06_train_tune_evaluate.ipynb         # Baseline vs tuned model evaluation
│
├── artifacts/
│   ├── 01_read_and_join/
│   │   ├── joined_orders.parquet            # Master ML table (99,441 rows x 33 columns)
│   │   └── joined_orders.csv                # Master ML table (CSV format)
│   ├── 02_create_labels/
│   │   └── labeled_orders.parquet           # Labeled dataset (96,470 rows x 37 columns)
│   ├── 03_split/
│   │   ├── train.parquet                    # Train split (67,529 rows, 70%)
│   │   ├── val.parquet                      # Validation split (14,470 rows, 15%)
│   │   └── test.parquet                     # Test split (14,471 rows, 15%)
│   ├── 04_eda/
│   ├── 05_feature_engineering/
│   └── 06_train_tune_evaluate/
│
└── report/
```

---

## Step 1 Deep Dive: Read & Join Tables

### Key Design Decisions & Grain Alignment

1. **Grain Definition**:
   - The ML prediction task is at the **order level** (`order_id`).
   - Every row in the final master table represents exactly **1 unique order**.
   - Base table: `orders` (99,441 rows).

2. **Pre-Join Aggregations to Prevent Fan-Out**:
   - **`order_items`**: Multiple items per order are aggregated into order-level metrics:
     - `item_count`, `total_price`, `avg_item_price`, `total_freight`, `avg_item_freight`, `total_weight_g`, `total_volume_cm3`, `num_sellers`.
     - Primary seller and product category are selected based on the highest-priced item in the order.
   - **`order_payments`**: Multiple payments / installments per order are aggregated into:
     - `total_payment_value`, `payment_installments_max`, `payment_transactions_count`, `dominant_payment_type`.
   - **`geolocation`**: Over 1,000,000 coordinate rows are collapsed into median `geo_lat` and `geo_lng` per `zip_code_prefix`. Geocodes are mapped for both customer and primary seller.

3. **Data & Target Leakage Prevention**:
   - **`order_reviews`** is inspected to understand survey metrics, but is **intentionally excluded** from the master ML feature table because reviews are submitted *after* delivery occurs. Including post-delivery feedback would introduce severe target leakage.

4. **Master Table Schema (33 Columns)**:
   - **Identifiers & Keys**: `order_id`, `customer_id`, `customer_unique_id`, `primary_seller_id`
   - **Timestamps**: `order_purchase_timestamp`, `order_approved_at`, `order_delivered_carrier_date`, `order_delivered_customer_date`, `order_estimated_delivery_date`
   - **Order Attributes**: `order_status`
   - **Customer Info**: `customer_zip_code_prefix`, `customer_city`, `customer_state`, `customer_lat`, `customer_lng`
   - **Seller Info**: `primary_seller_zip_code`, `primary_seller_city`, `primary_seller_state`, `seller_lat`, `seller_lng`, `num_sellers`
   - **Product & Items**: `item_count`, `total_price`, `avg_item_price`, `total_freight`, `avg_item_freight`, `total_weight_g`, `total_volume_cm3`, `primary_product_category`
   - **Payment Info**: `total_payment_value`, `payment_installments_max`, `payment_transactions_count`, `dominant_payment_type`

---

## Step 2 Deep Dive: Create Labels & Target Eligibility

### Target Definition
The late delivery classification target **`is_late`** is derived by comparing the actual customer delivery timestamp ($T_{\text{delivered}}$) against the delivery SLA promised at checkout ($T_{\text{estimated}}$):

$$\text{delay\_days} = \frac{T_{\text{delivered}} - T_{\text{estimated}}}{86,400\text{ seconds}}$$

$$\text{is\_late} = \begin{cases} 
1, & \text{if } T_{\text{delivered}} > T_{\text{estimated}} \\ 
0, & \text{otherwise} 
\end{cases}$$

Where:
- $T_{\text{delivered}}$ is `order_delivered_customer_date`
- $T_{\text{estimated}}$ is `order_estimated_delivery_date`

### Population Eligibility Filtering
- Non-delivered orders (`canceled`, `shipped`, `processing`, or `unavailable`) and records with missing delivery dates were filtered out.
- Orders where `order_delivered_customer_date < order_purchase_timestamp` were flagged and dropped as physical logging anomalies.
- **Eligible population**: **96,470 orders** (from 99,441 raw orders).

### Class Imbalance Diagnosis
- **On-Time (`is_late = 0`)**: 88,644 orders (**91.89%**)
- **Late (`is_late = 1`)**: 7,826 orders (**8.11%**)
- **Imbalance Ratio**: **11.33 : 1**
- **Evaluation Takeaway**: Accuracy is a misleading metric (a trivial majority-class classifier scores 91.89% accuracy with 0% late recall). Downstream model selection must prioritize **ROC-AUC**, **PR-AUC**, and minority class **F1 / Recall**.
- **Artifact**: `02_create_labels/labeled_orders.parquet` (96,470 rows $\times$ 37 columns).

---

## Step 3 Deep Dive: Train / Validation / Test Split Strategy

### Split Strategy Evaluation
We compared two candidates for dataset partitioning:
1. **Temporal Cutoff Split**:
   - Training on historical orders, evaluating on future quarters.
   - *Finding*: Olist operations experienced a severe external shock in March 2018 due to nationwide Brazilian postal courier strikes (`Correios`), causing late delivery rates to temporarily surge to **21.36%**. A temporal cutoff split creates severe covariate and prior probability shift between partitions (Train: 9.03%, Val: 5.34%, Test: 6.61%), distorting validation reliability.
2. **Stratified Random Split (Selected Strategy)**:
   - Partitions orders randomly while strictly preserving the empirical label distribution across all partitions.
   - Partition ratio: **70% Train (67,529 rows)**, **15% Validation (14,470 rows)**, **15% Test (14,471 rows)** with `random_state=42`.
   - All three partitions maintain exactly **8.11% late rate** ($\pm 0.00\%$) with zero order ID overlap.

### Leakage Firewall
- `val.parquet` and `test.parquet` are strictly quarantined.
- All exploratory data analysis, imputation, scaling, and encoding parameters are derived **exclusively from `train.parquet`**.
- The test set is held untouched until final model evaluation in Notebook 6.

---

## Step 4 Deep Dive: Exploratory Data Analysis (EDA)

Conducting EDA strictly on the 67,529 training rows revealed key predictive signals:
1. **Geographic Distance (`haversine_distance_km`)**:
   - Haversine great-circle distance between customer and seller coordinates correlates positively with late deliveries ($r = +0.071, p < 10^{-10}$).
2. **Interstate Logistics Friction**:
   - Deliveries crossing state lines face over **2.4x higher delay rate** (**9.92%** interstate vs. **4.09%** intrastate).
3. **Regional Disparities**:
   - North/Northeastern states (`RJ`: 13.4%, `BA`: 14.2%, `MA`: 15.6%) suffer higher delay rates than São Paulo (`SP`: 5.8%).
4. **Estimated Window Buffer (`estimated_delivery_days`)**:
   - Longer promised SLA buffers correlate negatively with late deliveries ($r = -0.059$).
5. **Missing Value Signals**:
   - Orders with unmapped coordinates experience a **12.2% delay rate** vs. 8.1% for mapped orders (+4.1% difference), indicating remote zip codes face higher logistics friction.
6. **Artifacts**:
   - Summary: `04_eda/eda_summary.md`
   - Saved visual charts in `04_eda/figures/`:
     - `target_distribution.png`: Class balance overview.
     - `numeric_distributions_and_boxplots.png`: Feature distributions and IQR outlier bounds.
     - `correlation_heatmap.png`: Pearson correlation matrix against `is_late`.
     - `temporal_patterns.png`: Hourly, daily, and longitudinal monthly trend plots.
     - `geographic_late_rates.png`: State-level and interstate delay comparisons.

---

## Step 5 Deep Dive: Feature Engineering & Preprocessing Pipeline

### Derived Predictive Features
1. **Haversine Distance**: Great-circle distance in kilometers calculated from customer and seller coordinates:
   $$d = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos \phi_1 \cos \phi_2 \sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
2. **Temporal Features**: `order_hour`, `order_dayofweek`, `order_month` extracted from `order_purchase_timestamp`.
3. **Operational Lags**:
   - `approval_lag_hrs`: Hours from purchase to payment approval.
   - `carrier_lag_days`: Days from purchase to carrier pickup (measures dispatch delay at carrier handover time).

### Leakage Pruning
Post-delivery timestamps and targets (`order_delivered_customer_date`, `actual_delivery_days`, `delivery_delay_days`, `order_status`) and high-cardinality IDs are strictly dropped.

### Scikit-Learn `ColumnTransformer` Pipeline
- **Numeric Pipeline** (18 features): `SimpleImputer(strategy='median')` $\rightarrow$ `StandardScaler()`.
- **Categorical Pipeline** (4 features: `customer_state`, `primary_seller_state`, `primary_product_category`, `dominant_payment_type`): `SimpleImputer(strategy='most_frequent')` $\rightarrow$ `OneHotEncoder(handle_unknown='ignore')`.
- **Fit Isolation**: Fitted **strictly on `train_fe`**, generating a final feature dimension of **145 columns**.
- **Artifacts**:
   - `05_feature_engineering/X_train.parquet`, `X_val.parquet`, `X_test.parquet`
   - `05_feature_engineering/y_train.parquet`, `y_val.parquet`, `y_test.parquet`
   - `05_feature_engineering/preprocessing_pipeline.joblib` (7.9 KB)
   - `05_feature_engineering/feature_names.txt` (145 features)

---

## Step 6 Deep Dive: Model Training, Tuning & Evaluation

### Benchmark Comparison on Validation Set
Models were trained on the training split and benchmarked on the held-out validation set (14,470 orders):

| Model | Class Weight | Val ROC-AUC | Val PR-AUC | Val F1 (Late) | Precision (Late) | Recall (Late) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Baseline** (Majority Class) | N/A | 0.5000 | 0.0811 | 0.0000 | 0.0000 | 0.0000 |
| **Logistic Regression** (L2) | `balanced` | 0.7788 | 0.3393 | 0.2800 | 0.1770 | 0.6800 |
| **Tuned Random Forest** | `balanced` | **0.8130** | **0.3617** | **0.3977** | **0.3200** | **0.5200** |

### Hyperparameter Tuning
Using 3-fold cross validation with ROC-AUC scoring on the training set:
- **Best Configuration**:
  - `n_estimators`: 100
  - `max_depth`: 15
  - `min_samples_split`: 10
  - `min_samples_leaf`: 2
  - `class_weight`: `balanced`

### Final Evaluation on Held-Out Test Set
The test set (14,471 orders) was touched **exactly once** at the very end using the frozen tuned Random Forest model:

| Metric | Validation Set | Test Set | Generalization Variance ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.8130 | **0.8156** | $+0.0026$ |
| **PR-AUC** | 0.3617 | **0.3528** | $-0.0089$ |
| **F1-Score (Late Class)** | 0.3977 | **0.3913** | $-0.0064$ |
| **Precision (Late Class)** | 0.3200 | **0.3100** | $-0.0100$ |
| **Recall (Late Class)** | 0.5200 | **0.5200** | $0.0000$ |

- **Artifacts**:
  - `06_train_tune_evaluate/best_model.joblib` (12.9 MB)
  - `06_train_tune_evaluate/metrics.json`

---

## How to Run & Reproduce

### 1. Prerequisites & Environment Setup

Ensure the virtual environment is active and requirements are installed:

```bash
# Verify database connection settings in .env
cat .env
```

Required `.env` keys:
```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=olist
DB_USER=your_username
DB_PASSWORD=your_password
```

### 2. Running Notebooks

Execute all notebooks in strict sequential order from Jupyter or via CLI, as each notebook consumes the artifacts produced by the previous step:

```bash
# Notebook 1: Ingestion, grain alignment & denormalization
jupyter nbconvert --to notebook --execute tasks/task-02-notebooks/notebooks/01_read_and_join_tables.ipynb --inplace

# Notebook 2: Target definition (is_late) & population eligibility filtering
jupyter nbconvert --to notebook --execute tasks/task-02-notebooks/notebooks/02_create_labels.ipynb --inplace

# Notebook 3: Stratified dataset split (train 70%, val 15%, test 15%)
jupyter nbconvert --to notebook --execute tasks/task-02-notebooks/notebooks/03_train_validation_test_split.ipynb --inplace

# Notebook 4: Exploratory data analysis strictly on train set
jupyter nbconvert --to notebook --execute tasks/task-02-notebooks/notebooks/04_eda.ipynb --inplace

# Notebook 5: Feature engineering & ColumnTransformer pipeline fitting (train fit only)
jupyter nbconvert --to notebook --execute tasks/task-02-notebooks/notebooks/05_feature_engineering.ipynb --inplace

# Notebook 6: Baselines, hyperparameter tuning & final test set evaluation
jupyter nbconvert --to notebook --execute tasks/task-02-notebooks/notebooks/06_train_tune_evaluate.ipynb --inplace
```

Or execute the complete end-to-end pipeline in one pass:

```bash
for nb in tasks/task-02-notebooks/notebooks/*.ipynb; do
  echo "Executing $nb..."
  jupyter nbconvert --to notebook --execute "$nb" --inplace
done
```

---

## Task Completion Criteria

Task 2 is complete when:
- [x] **01 — Read & Join Tables**: All 9 tables inspected, multi-row tables aggregated, master ML table created (99,441 rows $\times$ 33 columns), Parquet & CSV artifacts saved.
- [x] **02 — Create Labels**: Target label `is_late` constructed, population filtered (96,470 rows), delivery dynamics & class imbalance analyzed, `labeled_orders.parquet` artifact saved.
- [x] **03 — Dataset Split**: Compared temporal vs stratified splits, implemented 70/15/15 stratified random split (8.11% late rate preserved), saved `train.parquet`, `val.parquet`, `test.parquet`.
- [x] **04 — EDA**: Detailed exploratory analysis conducted on the training set.
- [x] **05 — Feature Engineering**: Preprocessing pipeline fitted on train set only and exported.
- [x] **06 — Train, Tune & Evaluate**: Baseline comparison, hyperparameter tuning, and test set evaluation documented.
- [x] **Task 2 Report**: Written summary explaining split strategy, findings, model performance, and trade-offs.
