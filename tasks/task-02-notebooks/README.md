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
| **04 — EDA** | Exploratory data analysis strictly on the training set; identify signals, correlations, and anomalies | `03_split/train.parquet` | `04_eda/eda_summary.md`<br>`04_eda/figures/` | Pending |
| **05 — Feature Engineering** | Create domain features, encode categoricals, handle scaling & missing values (fit on train only) | `03_split/` + EDA findings | `05_feature_engineering/X_train.parquet`, etc.<br>`05_feature_engineering/pipeline.joblib` | Pending |
| **06 — Train, Tune & Evaluate** | Train baseline and ML models (LightGBM/XGBoost/RandomForest), hyperparameter tuning, test set evaluation | `05_feature_engineering/` | `06_train_tune_evaluate/model.joblib`<br>`06_train_tune_evaluate/metrics.json` | Pending |

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

Execute notebooks sequentially from Jupyter or CLI:

```bash
# Execute Notebook 1
jupyter nbconvert --to notebook --execute tasks/task-02-notebooks/notebooks/01_read_and_join_tables.ipynb --inplace
```

---

## Task Completion Criteria

Task 2 is complete when:
- [x] **01 — Read & Join Tables**: All 9 tables inspected, multi-row tables aggregated, master ML table created (99,441 rows $\times$ 33 columns), Parquet & CSV artifacts saved.
- [x] **02 — Create Labels**: Target label `is_late` constructed, population filtered (96,470 rows), delivery dynamics & class imbalance analyzed, `labeled_orders.parquet` artifact saved.
- [x] **03 — Dataset Split**: Compared temporal vs stratified splits, implemented 70/15/15 stratified random split (8.11% late rate preserved), saved `train.parquet`, `val.parquet`, `test.parquet`.
- [ ] **04 — EDA**: Detailed exploratory analysis conducted on the training set.
- [ ] **05 — Feature Engineering**: Preprocessing pipeline fitted on train set only and exported.
- [ ] **06 — Train, Tune & Evaluate**: Baseline comparison, hyperparameter tuning, and test set evaluation documented.
- [ ] **Task 2 Report**: Written summary explaining split strategy, findings, model performance, and trade-offs.
