# Task 2 — From Tables to Notebooks

MLOps Training 2026/2027 · Olist MLOps

## Objective

Task 2 moves the Olist data from the PostgreSQL database into a reproducible notebook-based machine-learning workflow.

The goal is to prepare the data and build an initial machine-learning model for the late-delivery classification problem established in Task 1.

Task 2 consists of six numbered notebooks. Each notebook has one responsibility, runs in order, reads the artifacts produced by the previous step, and saves artifacts for the next step.

## Starting Point

Task 1 established the database foundation for this stage:

* PostgreSQL database: `olist`
* Database host: `localhost`
* Database port: `5432`
* Database contains nine Olist dataset tables
* Table relationships and JOIN operations were validated
* Delivery-related timestamps required for the late-delivery problem were confirmed

The Task 1 implementation is available in:

```text
tasks/task-01-database/
```

The database schema is also documented in:

```text
docs/erd/olist_erd.png
```

## What Is an Artifact?

An artifact is a file saved by a notebook so that the next step can use its output without running the previous notebook again.

Examples include:

* Joined ML tables
* Labeled datasets
* Train, validation, and test datasets
* Saved charts
* Statistics summaries
* Feature lists
* Fitted preprocessing objects
* Trained models
* Evaluation results

The artifact directories are organized by notebook so that the data flow between the six steps remains clear and reproducible.

## Notebook Pipeline

| Notebook                             | Purpose                                                                                             | Main Input                    | Main Output                                          |
| ------------------------------------ | --------------------------------------------------------------------------------------------------- | ----------------------------- | ---------------------------------------------------- |
| 01 — Read & Join Tables              | Read and inspect the database tables, aggregate one-to-many tables, and create one ML row per order | PostgreSQL `olist` database   | One ML table                                         |
| 02 — Create Labels                   | Create the late/on-time delivery label and inspect its distribution                                 | ML table                      | Labeled table                                        |
| 03 — Train / Validation / Test Split | Decide on a split strategy and create the dataset splits                                            | Labeled table                 | Train, validation, and test files                    |
| 04 — EDA                             | Perform detailed exploratory analysis on the training data and record findings                      | Training split                | Saved charts and findings summary                    |
| 05 — Feature Engineering             | Build features based on the EDA findings and fit preprocessing transformations on training data     | Dataset splits + EDA findings | Feature table, fitted transformers, and feature list |
| 06 — Train, Tune & Evaluate          | Establish a baseline, train and tune a model, and perform the final evaluation                      | Engineered features           | Trained model and results summary                    |

## Directory Structure

```text
task-02-notebooks/
├── README.md
│
├── notebooks/
│   ├── 01_read_and_join_tables.ipynb
│   ├── 02_create_labels.ipynb
│   ├── 03_train_validation_test_split.ipynb
│   ├── 04_eda.ipynb
│   ├── 05_feature_engineering.ipynb
│   └── 06_train_tune_evaluate.ipynb
│
├── artifacts/
│   ├── 01_read_and_join/
│   ├── 02_create_labels/
│   ├── 03_split/
│   ├── 04_eda/
│   ├── 05_feature_engineering/
│   └── 06_train_tune_evaluate/
│
└── report/
```

## Artifact Flow

```text
PostgreSQL Database
        │
        ▼
01 — Read & Join Tables
        │
        ▼
01_read_and_join/
        │
        ▼
02 — Create Labels
        │
        ▼
02_create_labels/
        │
        ▼
03 — Train / Validation / Test Split
        │
        ▼
03_split/
        │
        ▼
04 — EDA
        │
        ▼
04_eda/
        │
        ▼
05 — Feature Engineering
        │
        ▼
05_feature_engineering/
        │
        ▼
06 — Train, Tune & Evaluate
        │
        ▼
06_train_tune_evaluate/
```

## Task Completion Criteria

Task 2 is complete when:

* Six notebooks are implemented and run in order from a clean start.
* Each notebook performs one defined job.
* Each notebook reads the artifacts from the previous step where applicable.
* Each notebook saves the artifacts required by the next step.
* The data split strategy can be explained and justified.
* A first model result is compared against a simple baseline.
* The final test set is used only at the end of the evaluation process.

## Notes

Production Python scripts are not part of this task. The notebooks should remain organized and reproducible because they will later provide the basis for production scripts in a subsequent stage.
