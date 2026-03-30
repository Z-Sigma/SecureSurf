# 🏗️ MLOps Production Pipeline: The Architecture Guide

This document maps out the "Gold Standard" architecture we built for the **Delivery Delay Prediction** project. You can use this as a reference or template for any future production ML system.

---

## 📂 Phase 1: Infrastructure & Data Versioning
**Goal**: Establish a reproducible environment where data and code are decoupled.

1.  **Dockerized Stack**:
    - **PostgreSQL**: Stores experiment metadata (run names, parameters, metrics).
    - **MinIO (S3)**: Stores "Artifacts" (model files, plots, CSVs).
    - **MLflow**: The tracking UI to compare experiments.
2.  **Data Version Control (DVC)**:
    - Never store raw CSVs in Git. Store `.dvc` metadata files in a `dvc/` folder.
    - Use `dvc push` to send raw data to S3. This allows `dvc pull` to recover data on any machine.

---

## ⚙️ Phase 2: Modular Training Pipeline
**Goal**: Transform a research notebook into a production-grade software package.

1.  **Source Code (src/)**:
    - `data_ingestion.py`: Pure logic for loading files.
    - `preprocessing.py`: Cleaning, missing value imputation, and merging.
    - `feature_engineering.py`: Complex transformations (Distance, Time, Encoding).
    - `model_training.py`: Model builders (XGBoost, LighGBM, etc.) with MLflow logs.
    - `config.py`: Centralized paths and credentials.
2.  **Instrumentation**:
    - Use `mlflow.set_tracking_uri()` and `mlflow.set_experiment()`.
    - Log RMSE/MAE/R2 and auto-log model parameters.
3.  **Model Registry**:
    - Promoted the "Best Model" to the **MLflow Model Registry** for deployment versioning.

---

## 🧪 Phase 3: Quality Control & CI/CD
**Goal**: Ensure the pipeline doesn't break when code changes.

1.  **Unit Testing (tests/)**:
    - Write small tests for critical logic (e.g., "does distance calculation return a number?").
    - Use `pytest` for running suites.
2.  **GitHub Actions**:
    - automated `.github/workflows/ci.yml`.
    - Every "Push" triggers a fresh Ubuntu environment that:
        - Installs dependencies.
        - Lints code (`flake8`).
        - Runs unit tests (`pytest`).
    - Success = 🟢 Ready for Merge. Fail = 🔴 No broken code in production.

---

## 🚀 Phase 4: Consumption (Inference)
**Goal**: Use the model without the overhead of the training environment.

1.  **Schema-Aware Inference**:
    - Load model via `mlflow.pyfunc.load_model(models:/name/version)`.
    - Build a wrapper that ensures input data matches the training schema (handling one-hot columns).

---
