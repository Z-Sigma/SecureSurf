# 🛡️ SecureSurf: Malicious URL Detection Pipeline

SecureSurf is a production-grade MLOps pipeline designed to detect malicious URLs (Phishing, Malware, Defacement, Benign) with high accuracy and full reproducibility. This project is **fully Cloud-Agnostic**, meaning it runs anywhere without being locked into a single provider.

---

## 🏗️ MLOps Architecture

```mermaid
graph TD
    A[Raw Data] -->|DVC Tracking| B[MinIO S3 Bucket]
    B -->|Ingestion| C[Modular Pipeline]
    C -->|Feature Engineering| D[Trained Models]
    D -->|Logged with Metrics| E[MLflow Registry]
    E -->|Inference| F[CLI Tool Predict.py]
    F -->|Result| G[Human-Readable Label]
```

### 🌍 The Open-Source Advantage
Instead of using expensive and proprietary cloud services, this project uses a powerful **Open-Source Stack**:
- **MinIO (vs AWS S3)**: Provides an S3-compatible object store that runs entirely on your infrastructure.
- **PostgreSQL (vs AWS RDS)**: A robust, open-source relational database for all MLOps metadata.
- **MLflow**: An open framework for the machine learning lifecycle.
- **DVC (Data Version Control)**: Open-source version control for machine learning projects.

---

## 🏗️ Technical Features
- **Data Versioning (DVC)**: Tracks the `malicious_phish.csv` dataset, storing artifacts in MinIO to ensure your data stays private and versioned.
- **Experiment Tracking (MLflow)**: Automatically logs hyperparameters, accuracy metrics, feature importance, and model files for every training run.
- **Infrastructure (Docker)**: A full stack including MLflow, MinIO, and PostgreSQL, all orchestrated with Docker Compose.
- **Security & Sanitization**: Robust environment variable management via `.env` to keep your credentials safe and off GitHub.

---

## 🚀 Quick Start (on a new machine)

### 1. Requirements
Ensure you have **Python 3.10+** and **Docker Desktop** installed.

### 2. Configure Environment Secrets
Create a file named **`.env`** in the root directory and add your secure credentials (see `.env.example` for a template):
```bash
# MinIO & S3 Credentials
AWS_ACCESS_KEY_ID=securesurf_admin
AWS_SECRET_ACCESS_KEY=YourComplexPasswordHere
MLFLOW_S3_ENDPOINT_URL=http://localhost:9000

# Database Credentials
POSTGRES_USER=mlflow_user
POSTGRES_PASSWORD=YourComplexPasswordHere
POSTGRES_DB=mlflow_db
```

### 3. Infrastructure Setup
Spin up the local MLOps stack (MLflow, MinIO, Postgres):
```powershell
docker-compose up -d
```

### 4. Initialize Virtual Environment
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 5. Setup Storage Buckets & Data
Ensure the necessary S3 buckets exist in MinIO and pull the tracked data:
```powershell
python scripts/create_bucket.py
dvc remote modify minio access_key_id securesurf_admin --local
dvc remote modify minio secret_access_key YourComplexPasswordHere --local
dvc pull
```

---

## 🧪 Training & Inference

### Run the Pipeline
Train all 6 pre-configured models (Random Forest, AdaBoost, etc.) and log them to MLflow:
```powershell
python main.py
```

### Test a URL (Inference)
Use the `predict.py` CLI tool with a `run_id` from your MLflow dashboard (`http://localhost:5000`):
```powershell
python -m src.predict --url "google.com" --run_id "YOUR_RUN_ID"
```

---

## 📁 Project Structure

```bash
├── .github/workflows/   # CI/CD Pipeline
├── scripts/             # Utility scripts (bucket creation, etc.)
├── src/                 # Modular package
│   ├── config.py        # Central configuration & label maps
│   ├── feature_eng.py   # URL feature extraction logic
│   ├── model_trainer.py # MLflow training & logging engine
│   └── predict.py       # Robust CLI inference tool
├── main.py              # Main pipeline orchestrator
├── .env.example         # Template for environment variables
├── docker-compose.yml   # Infrastructure Definition
└── requirements.txt     # Project dependencies
```

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
