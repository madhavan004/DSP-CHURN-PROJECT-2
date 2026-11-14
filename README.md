# 📡 Telecom Customer Churn MLOps Production Platform

[![CI Pipeline](https://github.com/your-org/telecom-churn/actions/workflows/ci.yml/badge.svg)](https://github.com/your-org/telecom-churn/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Airflow 3.x](https://img.shields.io/badge/Airflow-3.x-teal.svg)](https://airflow.apache.org/)
[![Great Expectations v1.x](https://img.shields.io/badge/Great%20Expectations-v1.x-orange.svg)](https://greatexpectations.io/)
[![Docker Compose](https://img.shields.io/badge/Docker%20Compose-Ready-blue)](https://docs.docker.com/compose/)

An end-to-end, production-grade Machine Learning Operations (MLOps) platform for **Telecom Customer Churn** prediction and monitoring, built for the Data Science in Production (DSP) course.

---

## 🏗️ System Architecture

```
                               ┌──────────────────────────────────────────────┐
                               │               Streamlit WebApp               │
                               │  (Prediction Page & Past Predictions Page)   │
                               └───────────────────────┬──────────────────────┘
                                                       │ HTTP / JSON
                                                       ▼
                               ┌──────────────────────────────────────────────┐
                               │            FastAPI Model Service             │
                               │  (/health, /predict, /past-predictions, ...) │
                               └───────┬──────────────────────────────┬───────┘
                                       │ SQLAlchemy                   │ MLflow Client
                                       ▼                              ▼
                               ┌───────────────┐              ┌───────────────┐
                               │  PostgreSQL   │              │ MLflow Server │
                               │   Database    │              │ Model Registry│
                               └───────▲───────┘              └───────▲───────┘
                                       │ PostgresHook                 │ Tracking
                                       │                              │
┌──────────────────────────────────────┴──────────────────────────────┴──────────────────────────────────────┐
│                                            Airflow 3.x Scheduler                                           │
│                                                                                                            │
│   ┌──────────────────────────────┐ ┌──────────────────────────────┐ ┌───────────────────────────────────┐  │
│   │        Ingestion DAG         │ │        Prediction DAG        │ │           Training DAG            │  │
│   │  (GX Validation, Alerts,     │ │  (Checks good_data, calls    │ │  (Retrains, MLflow Evaluation,    │  │
│   │   Split good & bad data)     │ │   FastAPI batch predict)     │ │   Promotes @champion, Reloads API)│  │
│   └──────────────┬───────────────┘ └──────────────┬───────────────┘ └─────────────────┬─────────────────┘  │
└──────────────────┼────────────────────────────────┼───────────────────────────────────┼────────────────────┘
                   ▼                                ▼                                   ▼
           ┌───────────────┐                ┌───────────────┐                   ┌───────────────┐
           │ Data Folders  │                │ Nginx Server  │                   │    Grafana    │
           │ raw / good /  │                │ (GX Data Docs │                   │ (Data Quality │
           │ bad / archive │                │  HTML Reports)│                   │ & Drift Dash) │
           └───────────────┘                └───────────────┘                   └───────────────┘
```

---

## 🧩 Components Overview

| Component | Technology | Port | Description |
|-----------|------------|------|-------------|
| **Web UI** | Streamlit | `8501` | Multi-page interactive UI for single customer scoring, batch CSV uploads, and past prediction history. |
| **Model API** | FastAPI + Pydantic | `8000` | REST API for inference, persistence, and dynamic model reloading. |
| **Database** | PostgreSQL 15 | `5432` | Stores prediction logs, ingestion data quality statistics, and training drift baselines. |
| **Data Quality** | Great Expectations v1.x | N/A | Validation engine detecting 9+ data quality error types, computing criticality, and generating HTML Data Docs. |
| **Orchestration** | Apache Airflow 3.x | `8085` | Schedules continuous ingestion, batch scoring, and automated retraining workflows. |
| **Model Lifecycle**| MLflow Registry | `5000` | Experiment tracking and registry with `@champion` alias promotion. |
| **Report Serving**| Nginx | `8080` | Serves Great Expectations Data Docs HTML reports with clickable links. |
| **Monitoring** | Grafana | `3000` | Real-time operational dashboards for Data Quality and Model / Feature Drift. |

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Docker Engine & Docker Compose (`docker compose version` >= 2.20)
- Git
- Python 3.12+ (for local scripts/tests)

### 2. Clone and Configure Environment
```bash
git clone <your-repo-url>
cd "telecom churn"

# Copy environment template
cp .env.example .env
```

### 3. Launch All Containers with Docker Compose
```bash
docker compose up -d --build
```

### 4. Verify Services
Check running containers:
```bash
docker compose ps
```

Access Web Interfaces:
- **Streamlit WebApp**: [http://localhost:8501](http://localhost:8501)
- **FastAPI Docs (Swagger UI)**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Airflow Webserver**: [http://localhost:8085](http://localhost:8085) *(User: `admin`, Pass: `admin`)*
- **Grafana Monitoring**: [http://localhost:3000](http://localhost:3000) *(User: `admin`, Pass: `admin`)*
- **MLflow Tracking**: [http://localhost:5000](http://localhost:5000)
- **Data Docs Reports (Nginx)**: [http://localhost:8080](http://localhost:8080)

---

## 🧪 Testing & CI/CD Pipeline

### Local Unit Tests
Run the complete test suite with `pytest`:
```bash
pytest -v tests/
```

### Linting
```bash
flake8 .
```

### GitHub Actions CI/CD
- **CI (`.github/workflows/ci.yml`)**: Triggered on Pull Requests to `develop`. Executes separate steps for `flake8` linting and `pytest` unit tests.
- **CD (`.github/workflows/cd.yml`)**: Triggered on merge to `main`. Automatically drafts semantic version tags (e.g. `v1.0.0`) and creates GitHub Releases.

---

## 📊 Defense Demo Instructions & Checklist

### Preparation Checklist
1. **Repository**: Merge feature branch to `develop`, then create a Pull Request to `main`.
2. **Demo Files**: Ensure the 5 demo files exist in `data/demo_files/`:
   - `demo_1_all_errors.csv` (100% invalid rows $\rightarrow$ High Criticality Alert $\rightarrow$ all rows to `bad_data`)
   - `demo_2_clean_batch1.csv` (Clean batch $\rightarrow$ routed to `good_data`)
   - `demo_3_clean_batch2.csv` (Clean batch $\rightarrow$ routed to `good_data`)
   - `demo_4_clean_batch3.csv` (Clean batch $\rightarrow$ routed to `good_data`)
   - `demo_5_mixed_data.csv` (Mixed batch $\rightarrow$ Medium Criticality $\rightarrow$ splits into `good_data` and `bad_data`)
3. **Pre-populate Dashboards**:
   ```bash
   python scripts/populate_dashboard_data.py
   ```
4. **Clean Active Folders**:
   Make sure `data/good_data/` and `data/bad_data/` are clean before starting live demo.

### Step-by-Step Live Demo Flow (10 Minutes)
1. **Streamlit Prediction Demo**:
   - Open [http://localhost:8501](http://localhost:8501).
   - Show Single Customer Prediction form with risk gauge.
   - Upload CSV batch from `data/demo_files/demo_2_clean_batch1.csv` and show batch scoring results and CSV export.
   - Navigate to **Past Predictions Page** and demonstrate date range filtering and source selection (`webapp` vs `scheduled`).
2. **Data Ingestion & Error Handling Demo**:
   - Copy `data/demo_files/demo_1_all_errors.csv` to `data/raw_data/`.
   - Open Airflow at [http://localhost:8085](http://localhost:8085) and trigger / view `data_ingestion_pipeline`.
   - Show that validation flags high criticality and routes rows to `data/bad_data/`.
   - Open the auto-generated Great Expectations Data Docs link served on Nginx ([http://localhost:8080](http://localhost:8080)).
3. **Data Splitting Demo**:
   - Copy `data/demo_files/demo_5_mixed_data.csv` to `data/raw_data/`.
   - Watch ingestion DAG split valid rows into `good_data/` and corrupted rows into `bad_data/`.
4. **Scheduled Prediction Job Demo**:
   - Trigger `scheduled_prediction_pipeline` in Airflow.
   - Show batch API scoring in action, logging records to the database with `source='scheduled'`.
5. **Grafana Dashboards Demo**:
   - Open [http://localhost:3000](http://localhost:3000) $\rightarrow$ Dashboards $\rightarrow$ `Data Quality Monitoring` (show 4 temporal panels with thresholds).
   - Open `Data Drift & Model Serving Monitor` (show numerical drift, categorical share, churn rate, and prediction probability trends).
6. **Model Retraining & Promotion Demo (Defense 2)**:
   - Trigger `model_training_and_promotion_pipeline` in Airflow.
   - Show candidate model benchmark, MLflow `@champion` alias promotion, API `/reload-model` notification, and file archiving to `data/archived_data/`.

---

## 🛠️ Data Utility Scripts

| Script | Command | Purpose |
|--------|---------|---------|
| **Split Dataset** | `python scripts/split_data.py --num_files 30` | Splits clean dataset into $N$ files with **exactly 10 rows** per file. |
| **Inject Errors** | `python scripts/inject_errors.py --prob 0.1` | Corrupts data with 9 configurable error types across all categories. |
| **Generate Demo Files**| `python scripts/generate_demo_files.py` | Generates the 5 required defense presentation files. |
| **Train Initial Model**| `python scripts/train_initial_model.py` | Trains baseline model pipeline and extracts feature stats. |
| **Populate Dashboards**| `python scripts/populate_dashboard_data.py` | Seeds historical data for Grafana dashboards. |

---

## 📁 Repository Directory Structure

```
telecom-churn/
├── .github/
│   └── workflows/
│       ├── ci.yml                 # Linting (flake8) + Tests (pytest) on PR to develop
│       └── cd.yml                 # Automated release & tag creation on merge to main
├── dags/                          # Apache Airflow 3.x DAGs
│   ├── ingestion_dag.py           # Ingestion + Great Expectations + Alerts + Data Split
│   ├── prediction_dag.py          # Batch Prediction + Skip handling
│   └── training_dag.py            # Retraining + MLflow Registry Benchmark + Promotion
├── data/
│   ├── raw_data/                  # Ingestion input files (10 rows each)
│   ├── good_data/                 # Valid data for scoring and retraining
│   ├── bad_data/                  # Corrupted rows filtered out by validation
│   ├── archived_data/             # Processed data after retraining
│   ├── demo_files/                # 5 required defense demonstration files
│   └── telco_churn_clean.csv      # Clean base dataset
├── great_expectations/            # Great Expectations v1.x Suite
│   ├── great_expectations.yml
│   ├── expectations/
│   │   └── telco_churn_suite.json # Expectation rules JSON
│   └── validator.py               # Validation, criticality & Data Docs engine
├── model_service/                 # FastAPI Service
│   ├── app/
│   │   ├── main.py                # Endpoints: /health, /predict, /past-predictions, /reload-model
│   │   ├── database.py            # SQLAlchemy connection pooling & session
│   │   ├── models.py              # ORM: PredictionRecord, IngestionStatRecord, TrainingStatRecord
│   │   ├── schemas.py             # Pydantic validation schemas
│   │   └── ml_model.py            # ModelManager & MLflow integration
│   ├── model/                     # Trained model artifacts & baseline stats
│   ├── Dockerfile
│   └── requirements.txt
├── monitoring/                    # Grafana Provisioning
│   └── provisioning/
│       ├── datasources/postgres.yml
│       ├── dashboards/
│       │   ├── dashboards.yml
│       │   ├── data_quality.json  # Data Quality Dashboard (4+ panels)
│       │   └── data_drift.json    # Data Drift Dashboard (4+ panels)
│       └── alerting/alerts.yml    # Teams alerting configuration
├── nginx/                         # Nginx Server for GX Data Docs HTML
│   ├── default.conf
│   └── Dockerfile
├── scripts/                       # Automation & Data Engineering Scripts
│   ├── split_data.py
│   ├── inject_errors.py
│   ├── generate_demo_files.py
│   ├── train_initial_model.py
│   ├── populate_dashboard_data.py
│   └── init_postgres.sql
├── tests/                         # Pytest Unit Test Suite
│   ├── test_criticality.py
│   ├── test_data_splitting.py
│   ├── test_error_injection.py
│   ├── test_api_schemas.py
│   └── test_api_endpoints.py
├── webapp/                        # Streamlit Multi-Page App
│   ├── Home.py                    # Landing & System Health
│   ├── pages/
│   │   ├── 1_🔮_Prediction.py     # Single Form & Batch CSV scoring
│   │   └── 2_📜_Past_Predictions.py # Historical audit log & filters
│   ├── Dockerfile
│   └── requirements.txt
├── docker-compose.yml             # Container Orchestration
├── .env.example                   # Environment configuration template
├── .flake8                        # Flake8 lint configuration
├── .gitignore
└── README.md
```
