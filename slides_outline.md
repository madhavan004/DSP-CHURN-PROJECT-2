# Defense Presentation — 2-Slide Outline & Preparation Guide

According to the official defense guidelines ([5_defense_instructions.md](5_defense_instructions.md)), each team must submit a **2-slide PDF** before the defense.

---

## Slide 1: System Status & Feature Completion
- **Project Title:** Telecom Customer Churn Production MLOps Platform
- **Working Components:**
  - ✅ FastAPI Model Service (`/health`, `/predict`, `/past-predictions`, `/reload-model`)
  - ✅ Streamlit Multi-Page WebApp (Prediction Page with Form + Batch CSV & Past Predictions Audit Log)
  - ✅ PostgreSQL 15 Database (Predictions, Ingestion Stats, Training Baseline Stats)
  - ✅ Data Splitting Script (Strict 10 rows per file guarantee)
  - ✅ Data Error Injection Script (9 error types across Completeness, Validity, Consistency, Schema, Type, Duplicates, Regex, Cross-field, Outliers)
  - ✅ Airflow 3.x Ingestion Pipeline (Great Expectations validation, parallel alerts & stats saving, good/bad data routing)
  - ✅ Airflow 3.x Prediction Pipeline (Batch inference API calling, skipped run status handling)
  - ✅ Airflow 3.x Model Retraining Pipeline (MLflow Registry, candidate benchmarking, `@champion` alias promotion, runtime API reload)
  - ✅ Great Expectations v1.x Data Docs served over Nginx
  - ✅ Grafana Operational Dashboards (Data Quality & Model Drift with Teams alerts)
  - ✅ CI/CD Pipelines (Separate flake8 linting & pytest unit tests on PR to `develop`; automated semantic release on merge to `main`)
- **Non-Working / Unfinished Features:**
  - *None (All Defense 1 and Defense 2 requirements fully implemented and tested!)*

---

## Slide 2: Task Distribution & Team Contributions
*(Fill in team member names according to your group)*

| Team Member | Assigned Responsibilities & Deliverables | Current Defense Focus |
|-------------|------------------------------------------|-----------------------|
| **Member 1** | FastAPI Model Service, SQLAlchemy ORM, Model Reloading endpoint (`/reload-model`) | API & Database Architecture |
| **Member 2** | Streamlit Multi-Page WebApp (`1_🔮_Prediction.py`, `2_📜_Past_Predictions.py`), UX & Visualizations | User Interface & API Integration |
| **Member 3** | Data Engineering scripts (`split_data.py`, `inject_errors.py`), Great Expectations v1.x Suite & Data Docs | Data Validation & QA |
| **Member 4** | Apache Airflow 3.x DAGs (`ingestion_dag.py`, `prediction_dag.py`, `training_dag.py`), Skip Exception handling | Pipeline Orchestration |
| **Member 5** | Grafana Monitoring Dashboards, Teams Alerting, MLflow Model Registry, Docker Compose Orchestration, CI/CD | MLOps Infrastructure & Monitoring |

---

## 📋 Defense Day Checklist
- [ ] Merge code to `main` via PR from `develop`
- [ ] Delete feature branches (keep only `main` and `develop`)
- [ ] Export this 2-slide summary to PDF and submit along with repository link on Microsoft Teams
- [ ] Have Docker containers running on host machine (`docker compose up -d`)
- [ ] Have 5 demo files ready in `data/demo_files/`
- [ ] Keep cameras on during the 15-minute defense slot
