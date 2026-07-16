# Telco Customer Churn — MLOps Pipeline (Defense 1)

End-to-end MLOps pipeline for predicting customer churn on the
[Telco Customer Churn dataset](data/telco_full.csv), built for the Data
Science in Production course — **Defense 1** deliverables.

## What's included (Defense 1 scope)

- **Data prep scripts**: split a clean dataset into 10-row ingestion files, and inject 7 configurable types of data-quality errors
- **FastAPI model service**: serves churn predictions (model loaded from a local file), backed by PostgreSQL via SQLAlchemy
- **Streamlit webapp**: single & batch predictions, past-predictions browser — talks to the API only (never touches the model or DB directly)
- **Airflow 3.x DAGs**:
  - `ingestion_dag` (every minute): validates incoming files with Great Expectations, saves stats, sends Teams alerts, splits good/bad data
  - `prediction_dag` (every 2 minutes): batch-predicts newly ingested good data
- **Great Expectations v1.x**: expectation suite covering all 7 error categories, with persisted Data Docs HTML reports
- **Docker Compose**: every service containerized with health checks
- **CI**: GitHub Actions running flake8 + pytest (separate steps) on PRs to `develop`

Defense 2 items (MLflow, `/reload-model`, nginx, Grafana dashboards, CD pipeline) are **not** part of this delivery.

## Project structure

```
project/
├── dags/                     # Airflow DAGs
│   ├── ingestion_dag.py
│   ├── prediction_dag.py
│   ├── utils/                # ge_suite.py, split_logic.py, teams_alerts.py
│   └── tests/                # pytest unit tests (criticality, stats, split logic)
├── data/
│   ├── telco_full.csv        # original clean dataset
│   ├── raw_data/             # files waiting to be ingested
│   ├── good_data/            # valid rows after validation
│   ├── bad_data/             # invalid rows after validation
│   └── archived_data/        # used by the training DAG (Defense 2)
├── model_service/            # FastAPI app
│   ├── app/
│   ├── tests/
│   └── model_artifact/model.joblib
├── webapp/                   # Streamlit app
│   ├── Home.py
│   └── pages/
├── scripts/                  # schema.py, train_initial_model.py, split_data.py, inject_errors.py
├── great_expectations/        # persisted GX project (Data Docs, mounted volume)
├── postgres_init/            # creates the second (Airflow) database
├── docker-compose.yml
├── Dockerfile.airflow
├── .env.example
└── .github/workflows/ci.yml
```

## Prerequisites

- Docker + Docker Compose v2 (`docker compose version`)
- Python 3.12 (only needed if you want to run scripts locally, outside Docker)

## Setup

1. **Copy the env file and adjust if needed:**
   ```bash
   cp .env.example .env
   ```
   Defaults work out of the box for local use. Teams webhook URLs are optional — alerts are simply skipped (logged) if unset.

2. **Build and start everything:**
   ```bash
   docker compose up --build
   ```
   First boot takes a few minutes (building the custom Airflow image with Great Expectations). Services come up in this order thanks to health checks: `db` → `airflow-init` → everything else.

3. **Check everything is healthy:**
   ```bash
   docker compose ps
   ```

4. **Access the services:**
   | Service | URL | Notes |
   |---|---|---|
   | Streamlit webapp | http://localhost:8501 | Prediction + Past Predictions pages |
   | FastAPI docs (Swagger) | http://localhost:8000/docs | Try `/predict`, `/past-predictions` |
   | Airflow UI | http://localhost:8080 | login: `admin` / `admin` (from `.env`) |
   | PostgreSQL | `localhost:5432` (see below) | not exposed by default — see **Database access** |

### Database access

The `db` service doesn't publish a port to the host by default (only containers on the
Docker network can reach it directly at `db:5432`), so there are two ways in:

**Option A — psql inside the container (no setup needed):**
```bash
# App data (predictions, ingestion_stats)
docker exec -it dsp_postgres psql -U dsp_user -d dsp_db

# Airflow metadata
docker exec -it dsp_postgres psql -U dsp_user -d airflow_db
```
Credentials come from `.env` (`POSTGRES_USER` / `POSTGRES_PASSWORD`, default `dsp_user` / `dsp_password`).

Useful queries once connected to `dsp_db`:
```sql
\dt                                   -- list tables
SELECT * FROM predictions ORDER BY created_at DESC LIMIT 10;
SELECT * FROM ingestion_stats ORDER BY created_at DESC LIMIT 10;
```

**Option B — connect from a GUI client (DBeaver, TablePlus, pgAdmin, etc.):**
Add a port mapping for `db` in `docker-compose.yml` (or an override file) so it's reachable from the host:
```yaml
  db:
    ports:
      - "5432:5432"
```
Then connect with:
- Host: `localhost`, Port: `5432`
- User / Password: from `.env` (`POSTGRES_USER` / `POSTGRES_PASSWORD`)
- Database: `dsp_db` (app data) or `airflow_db` (Airflow metadata)

| Table | Database | Contains |
|---|---|---|
| `predictions` | `dsp_db` | Every prediction made (webapp or scheduled), including the feature payload used |
| `ingestion_stats` | `dsp_db` | Per-file data-quality summary stats from `ingestion_dag` (error counts, criticality) |
| Airflow's own tables (`dag_run`, `task_instance`, ...) | `airflow_db` | Managed by Airflow itself — use the Airflow UI instead of querying these directly |

## Generating demo data

The model is already trained and committed (`model_service/model_artifact/model.joblib`). To (re)train it:
```bash
pip install -r model_service/requirements.txt
python scripts/train_initial_model.py
```

To generate files for the ingestion job to pick up:
```bash
# 1. Split the clean dataset into N files of 10 rows each
python scripts/split_data.py --input data/telco_full.csv --output-dir data/raw_data --num-files 30

# 2. (Optional) corrupt some of them in place, to see validation catch errors
python scripts/inject_errors.py --input-dir data/raw_data \
    --completeness-null 0.1 --validity-out-of-range 0.1 --consistency-invalid-cat 0.1 \
    --schema-missing-column 0.05 --type-wrong-dtype 0.1 \
    --duplicate-primary-key 0.05 --cross-field-inconsistency 0.1
```
With both DAGs unpaused in the Airflow UI, `ingestion_dag` picks up one random file per minute, validates it, and routes rows to `good_data/` / `bad_data/`. `prediction_dag` then picks up new `good_data/` files every 2 minutes and calls the API for batch predictions — check the webapp's **Past Predictions** page (source = `scheduled`) to see them.

### Demo file recipe (per the defense instructions)
Prepare 5 files *outside* `raw_data/` ahead of time, then copy them in live during the defense:
- 1 file, all rows corrupted (`inject_errors.py` with high probabilities) → demoes high-criticality alert + everything routed to `bad_data`
- 3 clean files (straight from `split_data.py`) → demoes clean ingestion + scheduled predictions
- 1 mixed file (moderate corruption probabilities) → demoes the good/bad split

## Running tests & linting locally

```bash
# model_service
cd model_service && pip install -r requirements.txt && pip install flake8 pytest
flake8 app/ tests/ --max-line-length=100
pytest tests/ -v

# scripts + dags (from repo root)
pip install pandas numpy great-expectations flake8 pytest
flake8 scripts/ dags/ --max-line-length=100
pytest scripts/tests/ dags/tests/ -v

# webapp
cd webapp && pip install -r requirements.txt && pip install flake8
flake8 . --max-line-length=110
```

CI runs the same checks (lint and test as **separate** steps) on every PR into `develop` — see `.github/workflows/ci.yml`.

## Key design notes

- **Great Expectations**: a single suite (`dags/utils/ge_suite.py`) covers all 7 error categories via `meta={"category": ...}` tags on each expectation, which is how we aggregate per-category counts for `ingestion_stats` without needing 7 separate suites. Numeric columns are checked twice on purpose: a regex check on the raw string column catches wrong-dtype errors, and a range check on a `pd.to_numeric(errors="coerce")` shadow column catches out-of-range values — this avoids Great Expectations raising a metric-resolution error on genuinely mixed-type columns. Data Docs are written to a **file-based** GX context (`great_expectations/gx/...`, a mounted volume) so reports persist across DAG runs and containers.
- **Missing-column (schema) detection** is done in plain pandas before the GX suite runs, since a truly absent column can't be safely referenced by a column-level expectation.
- **Cross-field consistency** (e.g. `PhoneService=No` but `MultipleLines=Yes`) is checked with plain pandas rather than a custom GX Expectation class, to keep things simple for a course project.
- **Prediction DAG "full skip"**: `check_for_new_data` raises `AirflowSkipException` itself (not just a downstream task) when there's nothing new, so the whole run shows as skipped rather than just `make_predictions`.
- **Single Postgres instance**, two databases (`dsp_db`, `airflow_db`) — created via `postgres_init/01_create_airflow_db.sql`, owned by the same superuser so one set of credentials works for both.

## Known limitations / not yet implemented (Defense 2)

- `/reload-model` endpoint, MLflow model registry & training DAG
- nginx (Data Docs are stored on disk with a path recorded in `ingestion_stats.report_file_name`, but there's no clickable link yet)
- Grafana dashboards, alert routing
- CD pipeline / GitHub Releases
