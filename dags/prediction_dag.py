"""
Prediction DAG — Defense 1

Runs every 2 minutes. Checks `good_data/` for files that have appeared
since the last successful run (tracked via an Airflow Variable holding a
file-modification-time cursor). If there are new files, reads them and
makes a SINGLE batch API call to the model service. If there is nothing
new, the DAG run is skipped entirely (both tasks show as skipped, not
just `make_predictions`), as required by the project spec.

Note: this DAG never deletes or moves files from `good_data/` — that is
the training DAG's job (Defense 2), after a successful model promotion.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from airflow.sdk import Variable, dag, task
from airflow.exceptions import AirflowSkipException

GOOD_DATA_DIR = Path("/opt/airflow/data/good_data")
MODEL_API_URL = os.getenv("MODEL_API_URL", "http://api:8000")
LAST_PROCESSED_VARIABLE = "prediction_dag_last_processed_ts"


def _get_last_processed_ts() -> float:
    try:
        return float(Variable.get(LAST_PROCESSED_VARIABLE))
    except Exception:
        return 0.0


@dag(
    dag_id="prediction_dag",
    description="Make scheduled batch churn predictions on newly ingested good_data.",
    schedule=timedelta(minutes=2),
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["dsp", "prediction"],
)
def prediction_dag():
    @task
    def check_for_new_data() -> dict:
        """
        Look for good_data files newer than the last processed cursor.
        Skips the ENTIRE dag run (not just make_predictions) if none are
        found, by raising AirflowSkipException here.
        """
        GOOD_DATA_DIR.mkdir(parents=True, exist_ok=True)
        last_ts = _get_last_processed_ts()

        new_files = [
            f
            for f in GOOD_DATA_DIR.glob("*.csv")
            if f.stat().st_mtime > last_ts
        ]

        if not new_files:
            raise AirflowSkipException(
                "No new files in good_data/ since last run - skipping."
            )

        new_files.sort(key=lambda f: f.stat().st_mtime)
        max_ts = max(f.stat().st_mtime for f in new_files)

        return {
            "file_paths": [str(f) for f in new_files],
            "max_ts": max_ts,
        }

    @task
    def make_predictions(check_result: dict) -> None:
        """Read all new files and make a SINGLE batch prediction API call.

        Rows with null values in any required feature column are silently
        dropped before the API call.  Such rows can slip through Great
        Expectations when completeness checks fire at the file level while
        the row still has a null in a field that the API schema requires to
        be a specific literal (e.g. 'Yes'/'No').  Sending them would cause
        a 422 Unprocessable Entity that aborts the entire batch.
        """
        file_paths = [Path(p) for p in check_result["file_paths"]]

        dataframes = [pd.read_csv(p) for p in file_paths]
        combined_df = pd.concat(dataframes, ignore_index=True)

        # Required feature columns (customerID is optional).
        required_cols = [
            "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
            "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
            "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
            "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
            "MonthlyCharges", "TotalCharges",
        ]
        cols_to_check = [c for c in required_cols if c in combined_df.columns]
        before = len(combined_df)
        combined_df = combined_df.dropna(subset=cols_to_check)
        dropped = before - len(combined_df)
        if dropped:
            print(
                f"WARNING: Dropped {dropped} row(s) with null required fields "
                f"before prediction (kept {len(combined_df)}/{before})."
            )

        if combined_df.empty:
            raise AirflowSkipException(
                "All rows had null required fields after filtering — nothing to predict."
            )

        # Convert through to_json/loads so every value is a native
        # JSON-safe Python type (no numpy int64/float64 leaking into the
        # request body).
        records = json.loads(combined_df.to_json(orient="records"))

        response = requests.post(
            f"{MODEL_API_URL}/predict",
            json={"features": records, "source": "scheduled"},
            timeout=120,
        )
        response.raise_for_status()
        result = response.json()
        print(
            f"Predicted {result.get('count', 0)} rows from "
            f"{len(file_paths)} file(s): {[p.name for p in file_paths]}"
        )

        # Only advance the cursor after a successful prediction call, so
        # a failure gets retried against the same files next run.
        Variable.set(LAST_PROCESSED_VARIABLE, str(check_result["max_ts"]))

    check_result = check_for_new_data()
    make_predictions(check_result)


prediction_dag()
