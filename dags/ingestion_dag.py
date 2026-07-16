"""
Ingestion DAG — Defense 1

Runs every minute. Picks one random file from `raw_data/`, validates it
with Great Expectations, saves summary statistics to PostgreSQL, sends a
Teams alert for medium/high criticality issues, and splits the file into
`good_data/` and/or `bad_data/` depending on the validation outcome.

Tasks `save_statistics`, `send_alerts` and `split_and_save_data` all run
in parallel after `validate_data`, as required by the project spec.
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sdk import Variable, dag, task
from airflow.exceptions import AirflowSkipException

sys.path.append(str(Path(__file__).resolve().parent / "utils"))
sys.path.append(str(Path(__file__).resolve().parent.parent / "scripts"))

from ge_suite import (  # noqa: E402
    compute_criticality,
    get_latest_data_docs_report,
    validate_dataframe,
)
from split_logic import split_dataframe  # noqa: E402
from teams_alerts import send_teams_alert  # noqa: E402

RAW_DATA_DIR = Path("/opt/airflow/data/raw_data")
GOOD_DATA_DIR = Path("/opt/airflow/data/good_data")
BAD_DATA_DIR = Path("/opt/airflow/data/bad_data")

POSTGRES_CONN_ID = "dsp_postgres"

ERROR_CATEGORIES = [
    "completeness",
    "validity",
    "consistency",
    "schema",
    "type",
    "duplicate",
    "cross_field",
]


def _get_variable(name: str) -> str | None:
    """Read an Airflow Variable, returning None if it isn't set."""
    try:
        return Variable.get(name)
    except Exception:
        return None


@dag(
    dag_id="ingestion_dag",
    description="Validate newly ingested Telco Churn data files with Great Expectations.",
    schedule="* * * * *",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=30)},
    tags=["dsp", "ingestion", "data-quality"],
)
def ingestion_dag():
    @task
    def read_data() -> dict:
        """Read one random raw_data file (delete it after reading)."""
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(RAW_DATA_DIR.glob("*.csv"))
        if not files:
            raise AirflowSkipException(
                "No new files in raw_data/, nothing to ingest this run."
            )

        file_path = random.choice(files)
        # dtype=str keeps every value as-read (or NaN for blanks), so null
        # / regex-based checks behave predictably regardless of column mix.
        df = pd.read_csv(file_path, dtype=str)
        payload = {
            "file_name": file_path.name,
            "columns": list(df.columns),
            "records": df.to_dict(orient="records"),
        }

        file_path.unlink()  # avoid re-processing this file
        return payload

    @task
    def validate_data(read_result: dict) -> dict:
        """Run the Great Expectations suite and determine criticality."""
        df = pd.DataFrame.from_records(
            read_result["records"], columns=read_result["columns"]
        )
        stats, invalid_row_indices, missing_columns = validate_dataframe(df)
        criticality = compute_criticality(stats, missing_columns)
        report_path = get_latest_data_docs_report()

        return {
            "file_name": read_result["file_name"],
            "columns": read_result["columns"],
            "records": read_result["records"],
            "stats": stats,
            "invalid_row_indices": sorted(invalid_row_indices),
            "missing_columns": missing_columns,
            "criticality": criticality,
            "report_path": report_path,
        }

    @task
    def save_statistics(validate_result: dict) -> None:
        """Save summary statistics to the ingestion_stats table (PostgresHook)."""
        stats = validate_result["stats"]
        hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        sql = """
            INSERT INTO ingestion_stats (
                file_name, total_rows, valid_rows, invalid_rows, criticality,
                completeness_errors, validity_errors, consistency_errors,
                schema_errors, type_errors, duplicate_errors, cross_field_errors,
                report_file_name
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        hook.run(
            sql,
            parameters=(
                validate_result["file_name"],
                stats["total_rows"],
                stats["valid_rows"],
                stats["invalid_rows"],
                validate_result["criticality"],
                stats["completeness"],
                stats["validity"],
                stats["consistency"],
                stats["schema"],
                stats["type"],
                stats["duplicate"],
                stats["cross_field"],
                validate_result["report_path"],
            ),
        )

    @task
    def send_alerts(validate_result: dict) -> None:
        """Send a Teams alert for medium/high criticality only (no spam)."""
        criticality = validate_result["criticality"]
        if criticality not in ("medium", "high"):
            return

        stats = validate_result["stats"]
        lines = [
            f"**File:** {validate_result['file_name']}",
            f"**Criticality:** {criticality.upper()}",
            f"**Invalid rows:** {stats['invalid_rows']} / {stats['total_rows']}",
        ]
        if validate_result["missing_columns"]:
            lines.append(
                "**Missing columns:** "
                + ", ".join(validate_result["missing_columns"])
            )
        error_summary = ", ".join(
            f"{cat}={stats[cat]}" for cat in ERROR_CATEGORIES if stats.get(cat)
        )
        if error_summary:
            lines.append(f"**Errors by category:** {error_summary}")
        lines.append(f"**Report:** {validate_result['report_path']}")

        send_teams_alert(
            webhook_url=_get_variable("TEAMS_WEBHOOK_DATA_QUALITY"),
            title=f"Data Quality Alert - {criticality.upper()}",
            text="\n\n".join(lines),
            color="FF0000" if criticality == "high" else "FFA500",
        )

    @task
    def split_and_save_data(validate_result: dict) -> None:
        """Route rows to good_data/ and/or bad_data/ based on validation."""
        GOOD_DATA_DIR.mkdir(parents=True, exist_ok=True)
        BAD_DATA_DIR.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame.from_records(
            validate_result["records"], columns=validate_result["columns"]
        )
        file_name = validate_result["file_name"]
        stem = Path(file_name).stem

        good_df, bad_df = split_dataframe(df, validate_result["invalid_row_indices"])

        if good_df is not None and bad_df is None:
            good_df.to_csv(GOOD_DATA_DIR / file_name, index=False)
        elif bad_df is not None and good_df is None:
            bad_df.to_csv(BAD_DATA_DIR / file_name, index=False)
        else:
            good_df.to_csv(GOOD_DATA_DIR / f"{stem}_good.csv", index=False)
            bad_df.to_csv(BAD_DATA_DIR / f"{stem}_bad.csv", index=False)

    read_result = read_data()
    validate_result = validate_data(read_result)

    # Fan out: these three tasks all depend only on validate_data and run
    # in parallel with each other.
    save_statistics(validate_result)
    send_alerts(validate_result)
    split_and_save_data(validate_result)


ingestion_dag()
