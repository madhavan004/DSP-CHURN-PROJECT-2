"""
Great Expectations (GX Core v1.x) suite definition and validation runner
for the Telco Customer Churn ingestion pipeline.

Design notes
------------
- We use a **file-based** GX Data Context (not ephemeral) so that
  Expectation Suites, Validation Definitions, Checkpoints and — most
  importantly — Data Docs HTML reports persist to disk across DAG runs.
  The GX project directory is mounted as a Docker volume and (Defense 2)
  served by nginx.
- Numeric columns are validated twice, deliberately, against two
  different representations of the same raw string column:
    1. The RAW string column -> regex check -> catches wrong-dtype
       errors (e.g. "abc" in a numeric column) without GE raising a
       metric-resolution error on mixed-type columns.
    2. A coerced numeric "shadow" column (`<col>_numeric`, built with
       `pd.to_numeric(errors="coerce")`) -> range check -> catches
       out-of-range values. Non-numeric values become NaN in the shadow
       column and are silently skipped by the range expectation (they
       are already counted as type errors via the regex check), so
       there is no double counting.
- Each expectation carries a `meta={"category": ...}` tag. We use that
  tag (not the expectation type) to bucket failures into the 5 required
  data-quality categories + the 2 extra ones, for the ingestion_stats
  table / Grafana dashboard.
- Missing-column (schema) detection is done in plain pandas *before*
  building the GX suite, since a genuinely missing column can't be
  referenced by a column-level expectation without erroring. We still
  add a `expect_table_columns_to_match_set` expectation so the schema
  check shows up in the Data Docs report.
"""
from pathlib import Path
from typing import Dict, List, Tuple
import os

import great_expectations as gx
import pandas as pd
from great_expectations.checkpoint import UpdateDataDocsAction

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from schema import (  # noqa: E402
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    ID_COLUMN,
    NUMERIC_COLUMNS,
    NUMERIC_RANGES,
)

# Columns that must never be null in an ingestion file.
REQUIRED_NOT_NULL_COLUMNS = [ID_COLUMN, "tenure", "MonthlyCharges", "InternetService"]

# All columns an ingestion file must contain (features + id, no target).
EXPECTED_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS

NUMERIC_STRING_REGEX = r"^-?\d+(?:\.\d+)?$"

GX_PROJECT_DIR = "/opt/airflow/great_expectations"
SUITE_NAME = "telco_ingestion_suite"
DATASOURCE_NAME = "telco_pandas_datasource"
ASSET_NAME = "telco_ingestion_asset"
BATCH_DEFINITION_NAME = "telco_whole_dataframe"
CHECKPOINT_NAME = "telco_ingestion_checkpoint"
VALIDATION_DEFINITION_NAME = "telco_ingestion_validation"


def get_context(project_root_dir: str = None):
    root = project_root_dir or os.getenv("GE_PROJECT_DIR", GX_PROJECT_DIR)
    Path(root).mkdir(parents=True, exist_ok=True)
    return gx.get_context(mode="file", project_root_dir=root)


def _get_or_add_datasource(context):
    return context.data_sources.add_or_update_pandas(name=DATASOURCE_NAME)


def _get_or_add_asset(datasource):
    try:
        return datasource.get_asset(ASSET_NAME)
    except Exception:
        return datasource.add_dataframe_asset(name=ASSET_NAME)


def _get_or_add_batch_definition(asset):
    try:
        return asset.get_batch_definition(BATCH_DEFINITION_NAME)
    except Exception:
        return asset.add_batch_definition_whole_dataframe(BATCH_DEFINITION_NAME)


def build_suite(context) -> gx.ExpectationSuite:
    """
    (Re)build the Expectation Suite from scratch every time so the suite
    definition in code is always the source of truth (idempotent).
    """
    suite = gx.ExpectationSuite(name=SUITE_NAME)
    suite = context.suites.add_or_update(suite)
    # Clear any previously defined expectations before re-adding, so the
    # suite always matches this module exactly.
    for exp in list(suite.expectations):
        suite.delete_expectation(exp)

    # --- Schema (documentation only - actual gating logic is in Python) ---
    suite.add_expectation(
        gx.expectations.ExpectTableColumnsToMatchSet(
            column_set=EXPECTED_COLUMNS,
            exact_match=False,
            meta={"category": "schema"},
        )
    )

    # --- Completeness ---
    for col in REQUIRED_NOT_NULL_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column=col, meta={"category": "completeness"}
            )
        )

    # --- Type (row-level, on RAW string columns) ---
    for col in NUMERIC_COLUMNS:
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToMatchRegex(
                column=col, regex=NUMERIC_STRING_REGEX, meta={"category": "type"}
            )
        )

    # --- Validity (row-level, on coerced numeric shadow columns) ---
    for col in NUMERIC_COLUMNS:
        low, high = NUMERIC_RANGES[col]
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeBetween(
                column=f"{col}_numeric",
                min_value=low,
                max_value=high,
                meta={"category": "validity"},
            )
        )

    # --- Consistency (invalid categorical values) ---
    for col, allowed_values in CATEGORICAL_COLUMNS.items():
        suite.add_expectation(
            gx.expectations.ExpectColumnValuesToBeInSet(
                column=col, value_set=allowed_values, meta={"category": "consistency"}
            )
        )

    # --- Duplicate primary key ---
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeUnique(
            column=ID_COLUMN, meta={"category": "duplicate"}
        )
    )

    return suite


def build_checkpoint(context, suite):
    datasource = _get_or_add_datasource(context)
    asset = _get_or_add_asset(datasource)
    batch_definition = _get_or_add_batch_definition(asset)

    validation_definition = gx.ValidationDefinition(
        name=VALIDATION_DEFINITION_NAME, data=batch_definition, suite=suite
    )
    validation_definition = context.validation_definitions.add_or_update(
        validation_definition
    )

    checkpoint = gx.Checkpoint(
        name=CHECKPOINT_NAME,
        validation_definitions=[validation_definition],
        actions=[UpdateDataDocsAction(name="update_data_docs")],
        result_format="COMPLETE",
    )
    return context.checkpoints.add_or_update(checkpoint)


def _detect_cross_field_inconsistencies(df: pd.DataFrame) -> set:
    """Manual (non-GE) cross-field consistency checks. Returns row indices."""
    bad_indices = set()

    if "PhoneService" in df.columns and "MultipleLines" in df.columns:
        mask = (df["PhoneService"] == "No") & (df["MultipleLines"] == "Yes")
        bad_indices.update(df.index[mask].tolist())

    if "tenure_numeric" in df.columns and "TotalCharges_numeric" in df.columns:
        mask = (df["tenure_numeric"] == 0) & (df["TotalCharges_numeric"] > 0)
        bad_indices.update(df.index[mask].tolist())

    return bad_indices


def validate_dataframe(df_raw: pd.DataFrame) -> Tuple[dict, set, List[str]]:
    """
    Run the full validation pipeline on a single ingestion file (already
    loaded as a pandas DataFrame with every value read as a string).

    Returns:
        stats: dict of per-category error counts + totals, ready to be
               saved to the `ingestion_stats` table.
        invalid_row_indices: set of row indices considered invalid
               (union across every category).
        missing_columns: list of required columns absent from the file.
    """
    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df_raw.columns]

    total_rows = len(df_raw)
    category_counts = {
        "completeness": 0,
        "validity": 0,
        "consistency": 0,
        "schema": 0,
        "type": 0,
        "duplicate": 0,
        "cross_field": 0,
    }

    if missing_columns:
        # File-level schema problem: every row is considered affected,
        # skip granular row-level checks (columns may not even exist).
        category_counts["schema"] = total_rows
        invalid_row_indices = set(df_raw.index.tolist())
        stats = {
            **category_counts,
            "total_rows": total_rows,
            "valid_rows": 0,
            "invalid_rows": total_rows,
        }
        return stats, invalid_row_indices, missing_columns

    # Build numeric shadow columns for range validation.
    df = df_raw.copy()
    for col in NUMERIC_COLUMNS:
        df[f"{col}_numeric"] = pd.to_numeric(df[col], errors="coerce")

    context = get_context()
    suite = build_suite(context)
    checkpoint = build_checkpoint(context, suite)

    result = checkpoint.run(batch_parameters={"dataframe": df})

    invalid_row_indices: set = set()

    for _, validation_result in result.run_results.items():
        for expectation_result in validation_result.results:
            category = expectation_result.expectation_config.meta.get("category")
            if category is None or category == "schema":
                continue  # schema handled separately above

            if expectation_result.success:
                continue

            index_list = expectation_result.result.get("unexpected_index_list")
            if index_list:
                category_counts[category] += len(index_list)
                invalid_row_indices.update(index_list)
            elif expectation_result.exception_info and any(
                info.get("raised_exception")
                for info in expectation_result.exception_info.values()
            ):
                # Defensive fallback: an expectation errored out entirely
                # (should be rare given the shadow-column design). Count
                # it against the whole batch for that category.
                category_counts[category] += expectation_result.result.get(
                    "unexpected_count"
                ) or 0

    cross_field_bad = _detect_cross_field_inconsistencies(df)
    category_counts["cross_field"] = len(cross_field_bad)
    invalid_row_indices.update(cross_field_bad)

    stats = {
        **category_counts,
        "total_rows": total_rows,
        "valid_rows": total_rows - len(invalid_row_indices),
        "invalid_rows": len(invalid_row_indices),
    }

    # Trigger Data Docs build explicitly too (the checkpoint action also
    # does this, kept here for clarity / in case actions are removed).
    context.build_data_docs()

    return stats, invalid_row_indices, missing_columns


def compute_criticality(stats: Dict, missing_columns: List[str]) -> str:
    """High: missing column OR >50% invalid. Medium: 10-50%. Low: 0-10%. none: 0."""
    if missing_columns:
        return "high"

    total = stats["total_rows"]
    invalid = stats["invalid_rows"]
    if total == 0:
        return "none"

    pct_invalid = invalid / total
    if pct_invalid > 0.5:
        return "high"
    if pct_invalid >= 0.10:
        return "medium"
    if pct_invalid > 0:
        return "low"
    return "none"


def get_latest_data_docs_report(project_root_dir: str = None) -> str:
    """
    Return the path (relative to the GX project root) of the most recently
    generated validation result HTML page, for inclusion in Teams alerts
    and the `ingestion_stats.report_file_name` column. Falls back to the
    Data Docs index page if no validation result page is found yet.

    Defense 1: this is stored as plain text (nginx isn't part of the
    stack yet). Defense 2 mounts the GX project dir into nginx so this
    path becomes a clickable link.
    """
    root = Path(project_root_dir or os.getenv("GE_PROJECT_DIR", GX_PROJECT_DIR))
    local_site_dir = root / "gx" / "uncommitted" / "data_docs" / "local_site"
    validations_dir = local_site_dir / "validations"

    if validations_dir.exists():
        html_files = list(validations_dir.rglob("*.html"))
        if html_files:
            latest = max(html_files, key=lambda p: p.stat().st_mtime)
            return str(latest.relative_to(root))

    return str((local_site_dir / "index.html").relative_to(root))
