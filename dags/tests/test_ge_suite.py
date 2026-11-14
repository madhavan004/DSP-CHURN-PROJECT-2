import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from ge_suite import (  # noqa: E402
    build_checkpoint,
    build_suite,
    compute_criticality,
    get_context,
    get_latest_data_docs_report,
    validate_dataframe,
)

VALID_ROW = {
    "customerID": "1234-ABCDE",
    "gender": "Female",
    "SeniorCitizen": "0",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": "12",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": "29.85",
    "TotalCharges": "358.2",
}


def make_row(**overrides):
    row = dict(VALID_ROW)
    row.update(overrides)
    return row


@pytest.fixture(autouse=True)
def isolate_gx_project(tmp_path, monkeypatch):
    # Keep each test's GX project (and Data Docs) isolated on disk.
    monkeypatch.setenv("GE_PROJECT_DIR", str(tmp_path / "gx_project"))


# --- Criticality calculation -------------------------------------------------


def test_criticality_none_when_no_errors():
    stats = {"total_rows": 10, "invalid_rows": 0}
    assert compute_criticality(stats, missing_columns=[]) == "none"


def test_criticality_low_under_10_percent():
    stats = {"total_rows": 20, "invalid_rows": 1}  # 5%
    assert compute_criticality(stats, missing_columns=[]) == "low"


def test_criticality_medium_between_10_and_50_percent():
    stats = {"total_rows": 10, "invalid_rows": 3}  # 30%
    assert compute_criticality(stats, missing_columns=[]) == "medium"


def test_criticality_high_above_50_percent():
    stats = {"total_rows": 10, "invalid_rows": 8}  # 80%
    assert compute_criticality(stats, missing_columns=[]) == "high"


def test_criticality_high_when_column_missing_regardless_of_row_errors():
    stats = {"total_rows": 10, "invalid_rows": 0}
    assert compute_criticality(stats, missing_columns=["InternetService"]) == "high"


def test_criticality_none_when_zero_total_rows():
    stats = {"total_rows": 0, "invalid_rows": 0}
    assert compute_criticality(stats, missing_columns=[]) == "none"


def test_criticality_medium_at_boundary_10_percent():
    stats = {"total_rows": 10, "invalid_rows": 1}  # exactly 10%
    assert compute_criticality(stats, missing_columns=[]) == "medium"


# --- Stats computation --------------------------------------------------------


def test_clean_batch_has_zero_invalid_rows():
    df = pd.DataFrame(
        [make_row(customerID=f"1234-ABCD{i}") for i in range(5)]
    )
    stats, invalid_idx, missing = validate_dataframe(df)
    assert missing == []
    assert stats["invalid_rows"] == 0
    assert stats["valid_rows"] == 5
    assert invalid_idx == set()


def test_null_required_column_counted_as_completeness_error():
    rows = [make_row() for _ in range(3)]
    rows[0]["customerID"] = None
    df = pd.DataFrame(rows)
    stats, invalid_idx, missing = validate_dataframe(df)
    assert stats["completeness"] >= 1
    assert 0 in invalid_idx


def test_out_of_range_numeric_counted_as_validity_error():
    rows = [make_row() for _ in range(3)]
    rows[1]["tenure"] = "999"
    df = pd.DataFrame(rows)
    stats, invalid_idx, missing = validate_dataframe(df)
    assert stats["validity"] >= 1
    assert 1 in invalid_idx


def test_invalid_categorical_counted_as_consistency_error():
    rows = [make_row() for _ in range(3)]
    rows[2]["gender"] = "UNKNOWN"
    df = pd.DataFrame(rows)
    stats, invalid_idx, missing = validate_dataframe(df)
    assert stats["consistency"] >= 1
    assert 2 in invalid_idx


def test_missing_column_flags_schema_error_for_whole_file():
    rows = [make_row() for _ in range(4)]
    df = pd.DataFrame(rows).drop(columns=["InternetService"])
    stats, invalid_idx, missing = validate_dataframe(df)
    assert missing == ["InternetService"]
    assert stats["schema"] == 4
    assert stats["invalid_rows"] == 4


def test_non_numeric_value_counted_as_type_error():
    rows = [make_row() for _ in range(3)]
    rows[0]["MonthlyCharges"] = "abc"
    df = pd.DataFrame(rows)
    stats, invalid_idx, missing = validate_dataframe(df)
    assert stats["type"] >= 1
    assert 0 in invalid_idx


def test_duplicate_customer_id_counted_as_duplicate_error():
    rows = [make_row(customerID="dup-1") for _ in range(2)] + [make_row()]
    df = pd.DataFrame(rows)
    stats, invalid_idx, missing = validate_dataframe(df)
    assert stats["duplicate"] >= 1


# --- Cross-field inconsistency -----------------------------------------------


def test_cross_field_phone_service_no_multiple_lines_yes():
    """PhoneService=No + MultipleLines=Yes is an impossible combination."""
    rows = [make_row(customerID=f"CF{i:02d}-CROSS") for i in range(3)]
    rows[1]["PhoneService"] = "No"
    rows[1]["MultipleLines"] = "Yes"
    df = pd.DataFrame(rows)
    stats, invalid_idx, missing = validate_dataframe(df)
    assert stats["cross_field"] >= 1
    assert 1 in invalid_idx


# --- GX Context, Suite & Checkpoint ------------------------------------------


def test_get_context_creates_project_directory(tmp_path):
    """get_context should create the project root if it doesn't exist."""
    project_dir = str(tmp_path / "new_gx_project")
    context = get_context(project_root_dir=project_dir)
    assert context is not None
    assert Path(project_dir).exists()


def test_build_suite_returns_expectation_suite(tmp_path):
    """build_suite should return a GX ExpectationSuite with expectations."""
    context = get_context(project_root_dir=str(tmp_path / "gx_suite_test"))
    suite = build_suite(context)
    assert suite is not None
    assert suite.name == "telco_ingestion_suite"
    assert len(suite.expectations) > 0


def test_build_suite_is_idempotent(tmp_path):
    """Calling build_suite twice should produce the same number of expectations."""
    context = get_context(project_root_dir=str(tmp_path / "gx_idempotent"))
    suite1 = build_suite(context)
    count1 = len(suite1.expectations)
    suite2 = build_suite(context)
    count2 = len(suite2.expectations)
    assert count1 == count2


def test_build_checkpoint_creates_checkpoint(tmp_path):
    """build_checkpoint should return a valid checkpoint object."""
    context = get_context(project_root_dir=str(tmp_path / "gx_cp_test"))
    suite = build_suite(context)
    checkpoint = build_checkpoint(context, suite)
    assert checkpoint is not None
    assert checkpoint.name == "telco_ingestion_checkpoint"


# --- Data Docs generation ----------------------------------------------------


def test_validate_generates_data_docs(tmp_path, monkeypatch):
    """After validation, GX Data Docs HTML should exist on disk."""
    project_dir = str(tmp_path / "gx_docs_test")
    monkeypatch.setenv("GE_PROJECT_DIR", project_dir)

    df = pd.DataFrame(
        [make_row(customerID=f"DOC{i:02d}-TEST") for i in range(3)]
    )
    validate_dataframe(df)

    data_docs_dir = Path(project_dir) / "gx" / "uncommitted" / "data_docs" / "local_site"
    assert data_docs_dir.exists(), "Data Docs directory was not created"
    html_files = list(data_docs_dir.rglob("*.html"))
    assert len(html_files) > 0, "No HTML files generated in Data Docs"


def test_get_latest_data_docs_report_returns_path(tmp_path, monkeypatch):
    """get_latest_data_docs_report should return a path after validation."""
    project_dir = str(tmp_path / "gx_report_test")
    monkeypatch.setenv("GE_PROJECT_DIR", project_dir)

    df = pd.DataFrame(
        [make_row(customerID=f"RPT{i:02d}-TEST") for i in range(3)]
    )
    validate_dataframe(df)

    report_path = get_latest_data_docs_report(project_root_dir=project_dir)
    assert report_path is not None
    assert report_path.endswith(".html")


# --- Stats dict structure ----------------------------------------------------


def test_stats_dict_contains_all_required_keys():
    """The stats dict should contain all error categories plus totals."""
    df = pd.DataFrame(
        [make_row(customerID=f"KEY{i:02d}-CHECK") for i in range(3)]
    )
    stats, _, _ = validate_dataframe(df)

    required_keys = [
        "completeness", "validity", "consistency", "schema",
        "type", "duplicate", "cross_field",
        "total_rows", "valid_rows", "invalid_rows",
    ]
    for key in required_keys:
        assert key in stats, f"Missing key '{key}' in stats dict"
