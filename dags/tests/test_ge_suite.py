import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))

from ge_suite import compute_criticality, validate_dataframe  # noqa: E402

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
