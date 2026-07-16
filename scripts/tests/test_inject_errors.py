import random
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inject_errors import (  # noqa: E402
    DEFAULT_ERROR_CONFIG,
    corrupt_dataframe,
    inject_completeness_null,
    inject_duplicate_primary_key,
    inject_schema_missing_column,
)
from schema import ID_COLUMN  # noqa: E402


@pytest.fixture
def clean_df():
    return pd.DataFrame(
        {
            "customerID": [f"1234-ABCD{i}" for i in range(20)],
            "gender": ["Female"] * 20,
            "SeniorCitizen": [0] * 20,
            "Partner": ["Yes"] * 20,
            "Dependents": ["No"] * 20,
            "tenure": [12] * 20,
            "PhoneService": ["Yes"] * 20,
            "MultipleLines": ["No"] * 20,
            "InternetService": ["DSL"] * 20,
            "OnlineSecurity": ["No"] * 20,
            "OnlineBackup": ["Yes"] * 20,
            "DeviceProtection": ["No"] * 20,
            "TechSupport": ["No"] * 20,
            "StreamingTV": ["No"] * 20,
            "StreamingMovies": ["No"] * 20,
            "Contract": ["Month-to-month"] * 20,
            "PaperlessBilling": ["Yes"] * 20,
            "PaymentMethod": ["Electronic check"] * 20,
            "MonthlyCharges": [50.0] * 20,
            "TotalCharges": [600.0] * 20,
        }
    )


def test_zero_probability_leaves_dataframe_unchanged(clean_df):
    config = {name: 0.0 for name in DEFAULT_ERROR_CONFIG}
    result = corrupt_dataframe(clean_df, config)
    pd.testing.assert_frame_equal(result, clean_df)


def test_completeness_null_introduces_nulls(clean_df):
    random.seed(42)
    result = inject_completeness_null(clean_df.copy(), prob=1.0)
    assert result.isna().any().any()


def test_duplicate_primary_key_creates_a_repeated_id(clean_df):
    random.seed(1)
    result = inject_duplicate_primary_key(clean_df.copy(), prob=1.0)
    assert result[ID_COLUMN].duplicated().any()


def test_schema_missing_column_drops_a_column(clean_df):
    random.seed(2)
    result = inject_schema_missing_column(clean_df.copy(), prob=1.0)
    assert len(result.columns) == len(clean_df.columns) - 1
    assert ID_COLUMN in result.columns  # never drop the primary key


def test_corrupt_dataframe_preserves_row_count(clean_df):
    random.seed(3)
    result = corrupt_dataframe(clean_df, DEFAULT_ERROR_CONFIG)
    # Row-level corruption never adds/removes rows (only schema errors
    # remove a column, not rows).
    assert len(result) == len(clean_df)
