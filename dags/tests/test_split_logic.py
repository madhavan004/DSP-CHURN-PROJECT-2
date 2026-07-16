import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "utils"))

from split_logic import split_dataframe  # noqa: E402


@pytest.fixture
def sample_df():
    return pd.DataFrame(
        {"customerID": ["a", "b", "c", "d"], "tenure": [1, 2, 3, 4]}
    )


def test_no_invalid_rows_returns_all_as_good(sample_df):
    good_df, bad_df = split_dataframe(sample_df, [])
    assert bad_df is None
    assert len(good_df) == 4


def test_all_invalid_rows_returns_all_as_bad(sample_df):
    good_df, bad_df = split_dataframe(sample_df, [0, 1, 2, 3])
    assert good_df is None
    assert len(bad_df) == 4


def test_partial_invalid_rows_splits_correctly(sample_df):
    good_df, bad_df = split_dataframe(sample_df, [1, 3])
    assert sorted(good_df["customerID"].tolist()) == ["a", "c"]
    assert sorted(bad_df["customerID"].tolist()) == ["b", "d"]


def test_split_is_exhaustive_and_non_overlapping(sample_df):
    good_df, bad_df = split_dataframe(sample_df, [2])
    assert len(good_df) + len(bad_df) == len(sample_df)
    assert set(good_df.index).isdisjoint(set(bad_df.index))
