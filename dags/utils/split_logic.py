"""
Pure (Airflow-free) helper functions for splitting an ingested batch into
good/bad rows. Kept separate from ingestion_dag.py so it can be unit
tested with plain pytest, without needing Apache Airflow installed.
"""
from typing import Optional, Tuple

import pandas as pd


def split_dataframe(
    df: pd.DataFrame, invalid_row_indices
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Split a dataframe into (good_df, bad_df) based on a set/list of
    invalid row indices.

    Returns:
        (good_df, None)  if there are no invalid rows at all
        (None, bad_df)   if every row is invalid
        (good_df, bad_df) otherwise
    """
    invalid_idx = set(invalid_row_indices)

    if not invalid_idx:
        return df, None

    if len(invalid_idx) >= len(df):
        return None, df

    good_df = df.drop(index=list(invalid_idx))
    bad_df = df.loc[sorted(invalid_idx)]
    return good_df, bad_df
