"""
Generate 5 demo files for the defense presentation.

Files:
  1. demo_all_errors.csv   - Every row has at least one error (all 7 types)
  2. demo_clean_01.csv     - 10 clean rows, no errors
  3. demo_clean_02.csv     - 10 clean rows, no errors
  4. demo_clean_03.csv     - 10 clean rows, no errors
  5. demo_mixed.csv        - Mix of clean and error rows (~50%)

Output goes to data/demo_files/ (NOT raw_data, kept separate per spec).
"""
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from schema import (
    FEATURE_COLUMNS,
    ID_COLUMN,
)

DEMO_DIR = Path(__file__).resolve().parent.parent / "data" / "demo_files"
SOURCE_DATA = Path(__file__).resolve().parent.parent / "data" / "telco_full.csv"
ROWS_PER_FILE = 10


def load_source():
    """Load source CSV as object dtype so string assignment works."""
    df = pd.read_csv(SOURCE_DATA)
    for col in df.columns:
        df[col] = df[col].astype(object)
    return df


def generate_clean_batch(df_source: pd.DataFrame, batch_num: int) -> pd.DataFrame:
    """Sample 10 clean rows with unique customerIDs."""
    sample = df_source.sample(n=ROWS_PER_FILE, random_state=42 + batch_num).reset_index(drop=True)
    if "Churn" in sample.columns:
        sample = sample.drop(columns=["Churn"])
    for i in range(len(sample)):
        sample.at[i, ID_COLUMN] = f"{batch_num:02d}{i:02d}-CLEAN"
    return sample


def generate_all_errors_batch(df_source: pd.DataFrame) -> pd.DataFrame:
    """Generate a batch where every row has at least one error."""
    sample = df_source.sample(n=ROWS_PER_FILE, random_state=99).reset_index(drop=True)
    if "Churn" in sample.columns:
        sample = sample.drop(columns=["Churn"])

    df = sample.copy()

    for i in range(len(df)):
        df.at[i, ID_COLUMN] = f"ERR{i:02d}-BADID"

    # Row 0: completeness error (null in required column)
    df.at[0, "tenure"] = np.nan

    # Row 1: validity error (out-of-range numeric)
    df.at[1, "MonthlyCharges"] = "99999"

    # Row 2: consistency error (invalid categorical value)
    df.at[2, "gender"] = "INVALID_GENDER"

    # Row 3: type error (wrong dtype in numeric column)
    df.at[3, "tenure"] = "abc"

    # Row 4: cross-field inconsistency (PhoneService=No + MultipleLines=Yes)
    df.at[4, "PhoneService"] = "No"
    df.at[4, "MultipleLines"] = "Yes"

    # Row 5: format error (invalid customerID regex)
    df.at[5, ID_COLUMN] = "!!!BAD-FORMAT"

    # Row 6: statistical outlier
    df.at[6, "MonthlyCharges"] = "99999"
    df.at[6, "TotalCharges"] = "999999"

    # Row 7: duplicate primary key (same as row 0)
    df.at[7, ID_COLUMN] = df.at[0, ID_COLUMN]

    # Row 8: another completeness + validity combo
    df.at[8, "InternetService"] = np.nan
    df.at[8, "MonthlyCharges"] = "-50"

    # Row 9: type + consistency combo
    df.at[9, "TotalCharges"] = "$$$"
    df.at[9, "Contract"] = "BOGUS_CONTRACT"

    return df


def generate_mixed_batch(df_source: pd.DataFrame) -> pd.DataFrame:
    """Generate a batch with ~50% clean and ~50% error rows."""
    sample = df_source.sample(n=ROWS_PER_FILE, random_state=77).reset_index(drop=True)
    if "Churn" in sample.columns:
        sample = sample.drop(columns=["Churn"])

    df = sample.copy()

    for i in range(len(df)):
        df.at[i, ID_COLUMN] = f"MIX{i:02d}-MIXED"

    # Inject errors into first 5 rows only (50%)
    df.at[0, "tenure"] = np.nan                  # completeness
    df.at[1, "MonthlyCharges"] = "99999"          # validity (out of range)
    df.at[2, "gender"] = "UNKNOWN"                # consistency (invalid categorical)
    df.at[3, "tenure"] = "abc"                    # type (wrong dtype)
    df.at[4, "PhoneService"] = "No"
    df.at[4, "MultipleLines"] = "Yes"             # cross-field inconsistency

    # Rows 5-9 remain clean
    return df


def main():
    random.seed(42)
    np.random.seed(42)

    DEMO_DIR.mkdir(parents=True, exist_ok=True)

    df_source = load_source()
    print(f"Loaded source data: {len(df_source)} rows")

    # File 1: All errors
    all_errors = generate_all_errors_batch(df_source)
    all_errors.to_csv(DEMO_DIR / "demo_all_errors.csv", index=False)
    print(f"  Created demo_all_errors.csv ({len(all_errors)} rows)")

    # Files 2-4: Clean batches
    for i in range(1, 4):
        clean = generate_clean_batch(df_source, batch_num=i)
        clean.to_csv(DEMO_DIR / f"demo_clean_{i:02d}.csv", index=False)
        print(f"  Created demo_clean_{i:02d}.csv ({len(clean)} rows)")

    # File 5: Mixed batch
    mixed = generate_mixed_batch(df_source)
    mixed.to_csv(DEMO_DIR / "demo_mixed.csv", index=False)
    print(f"  Created demo_mixed.csv ({len(mixed)} rows)")

    print(f"\nAll 5 demo files written to {DEMO_DIR}")
    print("Copy them to data/raw_data/ before the defense demo.")


if __name__ == "__main__":
    main()

