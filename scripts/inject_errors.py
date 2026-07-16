"""
Inject configurable, random data-quality errors into clean CSV files.

Implements 7 error types (5 required + 2 extra), each with an
independently configurable probability:

Required:
  1. completeness_null        - Null in a required column          (row)
  2. validity_out_of_range    - Numeric value outside valid range   (row)
  3. consistency_invalid_cat  - Invalid categorical value           (row)
  4. schema_missing_column    - A required column is dropped        (file)
  5. type_wrong_dtype         - Wrong data type in a numeric column (row)

Extra (>= 2 required, we implement 2):
  6. duplicate_primary_key    - Duplicate customerID within a file  (row)
  7. cross_field_inconsistency- e.g. PhoneService=No but MultipleLines=Yes
                                 or tenure=0 but TotalCharges > 0    (row)

Usage:
    # single file
    python scripts/inject_errors.py --input data/raw_data/batch_0001.csv \
        --output data/raw_data_corrupted/batch_0001.csv

    # whole directory (in place)
    python scripts/inject_errors.py --input-dir data/raw_data \
        --output-dir data/raw_data_corrupted
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
from schema import CATEGORICAL_COLUMNS, ID_COLUMN, NUMERIC_COLUMNS  # noqa: E402

# Default probability that a given row-level error is applied to any
# single row. File-level errors use their own probability, applied once
# per file.
DEFAULT_ERROR_CONFIG = {
    "completeness_null": 0.05,
    "validity_out_of_range": 0.05,
    "consistency_invalid_cat": 0.05,
    "schema_missing_column": 0.03,  # file-level
    "type_wrong_dtype": 0.05,
    "duplicate_primary_key": 0.03,
    "cross_field_inconsistency": 0.05,
}

REQUIRED_COLUMNS_FOR_NULL = [ID_COLUMN, "tenure", "MonthlyCharges", "InternetService"]


def inject_completeness_null(df: pd.DataFrame, prob: float) -> pd.DataFrame:
    for idx in df.index:
        if random.random() < prob:
            col = random.choice(REQUIRED_COLUMNS_FOR_NULL)
            df.at[idx, col] = np.nan
    return df


def inject_validity_out_of_range(df: pd.DataFrame, prob: float) -> pd.DataFrame:
    out_of_range_values = {
        "tenure": [-5, -1, 150, 999],
        "MonthlyCharges": [-20.0, -1.0, 5000.0],
        "TotalCharges": [-100.0, 999999.0],
    }
    for idx in df.index:
        if random.random() < prob:
            col = random.choice(NUMERIC_COLUMNS)
            df.at[idx, col] = random.choice(out_of_range_values[col])
    return df


def inject_consistency_invalid_cat(df: pd.DataFrame, prob: float) -> pd.DataFrame:
    invalid_values = ["UNKNOWN", "N/A", "???", "invalid_value"]
    for idx in df.index:
        if random.random() < prob:
            col = random.choice(list(CATEGORICAL_COLUMNS.keys()))
            df.at[idx, col] = random.choice(invalid_values)
    return df


def inject_schema_missing_column(df: pd.DataFrame, prob: float) -> pd.DataFrame:
    if random.random() < prob:
        droppable = [c for c in df.columns if c != ID_COLUMN]
        col = random.choice(droppable)
        df = df.drop(columns=[col])
    return df


def inject_type_wrong_dtype(df: pd.DataFrame, prob: float) -> pd.DataFrame:
    bogus_strings = ["abc", "N/A", "twelve", "$$$"]
    # Cast target numeric columns to object dtype so we can safely inject
    # non-numeric strings without pandas raising a dtype-cast error.
    cols_to_cast = [c for c in NUMERIC_COLUMNS if c in df.columns]
    for col in cols_to_cast:
        df[col] = df[col].astype(object)
    for idx in df.index:
        if random.random() < prob:
            col = random.choice(cols_to_cast)
            df.at[idx, col] = random.choice(bogus_strings)
    return df


def inject_duplicate_primary_key(df: pd.DataFrame, prob: float) -> pd.DataFrame:
    if random.random() < prob and len(df) > 1:
        src_idx = random.choice(df.index)
        dst_idx = random.choice([i for i in df.index if i != src_idx])
        df.at[dst_idx, ID_COLUMN] = df.at[src_idx, ID_COLUMN]
    return df


def inject_cross_field_inconsistency(df: pd.DataFrame, prob: float) -> pd.DataFrame:
    for idx in df.index:
        if random.random() < prob:
            choice = random.choice(["phone_lines", "tenure_charges"])
            if choice == "phone_lines" and "PhoneService" in df.columns:
                df.at[idx, "PhoneService"] = "No"
                df.at[idx, "MultipleLines"] = "Yes"  # impossible combination
            elif "tenure" in df.columns and "TotalCharges" in df.columns:
                df.at[idx, "tenure"] = 0
                df.at[idx, "TotalCharges"] = round(random.uniform(500, 3000), 2)
    return df


ERROR_FUNCTIONS = {
    "completeness_null": inject_completeness_null,
    "validity_out_of_range": inject_validity_out_of_range,
    "consistency_invalid_cat": inject_consistency_invalid_cat,
    "schema_missing_column": inject_schema_missing_column,
    "type_wrong_dtype": inject_type_wrong_dtype,
    "duplicate_primary_key": inject_duplicate_primary_key,
    "cross_field_inconsistency": inject_cross_field_inconsistency,
}


def corrupt_dataframe(df: pd.DataFrame, error_config: dict) -> pd.DataFrame:
    df = df.copy()
    for error_name, func in ERROR_FUNCTIONS.items():
        prob = error_config.get(error_name, 0.0)
        if prob > 0:
            df = func(df, prob)
    return df


def process_file(input_path: Path, output_path: Path, error_config: dict):
    df = pd.read_csv(input_path)
    corrupted = corrupt_dataframe(df, error_config)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    corrupted.to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Inject configurable random data-quality errors into CSV files."
    )
    parser.add_argument("--input", help="Path to a single clean CSV file")
    parser.add_argument("--output", help="Path to write the corrupted CSV file")
    parser.add_argument("--input-dir", help="Directory of clean CSV files")
    parser.add_argument(
        "--output-dir", help="Directory to write corrupted CSV files"
    )
    parser.add_argument("--seed", type=int, default=None)
    # Allow overriding individual error probabilities from the CLI
    for name, default in DEFAULT_ERROR_CONFIG.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}", type=float, default=default
        )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    error_config = {
        name: getattr(args, name) for name in DEFAULT_ERROR_CONFIG.keys()
    }

    if args.input:
        out = Path(args.output) if args.output else Path(args.input)
        process_file(Path(args.input), out, error_config)
        print(f"Corrupted file written to {out}")
    elif args.input_dir:
        in_dir = Path(args.input_dir)
        out_dir = Path(args.output_dir) if args.output_dir else in_dir
        files = sorted(in_dir.glob("*.csv"))
        for f in files:
            process_file(f, out_dir / f.name, error_config)
        print(f"Corrupted {len(files)} files into {out_dir}")
    else:
        raise SystemExit("Provide either --input or --input-dir")


if __name__ == "__main__":
    main()
