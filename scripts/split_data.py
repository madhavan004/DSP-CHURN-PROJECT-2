"""
Split a dataset into N files of exactly 10 rows each, to feed the
ingestion job's `raw_data` folder.

Usage:
    python scripts/split_data.py \
        --input data/telco_full.csv \
        --output-dir data/raw_data \
        --num-files 30

Note: the CLI argument is the NUMBER OF FILES to generate, not the number
of rows per file (each file always has exactly 10 rows, as required).
The target column (Churn) is dropped since ingestion files simulate new,
unlabeled inference data.
"""
import argparse
from pathlib import Path

import pandas as pd

ROWS_PER_FILE = 10


def split_dataset(
    input_path: str, output_dir: str, num_files: int, drop_target: bool = True
) -> list:
    df = pd.read_csv(input_path)

    if drop_target and "Churn" in df.columns:
        df = df.drop(columns=["Churn"])

    rows_needed = num_files * ROWS_PER_FILE
    if rows_needed > len(df):
        # Sample with replacement if we need more rows than the source
        # dataset has (keeps the demo generator flexible).
        df = df.sample(n=rows_needed, replace=True, random_state=None).reset_index(
            drop=True
        )
    else:
        df = df.sample(n=rows_needed, random_state=None).reset_index(drop=True)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for i in range(num_files):
        start = i * ROWS_PER_FILE
        chunk = df.iloc[start:start + ROWS_PER_FILE]
        file_path = out_dir / f"batch_{i + 1:04d}.csv"
        chunk.to_csv(file_path, index=False)
        written.append(str(file_path))

    return written


def main():
    parser = argparse.ArgumentParser(
        description="Split a dataset into N files of 10 rows each."
    )
    parser.add_argument("--input", default="data/telco_full.csv")
    parser.add_argument("--output-dir", default="data/raw_data")
    parser.add_argument(
        "--num-files",
        type=int,
        required=True,
        help="Number of files to generate (each will contain 10 rows).",
    )
    parser.add_argument(
        "--keep-target",
        action="store_true",
        help="Keep the Churn column in the output files (off by default).",
    )
    args = parser.parse_args()

    if args.num_files < 1:
        raise SystemExit("--num-files must be >= 1")

    written = split_dataset(
        args.input, args.output_dir, args.num_files, drop_target=not args.keep_target
    )
    print(f"Wrote {len(written)} files of {ROWS_PER_FILE} rows to {args.output_dir}")
    for f in written[:5]:
        print(f"  {f}")
    if len(written) > 5:
        print(f"  ... and {len(written) - 5} more")


if __name__ == "__main__":
    main()
