import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from split_data import ROWS_PER_FILE, split_dataset  # noqa: E402


@pytest.fixture
def sample_csv(tmp_path):
    df = pd.DataFrame(
        {
            "customerID": [f"id-{i}" for i in range(50)],
            "tenure": range(50),
            "Churn": ["No"] * 25 + ["Yes"] * 25,
        }
    )
    path = tmp_path / "clean.csv"
    df.to_csv(path, index=False)
    return path


def test_each_generated_file_has_exactly_ten_rows(sample_csv, tmp_path):
    out_dir = tmp_path / "raw_data"
    written = split_dataset(str(sample_csv), str(out_dir), num_files=3)

    assert len(written) == 3
    for file_path in written:
        df = pd.read_csv(file_path)
        assert len(df) == ROWS_PER_FILE


def test_split_drops_target_column_by_default(sample_csv, tmp_path):
    out_dir = tmp_path / "raw_data"
    written = split_dataset(str(sample_csv), str(out_dir), num_files=2)

    for file_path in written:
        df = pd.read_csv(file_path)
        assert "Churn" not in df.columns


def test_split_can_oversample_when_more_files_requested_than_rows(
    sample_csv, tmp_path
):
    out_dir = tmp_path / "raw_data"
    # 50 rows available, ask for 10 files * 10 rows = 100 rows -> must sample with replacement
    written = split_dataset(str(sample_csv), str(out_dir), num_files=10)
    assert len(written) == 10
    for file_path in written:
        assert len(pd.read_csv(file_path)) == ROWS_PER_FILE
