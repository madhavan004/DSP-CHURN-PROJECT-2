import os
import sys
from pathlib import Path

# Use a throwaway sqlite DB for tests instead of PostgreSQL, and point at
# the trained model artifact. Must be set BEFORE importing the app.
os.environ.setdefault("DATABASE_URL", "sqlite:///./test_dsp.db")
os.environ.setdefault(
    "MODEL_PATH",
    str(Path(__file__).resolve().parent.parent / "model_artifact" / "model.joblib"),
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def cleanup_test_db():
    yield
    db_path = Path("test_dsp.db")
    if db_path.exists():
        db_path.unlink()
