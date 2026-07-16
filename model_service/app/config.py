import os
from pathlib import Path


class Settings:
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql+psycopg2://dsp_user:dsp_password@localhost:5432/dsp_db",
    )
    _default_model_path = (
        Path(__file__).resolve().parent.parent / "model_artifact" / "model.joblib"
    )
    MODEL_PATH: str = os.getenv("MODEL_PATH", str(_default_model_path))


settings = Settings()
