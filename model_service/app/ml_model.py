import logging
from pathlib import Path
from typing import List

import joblib
import pandas as pd

from app.config import settings
from app.schemas import CustomerFeatures

logger = logging.getLogger("model_service")

FEATURE_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


class ChurnModel:
    """
    Thin wrapper around the scikit-learn pipeline artifact.

    Defense 1: the model is loaded once from a local file at API startup
    (see app/main.py `lifespan`). Defense 2 will add a `reload()` call
    triggered by the `/reload-model` endpoint to swap in a new champion
    model from the MLflow registry without restarting the container.
    """

    def __init__(self):
        self.pipeline = None
        self.model_version = None

    def load(self, model_path: str = None):
        path = Path(model_path or settings.MODEL_PATH)
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {path}. "
                "Run scripts/train_initial_model.py first."
            )
        artifact = joblib.load(path)
        self.pipeline = artifact["pipeline"]
        self.model_version = artifact["model_version"]
        logger.info(
            "Loaded model version=%s from %s", self.model_version, path
        )

    @property
    def is_ready(self) -> bool:
        return self.pipeline is not None

    def predict(self, features: List[CustomerFeatures]):
        if not self.is_ready:
            raise RuntimeError("Model is not loaded")

        rows = [f.model_dump(exclude={"customerID"}) for f in features]
        df = pd.DataFrame(rows)[FEATURE_COLUMNS]

        probabilities = self.pipeline.predict_proba(df)[:, 1]  # P(Churn=Yes)
        predictions = (probabilities >= 0.5).astype(int)

        results = []
        for pred, proba in zip(predictions, probabilities):
            results.append(
                {
                    "prediction": "Yes" if pred == 1 else "No",
                    "churn_probability": round(float(proba), 4),
                }
            )
        return results


# Singleton instance shared across the app
churn_model = ChurnModel()
