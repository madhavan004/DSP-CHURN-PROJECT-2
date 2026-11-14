"""
ML Model Service Wrapper
Handles:
1. Loading scikit-learn pipeline from local file or MLflow Model Registry (@champion)
2. Safe batch & single inference
3. Dynamic runtime model reloading via /reload-model
"""
import json
import logging
import os
from typing import Any, Dict, List, Tuple
import joblib
import pandas as pd

logger = logging.getLogger("model_service.ml_model")

# Feature order expected by preprocessing pipeline
CATEGORICAL_FEATURES = [
    "gender",
    "SeniorCitizen",
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
NUMERICAL_FEATURES = ["tenure", "MonthlyCharges", "TotalCharges"]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES


class ModelManager:
    def __init__(self):
        self.model = None
        self.model_version = "v1.0.0"
        self.model_source = "local_artifact"
        self.local_model_path = os.getenv("MODEL_PATH", "model_service/model/model.joblib")
        self.mlflow_model_uri = os.getenv(
            "MLFLOW_MODEL_URI", "models:/TelecomChurnClassifier@champion"
        )

    def load_model(self) -> bool:
        """Attempts to load model from MLflow registry first, then falls back to local artifact."""
        # 1. Try MLflow if configured
        mlflow_tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_tracking_uri:
            try:
                import mlflow.pyfunc
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                logger.info(f"Loading champion model from MLflow: {self.mlflow_model_uri}")
                self.model = mlflow.pyfunc.load_model(self.mlflow_model_uri)
                self.model_source = "mlflow_registry"
                self.model_version = "champion_mlflow"
                logger.info("Successfully loaded champion model from MLflow registry.")
                return True
            except Exception as e:
                logger.warning(f"MLflow model load failed ({e}), falling back to local file.")

        # 2. Try Local File
        if os.path.exists(self.local_model_path):
            try:
                logger.info(f"Loading model from local path: {self.local_model_path}")
                self.model = joblib.load(self.local_model_path)
                self.model_source = "local_artifact"
                # Check if metrics.json has version
                metrics_path = os.path.join(os.path.dirname(self.local_model_path), "metrics.json")
                if os.path.exists(metrics_path):
                    with open(metrics_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.model_version = data.get("model_version", "v1.0.0")
                logger.info(f"Loaded local model successfully. Version: {self.model_version}")
                return True
            except Exception as e:
                logger.error(f"Error loading local model from {self.local_model_path}: {e}")

        logger.error("No model could be loaded! Service is running in uninitialized model state.")
        return False

    def reload_model(self) -> Tuple[bool, str]:
        """Dynamically reloads model at runtime."""
        success = self.load_model()
        if success:
            msg = (
                f"Model successfully reloaded. "
                f"Source: {self.model_source}, Version: {self.model_version}"
            )
            logger.info(msg)
            return True, msg
        else:
            msg = "Failed to reload model from MLflow or local file."
            logger.error(msg)
            return False, msg

    def predict(self, feature_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Runs batch inference on a list of feature dictionaries."""
        if self.model is None:
            raise RuntimeError("Model is not loaded.")

        df = pd.DataFrame(feature_dicts)

        # Ensure all expected columns exist
        for col in NUMERICAL_FEATURES:
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

        for col in CATEGORICAL_FEATURES:
            if col not in df.columns:
                df[col] = "missing"
            else:
                df[col] = df[col].astype(str)

        X = df[ALL_FEATURES]

        # Handle both scikit-learn Pipeline and MLflow PyFunc
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)[:, 1]
            raw_preds = self.model.predict(X)
        else:
            # Pyfunc fallback
            preds = self.model.predict(X)
            if hasattr(preds, "values"):
                preds = preds.values
            probs = preds if preds.ndim == 1 else preds[:, 1]
            raw_preds = (probs >= 0.5).astype(int)

        results = []
        for i, row in enumerate(feature_dicts):
            p_val = float(probs[i])
            pred_label = "Yes" if (raw_preds[i] == 1 or p_val >= 0.5) else "No"

            if p_val >= 0.65:
                risk = "High"
            elif p_val >= 0.35:
                risk = "Medium"
            else:
                risk = "Low"

            results.append({
                "customer_id": row.get("customerID"),
                "prediction": pred_label,
                "churn_probability": round(p_val, 4),
                "risk_level": risk,
                "model_version": self.model_version,
                "features": row,
            })

        return results


# Global singleton instance
model_manager = ModelManager()
