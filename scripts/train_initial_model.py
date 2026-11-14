"""
Baseline Model Training Script for Telecom Customer Churn
Builds an end-to-end scikit-learn Pipeline (Preprocessing + Classification)
Evaluates metrics (Accuracy, F1, ROC-AUC, Latency), calculates baseline feature statistics
for Drift Monitoring, and saves the trained pipeline artifact to model_service/model/model.joblib.
"""
import argparse
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


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
TARGET_COL = "Churn"


def compute_feature_statistics(df: pd.DataFrame) -> dict:
    """Computes baseline statistics for numerical and categorical features for drift monitoring."""
    stats = {"numerical": {}, "categorical": {}, "sample_count": len(df)}

    for col in NUMERICAL_FEATURES:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            stats["numerical"][col] = {
                "mean": float(s.mean()),
                "std": float(s.std()),
                "min": float(s.min()),
                "p25": float(s.quantile(0.25)),
                "median": float(s.median()),
                "p75": float(s.quantile(0.75)),
                "max": float(s.max()),
            }

    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            s = df[col].astype(str)
            value_counts = s.value_counts(normalize=True).to_dict()
            stats["categorical"][col] = {str(k): float(v) for k, v in value_counts.items()}

    return stats


def train_model(
    data_path: str = "data/telco_churn_clean.csv",
    output_model_path: str = "model_service/model/model.joblib",
    model_type: str = "rf",
    seed: int = 42,
) -> dict:
    """Trains the Churn prediction model and saves artifact."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path not found: {data_path}")

    df = pd.read_csv(data_path)
    # Ensure numericals are float
    for col in NUMERICAL_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].astype(str)

    y = (df[TARGET_COL].astype(str).str.strip().str.lower() == "yes").astype(int)
    X = df[ALL_FEATURES]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Compute baseline feature statistics on training data
    feature_stats = compute_feature_statistics(X_train)

    # Build preprocessing pipeline
    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, NUMERICAL_FEATURES),
            ("cat", cat_transformer, CATEGORICAL_FEATURES),
        ]
    )

    if model_type == "gb":
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=seed)
    else:
        clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=seed, n_jobs=1)

    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", clf)])

    # Fit
    start_train = time.time()
    pipeline.fit(X_train, y_train)
    train_duration = time.time() - start_train

    # Inference & Metrics
    start_inf = time.time()
    y_pred = pipeline.predict(X_test)
    inf_duration_ms = ((time.time() - start_inf) / len(X_test)) * 1000.0

    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "avg_inference_latency_ms": float(inf_duration_ms),
        "train_duration_s": float(train_duration),
        "model_version": "v1.0.0",
        "model_type": model_type,
    }

    # Save pipeline artifact
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    joblib.dump(pipeline, output_model_path)

    # Save baseline stats alongside model
    stats_path = os.path.join(os.path.dirname(output_model_path), "baseline_feature_stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(feature_stats, f, indent=2)

    # Save metrics summary
    metrics_path = os.path.join(os.path.dirname(output_model_path), "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[train_model] Successfully trained {model_type} model.")
    print(f"  - Accuracy: {metrics['accuracy']:.4f}")
    print(f"  - F1 Score: {metrics['f1']:.4f}")
    print(f"  - ROC-AUC:  {metrics['roc_auc']:.4f}")
    print(f"  - Latency:  {metrics['avg_inference_latency_ms']:.3f} ms/sample")
    print(f"  - Saved to: {output_model_path}")
    print(f"  - Baseline stats saved to: {stats_path}")

    return {
        "metrics": metrics,
        "feature_stats": feature_stats,
        "model_path": output_model_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Train initial Churn prediction model.")
    parser.add_argument("--data", type=str, default="data/telco_churn_clean.csv", help="Input dataset path")
    parser.add_argument("--output", type=str, default="model_service/model/model.joblib", help="Output model path")
    parser.add_argument("--model_type", type=str, default="rf", choices=["rf", "gb"], help="Model type: rf or gb")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    train_model(args.data, args.output, args.model_type, args.seed)


if __name__ == "__main__":
    main()
