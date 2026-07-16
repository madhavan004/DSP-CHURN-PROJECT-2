"""
Train the initial churn-prediction model for Defense 1.

Defense 1 requirement: "API serving predictions (model loaded from file,
no MLflow yet)". This script trains a simple, fast scikit-learn pipeline
(< 1 hour of dev time as required by the project overview) and dumps it
to disk with joblib so the FastAPI service can load it at startup.

Usage:
    python scripts/train_initial_model.py \
        --input data/telco_full.csv \
        --output model_service/model_artifact/model.joblib
"""
import argparse
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.append(str(Path(__file__).resolve().parent))
from schema import (  # noqa: E402
    BINARY_NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    FEATURE_COLUMNS,
    NUMERIC_COLUMNS,
    TARGET_COLUMN,
    TARGET_POSITIVE_VALUE,
)


def load_clean_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # TotalCharges has 11 rows with blank strings (new customers, tenure=0).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    return df


def build_pipeline() -> Pipeline:
    categorical_cols = list(CATEGORICAL_COLUMNS.keys())
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLUMNS),
            ("bin", "passthrough", BINARY_NUMERIC_COLUMNS),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_cols,
            ),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/telco_full.csv")
    parser.add_argument(
        "--output", default="model_service/model_artifact/model.joblib"
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    df = load_clean_dataset(args.input)

    X = df[FEATURE_COLUMNS]
    y = (df[TARGET_COLUMN] == TARGET_POSITIVE_VALUE).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    print(f"Test accuracy: {acc:.4f}")
    print(f"Test F1-score: {f1:.4f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "feature_columns": FEATURE_COLUMNS,
            "model_version": "v1.0.0-initial",
            "metrics": {"accuracy": acc, "f1_score": f1},
        },
        output_path,
    )
    print(f"Model saved to {output_path}")


if __name__ == "__main__":
    main()
