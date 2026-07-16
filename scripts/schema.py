"""
Shared schema definition for the Telco Customer Churn dataset.

This module is the single source of truth for column names, types and
allowed categorical values. It is imported by:
- the data preparation scripts (split / inject errors)
- the model training script
- the FastAPI model service (Pydantic schemas)
- the Great Expectations suite definition (Airflow ingestion DAG)

Keeping this in one place avoids the schema drifting between the
different components of the project.
"""

# Primary key of the dataset. Kept in raw files but NOT used as a model
# feature (it carries no predictive signal).
ID_COLUMN = "customerID"

# Target column
TARGET_COLUMN = "Churn"
TARGET_POSITIVE_VALUE = "Yes"
TARGET_NEGATIVE_VALUE = "No"

# Numeric feature columns (continuous)
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]

# Binary 0/1 numeric column (kept separate because it is not scaled the
# same way as continuous features, but still numeric)
BINARY_NUMERIC_COLUMNS = ["SeniorCitizen"]

# Categorical feature columns, with their allowed values.
# These allowed-value lists are reused directly by the Great Expectations
# suite (expect_column_values_to_be_in_set).
CATEGORICAL_COLUMNS = {
    "gender": ["Female", "Male"],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["Yes", "No", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "OnlineBackup": ["Yes", "No", "No internet service"],
    "DeviceProtection": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
    "StreamingTV": ["Yes", "No", "No internet service"],
    "StreamingMovies": ["Yes", "No", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

# All feature columns used by the model (order matters for the sklearn
# ColumnTransformer, but pandas selects by name so this is mostly for
# documentation / iteration convenience).
FEATURE_COLUMNS = (
    NUMERIC_COLUMNS + BINARY_NUMERIC_COLUMNS + list(CATEGORICAL_COLUMNS.keys())
)

# All columns expected in a raw ingestion file (features + id + target).
# Note: the target column is NOT expected in inference/ingestion files,
# only in the original labelled dataset used for training.
ALL_RAW_COLUMNS = [ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]

# Valid ranges for numeric columns - used both for synthetic error
# injection and for Great Expectations range validation.
NUMERIC_RANGES = {
    "tenure": (0, 72),
    "MonthlyCharges": (0.0, 200.0),
    "TotalCharges": (0.0, 10000.0),
}
