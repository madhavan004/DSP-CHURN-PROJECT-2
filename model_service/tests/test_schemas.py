import pytest
from pydantic import ValidationError

from app.schemas import CustomerFeatures, PredictionRequest

VALID_FEATURES = {
    "customerID": "7590-VHVEG",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
}


def test_valid_customer_features():
    features = CustomerFeatures(**VALID_FEATURES)
    assert features.gender == "Female"
    assert features.tenure == 1


def test_invalid_categorical_value_raises():
    bad = dict(VALID_FEATURES, gender="Alien")
    with pytest.raises(ValidationError):
        CustomerFeatures(**bad)


def test_negative_tenure_raises():
    bad = dict(VALID_FEATURES, tenure=-5)
    with pytest.raises(ValidationError):
        CustomerFeatures(**bad)


def test_tenure_above_max_raises():
    bad = dict(VALID_FEATURES, tenure=500)
    with pytest.raises(ValidationError):
        CustomerFeatures(**bad)


def test_invalid_senior_citizen_raises():
    bad = dict(VALID_FEATURES, SeniorCitizen=5)
    with pytest.raises(ValidationError):
        CustomerFeatures(**bad)


def test_prediction_request_defaults_to_webapp_source():
    request = PredictionRequest(features=[VALID_FEATURES])
    assert request.source.value == "webapp"


def test_prediction_request_requires_at_least_one_feature_row():
    with pytest.raises(ValidationError):
        PredictionRequest(features=[])
