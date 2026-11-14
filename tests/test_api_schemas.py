"""
Unit Tests for Pydantic API Schemas
Verifies input validation, field bounds, defaults, and error handling.
"""
import pytest
from pydantic import ValidationError
from model_service.app.schemas import CustomerFeatures, PredictRequest


def test_customer_features_valid():
    feat = CustomerFeatures(
        customerID="7590-VHVEG",
        gender="Female",
        SeniorCitizen=0,
        Partner="Yes",
        Dependents="No",
        tenure=1.0,
        PhoneService="No",
        MultipleLines="No phone service",
        InternetService="DSL",
        OnlineSecurity="No",
        OnlineBackup="Yes",
        DeviceProtection="No",
        TechSupport="No",
        StreamingTV="No",
        StreamingMovies="No",
        Contract="Month-to-month",
        PaperlessBilling="Yes",
        PaymentMethod="Electronic check",
        MonthlyCharges=29.85,
        TotalCharges=29.85,
    )
    assert feat.customerID == "7590-VHVEG"
    assert feat.SeniorCitizen == 0
    assert feat.tenure == 1.0


def test_customer_features_negative_tenure_rejected():
    with pytest.raises(ValidationError):
        CustomerFeatures(tenure=-5.0)


def test_predict_request_empty_features_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(features=[], source="webapp")


def test_predict_request_invalid_source_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(features=[CustomerFeatures()], source="invalid_source")  # only webapp or scheduled allowed
