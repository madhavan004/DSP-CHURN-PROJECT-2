"""
Pydantic Schemas for Request and Response Validation
"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    customerID: Optional[str] = Field(
        default="7590-VHVEG", description="Unique Customer Identifier"
    )
    gender: str = Field(default="Female", description="Gender (Female / Male)")
    SeniorCitizen: int = Field(default=0, ge=0, le=1, description="Senior Citizen (0 or 1)")
    Partner: str = Field(default="Yes", description="Partner (Yes / No)")
    Dependents: str = Field(default="No", description="Dependents (Yes / No)")
    tenure: float = Field(default=1.0, ge=0.0, description="Months with telecom provider")
    PhoneService: str = Field(default="No", description="Phone Service (Yes / No)")
    MultipleLines: str = Field(default="No phone service", description="Multiple Lines")
    InternetService: str = Field(
        default="DSL", description="Internet Service (DSL, Fiber optic, No)"
    )
    OnlineSecurity: str = Field(default="No", description="Online Security")
    OnlineBackup: str = Field(default="Yes", description="Online Backup")
    DeviceProtection: str = Field(default="No", description="Device Protection")
    TechSupport: str = Field(default="No", description="Tech Support")
    StreamingTV: str = Field(default="No", description="Streaming TV")
    StreamingMovies: str = Field(default="No", description="Streaming Movies")
    Contract: str = Field(default="Month-to-month", description="Contract term")
    PaperlessBilling: str = Field(default="Yes", description="Paperless Billing (Yes / No)")
    PaymentMethod: str = Field(default="Electronic check", description="Payment Method")
    MonthlyCharges: float = Field(default=29.85, ge=0.0, description="Monthly Charges ($)")
    TotalCharges: float = Field(default=29.85, ge=0.0, description="Total Charges ($)")

    model_config = {
        "json_schema_extra": {
            "example": {
                "customerID": "7590-VHVEG",
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "tenure": 1.0,
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
        }
    }


class PredictRequest(BaseModel):
    features: List[CustomerFeatures] = Field(
        ..., min_length=1, description="List of feature records to score"
    )
    source: Literal["webapp", "scheduled"] = Field(
        default="webapp", description="Origin source of prediction request"
    )


class SinglePredictionResult(BaseModel):
    customer_id: Optional[str]
    prediction: str = Field(description="'Yes' (churn) or 'No' (retention)")
    churn_probability: float = Field(description="Probability between 0.0 and 1.0")
    risk_level: str = Field(description="Low (< 0.35), Medium (0.35 - 0.65), High (> 0.65)")
    model_version: str
    features: Dict[str, Any]


class BatchPredictResponse(BaseModel):
    total: int
    model_version: str
    predictions: List[SinglePredictionResult]


class PastPredictionItem(BaseModel):
    id: int
    customer_id: Optional[str]
    source: str
    prediction: str
    churn_probability: float
    model_version: str
    tenure: Optional[float]
    monthly_charges: Optional[float]
    total_charges: Optional[float]
    contract: Optional[str]
    internet_service: Optional[str]
    features_payload: Dict[str, Any]
    created_at: datetime

    model_config = {"from_attributes": True}


class PastPredictionsResponse(BaseModel):
    total: int
    items: List[PastPredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_source: str
    timestamp: datetime


class ReloadModelResponse(BaseModel):
    status: str
    message: str
    model_version: str
    model_source: str
