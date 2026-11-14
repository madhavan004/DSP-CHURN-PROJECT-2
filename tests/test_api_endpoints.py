"""
Unit Tests for FastAPI Model Service Endpoints
Uses TestClient and SQLite in-memory database for fast, isolated CI tests.
"""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from model_service.app.database import Base, get_db
from model_service.app.main import app
from model_service.app.ml_model import model_manager

# Setup in-memory SQLite test database
SQLALCHEMY_DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    Base.metadata.create_all(bind=engine)
    # Ensure model is loaded for tests
    model_manager.load_model()
    yield
    Base.metadata.drop_all(bind=engine)


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "model_version" in data


def test_predict_single_sample():
    sample_feature = {
        "customerID": "1234-TESTA",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2.0,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 80.5,
        "TotalCharges": 161.0,
    }

    payload = {"features": [sample_feature], "source": "webapp"}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total"] == 1
    pred = data["predictions"][0]
    assert pred["customer_id"] == "1234-TESTA"
    assert pred["prediction"] in ["Yes", "No"]
    assert 0.0 <= pred["churn_probability"] <= 1.0
    assert pred["risk_level"] in ["Low", "Medium", "High"]


def test_past_predictions_retrieval():
    # Insert prediction first
    test_predict_single_sample()

    response = client.get("/past-predictions?source=all&limit=10")
    assert response.status_code == 200
    data = response.json()
    assert data["total"] >= 1
    assert len(data["items"]) >= 1
    assert data["items"][0]["source"] == "webapp"


def test_predict_invalid_payload_rejected():
    # Missing required field or empty list
    response = client.post("/predict", json={"features": [], "source": "webapp"})
    assert response.status_code == 422
