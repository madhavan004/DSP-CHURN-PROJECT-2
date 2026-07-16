from fastapi.testclient import TestClient

from app.main import app

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


def test_health_check_returns_200():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_predict_single_row():
    with TestClient(app) as client:
        response = client.post(
            "/predict", json={"features": [VALID_FEATURES], "source": "webapp"}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == 1
        assert body["predictions"][0]["prediction"] in ("Yes", "No")
        assert 0.0 <= body["predictions"][0]["churn_probability"] <= 1.0


def test_predict_batch_rows():
    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={
                "features": [VALID_FEATURES, VALID_FEATURES],
                "source": "scheduled",
            },
        )
        assert response.status_code == 200
        assert response.json()["count"] == 2


def test_predict_rejects_empty_features():
    with TestClient(app) as client:
        response = client.post("/predict", json={"features": []})
        assert response.status_code == 422  # pydantic min_length validation


def test_predict_rejects_invalid_categorical():
    with TestClient(app) as client:
        bad_features = dict(VALID_FEATURES, InternetService="Satellite")
        response = client.post("/predict", json={"features": [bad_features]})
        assert response.status_code == 422


def test_past_predictions_after_predict():
    with TestClient(app) as client:
        client.post("/predict", json={"features": [VALID_FEATURES]})
        response = client.get("/past-predictions?source=webapp&limit=10")
        assert response.status_code == 200
        body = response.json()
        assert body["count"] >= 1
        assert all(p["source"] == "webapp" for p in body["predictions"])


def test_past_predictions_invalid_source_returns_400():
    with TestClient(app) as client:
        response = client.get("/past-predictions?source=invalid")
        assert response.status_code == 400
