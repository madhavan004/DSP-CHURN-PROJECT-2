"""
FastAPI Model Service for Telecom Customer Churn Prediction
Endpoints:
- GET /health
- POST /predict
- GET /past-predictions
- POST /reload-model
"""
from contextlib import asynccontextmanager
from datetime import datetime, timezone
import logging
from typing import Optional
from fastapi import FastAPI, Depends, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import desc, text
from sqlalchemy.orm import Session

from .database import get_db, init_db
from .models import PredictionRecord
from .schemas import (
    BatchPredictResponse,
    HealthResponse,
    PastPredictionItem,
    PastPredictionsResponse,
    PredictRequest,
    ReloadModelResponse,
    SinglePredictionResult,
)
from .ml_model import model_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("model_service.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    logger.info("Initializing database tables...")
    try:
        init_db()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.warning(f"Database initialization deferred or failed: {e}")

    logger.info("Loading ML Model at startup...")
    model_loaded = model_manager.load_model()
    if not model_loaded:
        logger.warning("Startup model load could not find artifact. Will retry on first request.")

    yield
    logger.info("Shutting down model service.")


app = FastAPI(
    title="Telecom Churn Prediction Service",
    description=(
        "Production API for Telecom Customer Churn prediction, "
        "batch scoring, and past inference logs."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check(db: Session = Depends(get_db)):
    """Health check: container readiness, database connectivity, and model state."""
    db_status = "ok"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"error: {str(e)}"
        logger.error(f"Health check DB probe error: {e}")

    if model_manager.model is None:
        # Attempt lazy load
        model_manager.load_model()

    return HealthResponse(
        status="healthy" if db_status == "ok" else "degraded",
        model_version=model_manager.model_version,
        model_source=model_manager.model_source,
        timestamp=datetime.now(timezone.utc),
    )


@app.post(
    "/predict", response_model=BatchPredictResponse,
    status_code=status.HTTP_200_OK, tags=["Prediction"]
)
def predict(request: PredictRequest, db: Session = Depends(get_db)):
    """
    Make predictions for a single sample or batch of samples.
    Validates input features with Pydantic, performs inference,
    and automatically stores predictions & features in PostgreSQL.
    """
    if model_manager.model is None:
        loaded = model_manager.load_model()
        if not loaded:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model artifact is not loaded. Train or provide a valid model artifact.",
            )

    feature_dicts = [f.model_dump() for f in request.features]

    try:
        results = model_manager.predict(feature_dicts)
    except Exception as e:
        logger.error(f"Inference error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing prediction model: {str(e)}",
        )

    # Persist predictions to Database
    records = []
    for r in results:
        feat = r["features"]
        rec = PredictionRecord(
            customer_id=r["customer_id"],
            source=request.source,
            prediction=r["prediction"],
            churn_probability=r["churn_probability"],
            model_version=r["model_version"],
            tenure=float(feat.get("tenure", 0) or 0),
            monthly_charges=float(feat.get("MonthlyCharges", 0) or 0),
            total_charges=float(feat.get("TotalCharges", 0) or 0),
            contract=str(feat.get("Contract", "")),
            internet_service=str(feat.get("InternetService", "")),
            payment_method=str(feat.get("PaymentMethod", "")),
            features_payload=feat,
            created_at=datetime.now(timezone.utc),
        )
        records.append(rec)

    try:
        db.add_all(records)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to persist prediction records to database: {e}")
        # Note: we still return predictions to the client, but log error

    output_preds = [
        SinglePredictionResult(
            customer_id=r["customer_id"],
            prediction=r["prediction"],
            churn_probability=r["churn_probability"],
            risk_level=r["risk_level"],
            model_version=r["model_version"],
            features=r["features"],
        )
        for r in results
    ]

    return BatchPredictResponse(
        total=len(output_preds),
        model_version=model_manager.model_version,
        predictions=output_preds,
    )


@app.get("/past-predictions", response_model=PastPredictionsResponse, tags=["Predictions History"])
def get_past_predictions(
    start_date: Optional[datetime] = Query(
        None, description="Filter records created on or after this UTC timestamp"
    ),
    end_date: Optional[datetime] = Query(
        None, description="Filter records created on or before this UTC timestamp"
    ),
    source: Optional[str] = Query(
        "all", description="Source filter: 'webapp', 'scheduled', or 'all'"
    ),
    limit: int = Query(100, ge=1, le=1000, description="Max number of records to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    db: Session = Depends(get_db),
):
    """
    Retrieve past predictions stored in the database.
    Supports filtering by date range and source ('webapp', 'scheduled', 'all').
    """
    query = db.query(PredictionRecord)

    if source and source.lower() != "all":
        query = query.filter(PredictionRecord.source == source.lower())

    if start_date:
        query = query.filter(PredictionRecord.created_at >= start_date)

    if end_date:
        query = query.filter(PredictionRecord.created_at <= end_date)

    total = query.count()
    records = query.order_by(desc(PredictionRecord.created_at)).offset(offset).limit(limit).all()

    items = [
        PastPredictionItem(
            id=rec.id,
            customer_id=rec.customer_id,
            source=rec.source,
            prediction=rec.prediction,
            churn_probability=rec.churn_probability,
            model_version=rec.model_version,
            tenure=rec.tenure,
            monthly_charges=rec.monthly_charges,
            total_charges=rec.total_charges,
            contract=rec.contract,
            internet_service=rec.internet_service,
            features_payload=rec.features_payload,
            created_at=rec.created_at,
        )
        for rec in records
    ]

    return PastPredictionsResponse(total=total, items=items)


@app.post("/reload-model", response_model=ReloadModelResponse, tags=["Model Lifecycle"])
def reload_model():
    """
    Reloads the model at runtime from MLflow Model Registry (@champion) or local artifact.
    Used by the Airflow Retraining DAG after successful promotion.
    """
    success, message = model_manager.reload_model()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=message,
        )
    return ReloadModelResponse(
        status="success",
        message=message,
        model_version=model_manager.model_version,
        model_source=model_manager.model_source,
    )
