import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlalchemy.orm import Session

from app import crud
from app.database import get_db, init_db
from app.ml_model import churn_model
from app.schemas import (
    HealthResponse,
    PastPredictionsResponse,
    PredictionItem,
    PredictionRequest,
    PredictionResponse,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("model_service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create tables if needed, load the model ONCE (not per request)
    init_db()
    churn_model.load()
    logger.info("Model service startup complete.")
    yield
    # Shutdown: nothing to clean up for now
    logger.info("Model service shutting down.")


app = FastAPI(
    title="Telco Churn Model Service",
    description="Serves churn predictions, and stores/retrieves them from PostgreSQL.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check():
    """Health check used by Docker healthchecks and dependent services."""
    if not churn_model.is_ready:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse, tags=["predictions"])
def predict(request: PredictionRequest, db: Session = Depends(get_db)):
    """
    Make single or batch churn predictions.

    Accepts a list of feature rows (`features`) validated against the
    Telco Customer Churn schema, and an optional `source` field
    ("webapp" or "scheduled") used to track where the request came from.
    Predictions and the features used to make them are saved to the
    database.
    """
    if not churn_model.is_ready:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    if not request.features:
        raise HTTPException(status_code=400, detail="No features provided")

    try:
        results = churn_model.predict(request.features)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500, detail=f"Prediction failed: {exc}"
        ) from exc

    features_dicts = [f.model_dump() for f in request.features]
    db_objects = crud.create_predictions(
        db=db,
        features_list=features_dicts,
        results=results,
        model_version=churn_model.model_version,
        source=request.source.value,
    )

    predictions = [PredictionItem.model_validate(obj) for obj in db_objects]
    return PredictionResponse(predictions=predictions, count=len(predictions))


@app.get(
    "/past-predictions", response_model=PastPredictionsResponse, tags=["predictions"]
)
def past_predictions(
    start_date: Optional[datetime] = Query(
        default=None, description="Filter predictions made on/after this date"
    ),
    end_date: Optional[datetime] = Query(
        default=None, description="Filter predictions made on/before this date"
    ),
    source: Optional[str] = Query(
        default="all", description="webapp | scheduled | all"
    ),
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    """Retrieve past predictions, filterable by date range and source."""
    if source not in ("webapp", "scheduled", "all"):
        raise HTTPException(
            status_code=400,
            detail="source must be one of: webapp, scheduled, all",
        )

    db_objects, total_count = crud.get_past_predictions(
        db=db,
        start_date=start_date,
        end_date=end_date,
        source=source,
        limit=limit,
        offset=offset,
    )
    predictions = [PredictionItem.model_validate(obj) for obj in db_objects]
    return PastPredictionsResponse(predictions=predictions, count=total_count)
