from datetime import datetime
from typing import List, Optional

from sqlalchemy import and_
from sqlalchemy.orm import Session

from app import models


def create_predictions(
    db: Session,
    features_list: List[dict],
    results: List[dict],
    model_version: str,
    source: str,
) -> List[models.Prediction]:
    db_objects = []
    for features, result in zip(features_list, results):
        customer_id = features.get("customerID")
        db_obj = models.Prediction(
            customer_id=customer_id,
            features=features,
            prediction=result["prediction"],
            churn_probability=result["churn_probability"],
            model_version=model_version,
            source=source,
        )
        db.add(db_obj)
        db_objects.append(db_obj)

    db.commit()
    for obj in db_objects:
        db.refresh(obj)
    return db_objects


def get_past_predictions(
    db: Session,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    source: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
):
    query = db.query(models.Prediction)

    filters = []
    if start_date is not None:
        filters.append(models.Prediction.created_at >= start_date)
    if end_date is not None:
        filters.append(models.Prediction.created_at <= end_date)
    if source is not None and source != "all":
        filters.append(models.Prediction.source == source)

    if filters:
        query = query.filter(and_(*filters))

    total_count = query.count()
    results = (
        query.order_by(models.Prediction.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )
    return results, total_count
