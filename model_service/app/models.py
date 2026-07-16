from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, Float, Integer, String
from sqlalchemy.sql import func

from app.database import Base


def utcnow():
    return datetime.now(timezone.utc)


class Prediction(Base):
    """
    Stores every prediction made by the model service, whether triggered
    on-demand from the webapp or by the scheduled prediction Airflow DAG.

    `features` keeps the full feature payload used for the prediction so
    the UI can display it, and so the Grafana drift dashboard can compare
    served feature distributions against training statistics.
    """

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    customer_id = Column(String, index=True, nullable=True)

    # Raw feature payload used to make this prediction (JSON blob).
    features = Column(JSON, nullable=False)

    prediction = Column(String, nullable=False)  # "Yes" / "No"
    churn_probability = Column(Float, nullable=False)

    model_version = Column(String, nullable=False, index=True)

    # Where the prediction request came from: "webapp" or "scheduled"
    source = Column(String, nullable=False, index=True)

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )


class IngestionStat(Base):
    """
    Summary statistics for a single ingestion DAG run (one raw_data file).
    Stores aggregated counts per error category rather than raw rows, so
    the Grafana data-quality dashboard can query trends over time.
    """

    __tablename__ = "ingestion_stats"

    id = Column(Integer, primary_key=True, index=True)

    file_name = Column(String, nullable=False, index=True)
    total_rows = Column(Integer, nullable=False)
    valid_rows = Column(Integer, nullable=False)
    invalid_rows = Column(Integer, nullable=False)

    criticality = Column(String, nullable=False, index=True)  # none/low/medium/high

    # Per data-quality-category error counts (row-level counts, a row can
    # contribute to more than one category).
    completeness_errors = Column(Integer, nullable=False, default=0)
    validity_errors = Column(Integer, nullable=False, default=0)
    consistency_errors = Column(Integer, nullable=False, default=0)
    schema_errors = Column(Integer, nullable=False, default=0)
    type_errors = Column(Integer, nullable=False, default=0)
    duplicate_errors = Column(Integer, nullable=False, default=0)
    cross_field_errors = Column(Integer, nullable=False, default=0)

    report_file_name = Column(String, nullable=True)

    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), index=True
    )
