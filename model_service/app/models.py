"""
SQLAlchemy ORM Database Models
Supports:
1. Predictions (used by API and UI)
2. Ingestion Stats (used by Airflow Ingestion DAG & Grafana Data Quality Dashboard)
3. Training Stats (used by Airflow Training DAG & Grafana Data Drift Dashboard)
"""
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    JSON,
    Text,
)
from .database import Base


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    customer_id = Column(String(50), nullable=True, index=True)
    source = Column(String(20), nullable=False, default="webapp", index=True)  # 'webapp' or 'scheduled'
    prediction = Column(String(10), nullable=False)  # 'Yes' or 'No'
    churn_probability = Column(Float, nullable=False)
    model_version = Column(String(50), nullable=False, default="v1.0.0", index=True)
    
    # Core feature columns for quick SQL indexing/filtering in Grafana
    tenure = Column(Float, nullable=True)
    monthly_charges = Column(Float, nullable=True)
    total_charges = Column(Float, nullable=True)
    contract = Column(String(50), nullable=True)
    internet_service = Column(String(50), nullable=True)
    payment_method = Column(String(100), nullable=True)
    
    # Full feature payload stored as JSON for fidelity
    features_payload = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class IngestionStatRecord(Base):
    __tablename__ = "ingestion_stats"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    batch_id = Column(String(100), nullable=False, index=True)
    file_name = Column(String(255), nullable=False)
    total_rows = Column(Integer, nullable=False, default=10)
    valid_rows = Column(Integer, nullable=False)
    invalid_rows = Column(Integer, nullable=False)
    error_rate = Column(Float, nullable=False)  # 0.0 to 1.0
    
    # Categorized error counts
    completeness_errors = Column(Integer, default=0)
    validity_errors = Column(Integer, default=0)
    consistency_errors = Column(Integer, default=0)
    schema_errors = Column(Integer, default=0)
    type_errors = Column(Integer, default=0)
    duplicate_errors = Column(Integer, default=0)
    format_errors = Column(Integer, default=0)
    cross_field_errors = Column(Integer, default=0)
    outlier_errors = Column(Integer, default=0)
    
    criticality = Column(String(20), nullable=False, index=True)  # 'Low', 'Medium', 'High'
    report_url = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)


class TrainingStatRecord(Base):
    __tablename__ = "training_stats"

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    model_version = Column(String(50), nullable=False, index=True)
    model_name = Column(String(100), nullable=False, default="TelecomChurnClassifier")
    run_id = Column(String(100), nullable=True)
    
    # Performance metrics
    accuracy = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    roc_auc = Column(Float, nullable=True)
    inference_time_ms = Column(Float, nullable=True)
    
    # Full distribution statistics stored as JSON for baseline drift comparisons
    feature_stats = Column(JSON, nullable=False)
    metrics_payload = Column(JSON, nullable=True)
    is_champion = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), index=True)
