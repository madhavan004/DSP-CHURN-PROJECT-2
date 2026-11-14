
"""Populate PostgreSQL with historical data for Grafana dashboards."""

from datetime import datetime, timedelta, timezone
import os
import random
import sys

# Support direct script execution.
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from model_service.app.database import SessionLocal, init_db  # noqa: E402
from model_service.app.models import (  # noqa: E402
    IngestionStatRecord,
    PredictionRecord,
    TrainingStatRecord,
)


def populate_historical_data(hours_back: int = 168, records_per_hour: int = 15):
    """Generate historical ingestion, prediction, and training records."""
    print("[populate_data] Initializing database tables...")
    init_db()
    db = SessionLocal()

    try:
        now = datetime.now(timezone.utc)
        rng = random.Random(12345)

        print(
            f"[populate_data] Generating historical data "
            f"for past {hours_back} hours..."
        )

        # 1. Populate ingestion statistics.
        ingestion_records = []

        for h in range(hours_back, -1, -1):
            for b in range(10):
                batch_time = now - timedelta(
                    hours=h,
                    minutes=b * 6,
                    seconds=rng.randint(0, 50),
                )

                if rng.random() < 0.20:
                    invalid_rows = rng.choice([2, 3, 4, 7, 10])
                else:
                    invalid_rows = rng.choice([0, 0, 0, 1])

                total_rows = 10
                valid_rows = total_rows - invalid_rows
                error_rate = invalid_rows / total_rows

                comp_err = (
                    rng.randint(0, invalid_rows)
                    if invalid_rows > 0 else 0
                )
                val_err = (
                    rng.randint(0, invalid_rows - comp_err)
                    if invalid_rows > comp_err else 0
                )
                cons_err = invalid_rows - comp_err - val_err
                type_err = 1 if rng.random() < 0.1 else 0
                schema_err = (
                    1
                    if invalid_rows == 10 and rng.random() < 0.2
                    else 0
                )

                if schema_err or error_rate > 0.50:
                    criticality = "High"
                elif error_rate >= 0.10:
                    criticality = "Medium"
                elif error_rate > 0.0:
                    criticality = "Low"
                else:
                    criticality = "None"

                report_file = (
                    f"report_batch_{h}_{b}_"
                    f"{batch_time.strftime('%Y%m%d_%H%M%S')}.html"
                )
                report_url = f"http://localhost:8080/{report_file}"

                record = IngestionStatRecord(
                    batch_id=(
                        f"batch_{batch_time.strftime('%Y%m%d%H%M%S')}_{b}"
                    ),
                    file_name=f"raw_batch_{h:02d}_{b:02d}.csv",
                    total_rows=total_rows,
                    valid_rows=valid_rows,
                    invalid_rows=invalid_rows,
                    error_rate=error_rate,
                    completeness_errors=comp_err,
                    validity_errors=val_err,
                    consistency_errors=cons_err,
                    schema_errors=schema_err,
                    type_errors=type_err,
                    duplicate_errors=1 if rng.random() < 0.1 else 0,
                    format_errors=1 if rng.random() < 0.15 else 0,
                    cross_field_errors=1 if rng.random() < 0.1 else 0,
                    outlier_errors=1 if rng.random() < 0.05 else 0,
                    criticality=criticality,
                    report_url=report_url,
                    created_at=batch_time,
                )
                ingestion_records.append(record)

        db.add_all(ingestion_records)
        db.commit()

        print(
            f"[populate_data] Inserted {len(ingestion_records)} "
            "historical IngestionStat records."
        )

        # 2. Populate predictions.
        contracts = ["Month-to-month", "One year", "Two year"]
        internets = ["DSL", "Fiber optic", "No"]
        prediction_records = []

        for h in range(hours_back, -1, -1):
            for r in range(records_per_hour):
                pred_time = now - timedelta(
                    hours=h,
                    minutes=rng.randint(0, 59),
                    seconds=rng.randint(0, 59),
                )

                contract = rng.choices(
                    contracts,
                    weights=[
                        0.55 + (0.01 * (hours_back - h)),
                        0.25,
                        0.20,
                    ],
                )[0]
                internet = rng.choices(
                    internets,
                    weights=[0.4, 0.45, 0.15],
                )[0]
                tenure = float(rng.randint(1, 72))

                base_monthly = 64.76 + (0.5 * (hours_back - h))
                monthly_charges = round(
                    max(18.0, rng.gauss(base_monthly, 25.0)), 2
                )
                total_charges = round(
                    monthly_charges * max(1.0, tenure), 2
                )

                # Churn probability heuristic.
                prob = 0.25

                if contract == "Month-to-month":
                    prob += 0.30
                if internet == "Fiber optic":
                    prob += 0.15
                if tenure < 6:
                    prob += 0.20

                prob = min(
                    0.95,
                    max(0.05, prob + rng.gauss(0, 0.08)),
                )

                prediction = "Yes" if prob >= 0.50 else "No"
                source = "webapp" if rng.random() < 0.35 else "scheduled"
                cid = (
                    f"{rng.randint(1000, 9999)}-"
                    f"{''.join(rng.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}"
                )

                feat_payload = {
                    "customerID": cid,
                    "gender": rng.choice(["Female", "Male"]),
                    "SeniorCitizen": rng.choice([0, 1]),
                    "Partner": rng.choice(["Yes", "No"]),
                    "Dependents": rng.choice(["Yes", "No"]),
                    "tenure": tenure,
                    "PhoneService": "Yes",
                    "MultipleLines": rng.choice(["No", "Yes"]),
                    "InternetService": internet,
                    "OnlineSecurity": rng.choice(["Yes", "No"]),
                    "OnlineBackup": rng.choice(["Yes", "No"]),
                    "DeviceProtection": rng.choice(["Yes", "No"]),
                    "TechSupport": rng.choice(["Yes", "No"]),
                    "StreamingTV": rng.choice(["Yes", "No"]),
                    "StreamingMovies": rng.choice(["Yes", "No"]),
                    "Contract": contract,
                    "PaperlessBilling": rng.choice(["Yes", "No"]),
                    "PaymentMethod": rng.choice([
                        "Electronic check",
                        "Mailed check",
                        "Bank transfer (automatic)",
                        "Credit card (automatic)",
                    ]),
                    "MonthlyCharges": monthly_charges,
                    "TotalCharges": total_charges,
                }

                prediction_record = PredictionRecord(
                    customer_id=cid,
                    source=source,
                    prediction=prediction,
                    churn_probability=round(prob, 4),
                    model_version="v1.0.0",
                    tenure=tenure,
                    monthly_charges=monthly_charges,
                    total_charges=total_charges,
                    contract=contract,
                    internet_service=internet,
                    payment_method=feat_payload["PaymentMethod"],
                    features_payload=feat_payload,
                    created_at=pred_time,
                )
                prediction_records.append(prediction_record)

        db.add_all(prediction_records)
        db.commit()

        print(
            f"[populate_data] Inserted {len(prediction_records)} "
            "historical Prediction records."
        )

        # 3. Populate baseline training statistics.
        baseline_stats = {
            "numerical": {
                "tenure": {
                    "mean": 32.37,
                    "std": 24.56,
                    "min": 0,
                    "max": 72,
                },
                "MonthlyCharges": {
                    "mean": 64.76,
                    "std": 30.09,
                    "min": 18.25,
                    "max": 118.75,
                },
                "TotalCharges": {
                    "mean": 2283.30,
                    "std": 2266.77,
                    "min": 0,
                    "max": 8684.80,
                },
            },
            "categorical": {
                "Contract": {
                    "Month-to-month": 0.5502,
                    "Two year": 0.2407,
                    "One year": 0.2091,
                },
                "InternetService": {
                    "Fiber optic": 0.4396,
                    "DSL": 0.3437,
                    "No": 0.2167,
                },
            },
        }

        training_record = TrainingStatRecord(
            model_version="v1.0.0",
            model_name="TelecomChurnClassifier",
            run_id="baseline_initial_run",
            accuracy=0.8041,
            f1_score=0.5881,
            roc_auc=0.8381,
            inference_time_ms=0.030,
            feature_stats=baseline_stats,
            metrics_payload={
                "accuracy": 0.8041,
                "f1": 0.5881,
                "roc_auc": 0.8381,
            },
            is_champion=True,
            created_at=now - timedelta(hours=hours_back),
        )

        db.add(training_record)
        db.commit()

        print(
            "[populate_data] Successfully seeded Training baseline stats."
        )

    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    populate_historical_data()
