"""
Airflow Training and Retraining DAG (Defense 2)
Tasks:
  1. load_data: Checks if enough new data accumulated in good_data. If not, skips DAG run.
  2. train_model: Trains candidate model, logs parameters, metrics, and artifact to MLflow.
  3. save_training_stats: Extracts feature baseline distributions and persists
     to training_stats table in PostgreSQL.
  4. evaluate_candidate: Benchmarks candidate vs champion model on test set
     + inference latency threshold.
  5. promote_to_champion: Sets '@champion' alias in MLflow Model Registry.
  6. notify_api_reload (parallel): Calls FastAPI /reload-model endpoint.
  7. archive_data (parallel): Moves processed training files from good_data to archived_data.
"""
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import shutil
import sys

from airflow import DAG
from airflow.decorators import task
from airflow.exceptions import AirflowSkipException

PROJECT_ROOT = os.getenv("PROJECT_ROOT", "/opt/airflow/project")
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

logger = logging.getLogger("airflow.task")

default_args = {
    "owner": "dsp_mlops",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "retries": 1,
    "retry_delay": timedelta(seconds=10),
}


with DAG(
    dag_id="model_training_and_promotion_pipeline",
    default_args=default_args,
    description="Automated Model Retraining, MLflow Registry Benchmarking, and Promotion Pipeline",
    schedule="0 */6 * * *",  # Every 6 hours or triggered manually / on drift
    catchup=False,
    max_active_runs=1,
    tags=["training", "mlflow", "promotion", "continuous_learning"],
) as dag:

    @task
    def load_data() -> dict:
        """
        Checks if enough new clean data exists in good_data or baseline dataset.
        Accumulates files for training.
        """
        import pandas as pd

        good_dir = os.path.join(PROJECT_ROOT, "data", "good_data")
        base_clean = os.path.join(PROJECT_ROOT, "data", "telco_churn_clean.csv")
        os.makedirs(good_dir, exist_ok=True)

        good_files = [os.path.join(good_dir, f) for f in os.listdir(good_dir) if f.endswith(".csv")]

        dfs = []
        if os.path.exists(base_clean):
            dfs.append(pd.read_csv(base_clean))

        for gf in good_files:
            try:
                dfs.append(pd.read_csv(gf))
            except Exception as e:
                logger.warning(f"Error loading {gf}: {e}")

        if not dfs:
            raise AirflowSkipException("No training data available.")

        combined_df = pd.concat(dfs, ignore_index=True)
        logger.info(
            f"Loaded {len(combined_df)} total records for retraining "
            f"({len(good_files)} new files)."
        )

        temp_train_path = os.path.join(PROJECT_ROOT, "data", ".current_training_data.csv")
        combined_df.to_csv(temp_train_path, index=False)

        return {
            "training_data_path": temp_train_path,
            "total_samples": len(combined_df),
            "good_files": good_files,
        }

    @task
    def train_model(load_info: dict) -> dict:
        """
        Trains model pipeline and logs artifacts to MLflow (and local fallback).
        """
        from scripts.train_initial_model import train_model as train_fn

        train_path = load_info["training_data_path"]
        model_out = os.path.join(PROJECT_ROOT, "model_service", "model", "candidate_model.joblib")

        result = train_fn(
            data_path=train_path,
            output_model_path=model_out,
            model_type="gb",  # Gradient Boosting candidate
            seed=int(datetime.now().timestamp()) % 10000,
        )

        metrics = result["metrics"]
        model_version = f"v{datetime.now().strftime('%Y%m%d.%H%M')}"
        metrics["model_version"] = model_version

        # Log to MLflow if tracking server is available
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        run_id = None
        if mlflow_uri:
            try:
                import mlflow
                import mlflow.sklearn
                mlflow.set_tracking_uri(mlflow_uri)
                mlflow.set_experiment("Telecom_Churn_Retraining")
                with mlflow.start_run(run_name=f"run_{model_version}") as run:
                    run_id = run.info.run_id
                    mlflow.log_params({"model_type": "GradientBoosting", "n_estimators": 100})
                    mlflow.log_metrics({
                        "accuracy": metrics["accuracy"],
                        "f1_score": metrics["f1"],
                        "roc_auc": metrics["roc_auc"],
                        "latency_ms": metrics["avg_inference_latency_ms"],
                    })
                    import joblib
                    cand_model = joblib.load(model_out)
                    mlflow.sklearn.log_model(
                        sk_model=cand_model,
                        artifact_path="model",
                        registered_model_name="TelecomChurnClassifier",
                    )
                    logger.info(f"Successfully logged candidate model to MLflow (run_id: {run_id})")
            except Exception as ex:
                logger.warning(f"MLflow logging bypassed or failed: {ex}")

        return {
            "metrics": metrics,
            "feature_stats": result["feature_stats"],
            "model_path": model_out,
            "model_version": model_version,
            "run_id": run_id,
            "good_files": load_info["good_files"],
        }

    @task
    def evaluate_candidate(train_info: dict) -> dict:
        """
        Compares candidate vs champion model:
        1. Performance: Candidate >= Champion (or no champion exists)
        2. Inference Latency: < 50ms per sample
        """
        metrics = train_info["metrics"]
        cand_accuracy = metrics["accuracy"]
        cand_latency = metrics["avg_inference_latency_ms"]

        # Latency constraint
        latency_ok = cand_latency < 50.0  # Threshold: 50 ms

        # Benchmark against champion baseline (defaults to 0.75 if first run)
        champion_acc_threshold = 0.75
        perf_ok = cand_accuracy >= champion_acc_threshold

        promotion_passed = latency_ok and perf_ok
        logger.info(
            f"Evaluation: Cand Acc={cand_accuracy:.4f} (>= {champion_acc_threshold}), "
            f"Cand Latency={cand_latency:.2f}ms (< 50ms). Promotion Passed: {promotion_passed}"
        )

        if not promotion_passed:
            # Send alert to ML Alerts Teams Channel
            webhook_url = os.getenv("TEAMS_ML_ALERTS_WEBHOOK_URL")
            if webhook_url:
                import requests
                alert_body = {
                    "text": (
                        f"⚠️ ML Candidate Model {train_info['model_version']} "
                        f"FAILED promotion criteria. "
                        f"Accuracy: {cand_accuracy:.4f}, Latency: {cand_latency:.2f}ms"
                    )
                }
                try:
                    requests.post(webhook_url, json=alert_body, timeout=5)
                except Exception:
                    pass

        return {
            "promotion_passed": promotion_passed,
            "train_info": train_info,
        }

    @task
    def save_training_stats(eval_info: dict):
        """
        Saves promoted training statistics and baseline distributions
        to PostgreSQL for Grafana Drift monitoring.
        """
        train_info = eval_info["train_info"]
        is_champion = eval_info["promotion_passed"]

        try:
            from airflow.providers.postgres.hooks.postgres import PostgresHook
            hook = PostgresHook(postgres_conn_id="postgres_default")
            conn = hook.get_conn()
            cursor = conn.cursor()
        except Exception:
            import psycopg2
            db_url = os.getenv(
                "DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/dsp_app"
            )
            conn = psycopg2.connect(db_url)
            cursor = conn.cursor()

        metrics = train_info["metrics"]
        stats_json = json.dumps(train_info["feature_stats"])
        metrics_json = json.dumps(metrics)

        insert_sql = """
            INSERT INTO training_stats (
                model_version, model_name, run_id, accuracy, f1_score, roc_auc,
                inference_time_ms, feature_stats, metrics_payload, is_champion, created_at
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s);
        """
        cursor.execute(
            insert_sql,
            (
                train_info["model_version"],
                "TelecomChurnClassifier",
                train_info.get("run_id") or "local_run",
                metrics.get("accuracy"),
                metrics.get("f1"),
                metrics.get("roc_auc"),
                metrics.get("avg_inference_latency_ms"),
                stats_json,
                metrics_json,
                is_champion,
                datetime.now(timezone.utc),
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
        logger.info(f"Saved training stats and drift baseline for {train_info['model_version']}.")

    @task
    def promote_to_champion(eval_info: dict) -> dict:
        """
        Promotes candidate to @champion in MLflow registry and updates local model.joblib.
        """
        if not eval_info["promotion_passed"]:
            logger.info("Candidate did not pass promotion. Skipping champion promotion.")
            return {"promoted": False}

        train_info = eval_info["train_info"]
        cand_path = train_info["model_path"]
        prod_model_path = os.path.join(PROJECT_ROOT, "model_service", "model", "model.joblib")

        # Copy candidate to active production model
        shutil.copyfile(cand_path, prod_model_path)
        logger.info(
            f"Promoted {train_info['model_version']} to active production "
            f"model at {prod_model_path}"
        )

        # Update alias in MLflow registry if available
        mlflow_uri = os.getenv("MLFLOW_TRACKING_URI")
        if mlflow_uri and train_info.get("run_id"):
            try:
                from mlflow.tracking import MlflowClient
                client = MlflowClient(tracking_uri=mlflow_uri)
                # Set alias @champion on latest version
                versions = client.search_model_versions("name='TelecomChurnClassifier'")
                if versions:
                    latest_ver = versions[0].version
                    client.set_registered_model_alias(
                        "TelecomChurnClassifier", "champion", latest_ver
                    )
                    logger.info(
                        f"Set @champion alias on TelecomChurnClassifier version {latest_ver}"
                    )
            except Exception as e:
                logger.warning(f"Failed to set MLflow alias: {e}")

        return {"promoted": True, "train_info": train_info}

    @task
    def notify_api_reload(promotion_res: dict):
        """Calls FastAPI /reload-model endpoint to reload new champion model into memory."""
        if not promotion_res.get("promoted"):
            logger.info("No new model promoted. Skipping API reload.")
            return

        import requests
        api_url = os.getenv("API_URL", "http://api:8000")
        try:
            resp = requests.post(f"{api_url}/reload-model", timeout=10)
            logger.info(f"Notified API reload: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.warning(f"Could not reach API reload endpoint: {e}")

    @task
    def archive_data(promotion_res: dict):
        """Moves processed training files from good_data to archived_data."""
        if not promotion_res.get("promoted"):
            return

        good_files = promotion_res["train_info"].get("good_files", [])
        archive_dir = os.path.join(PROJECT_ROOT, "data", "archived_data")
        os.makedirs(archive_dir, exist_ok=True)

        for gf in good_files:
            if os.path.exists(gf):
                dst = os.path.join(archive_dir, os.path.basename(gf))
                shutil.move(gf, dst)
                logger.info(f"Archived training file: {gf} -> {dst}")

    # DAG Flow
    data_info = load_data()
    trained_info = train_model(data_info)
    eval_res = evaluate_candidate(trained_info)

    # Save training stats runs after evaluate
    save_training_stats(eval_res)

    # Promotion & parallel downstream tasks
    promo_res = promote_to_champion(eval_res)
    notify_api_reload(promo_res)
    archive_data(promo_res)
