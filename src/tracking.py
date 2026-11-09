from pathlib import Path

from src.config import load_config


def log_run(run_name: str, params: dict, metrics: dict, artifacts: list[Path]) -> None:
    """Log one run to MLflow. Tracking is optional, so any failure is reported and skipped."""
    try:
        import mlflow

        cfg = load_config()["mlflow"]
        mlflow.set_tracking_uri(cfg["tracking_uri"])
        mlflow.set_experiment(cfg["experiment_name"])
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            for path in artifacts:
                mlflow.log_artifact(str(path))
        print("Logged to MLflow.")
    except Exception as e:
        print(f"MLflow logging skipped: {e}")
