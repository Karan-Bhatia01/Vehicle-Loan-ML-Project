import mlflow
import yaml
from pathlib import Path


def setup_mlflow(config_path: Path | None = None) -> str:
    """Configure MLflow tracking from config.yaml. Returns experiment name."""
    if config_path is None:
        config_path = Path(__file__).resolve().parent / "config" / "config.yaml"

    with open(config_path) as f:
        cfg = yaml.safe_load(f)["mlflow"]

    mlflow.set_tracking_uri(cfg["tracking_uri"])
    mlflow.set_experiment(cfg["experiment_name"])
    return cfg["experiment_name"]
