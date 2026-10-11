"""Retrain pipeline: loads best params, retrains on new data, compares, and optionally promotes."""
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load import load_data
from src.data.preprocess import preprocess
from src.features.encoder import TargetEncoderWrapper
from src.training.weights import compute_normalized_weights
from src.training.train import calibrate_model
from src.evaluation.metrics import compute_all_metrics


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def retrain(new_data_path: Path | None = None):
    """Retrain the best model on fresh data and compare against the current production pickle."""
    cfg = load_config()
    rs = cfg["training"]["random_state"]
    mono = cfg["features"]["monotonic_constraints"]

    models_dir = PROJECT_ROOT / cfg["output"]["models_dir"]
    pickle_path = models_dir / cfg["output"]["pickle_name"]

    # Load current production pipeline
    with open(pickle_path, "rb") as f:
        current = pickle.load(f)
    print(f"Current model: {current['model_name']} | PR-AUC: {current['metrics']['pr_auc']:.4f}")

    # Load data (new or original)
    if new_data_path:
        # Assumes new data has the same schema as the processed files
        print(f"Loading new data from {new_data_path}")
        df = load_data(
            raw_path=new_data_path,
            processed_dir=PROJECT_ROOT / cfg["data"]["processed_dir"],
        )
    else:
        df = load_data(
            raw_path=PROJECT_ROOT / cfg["data"]["raw_path"],
            processed_dir=PROJECT_ROOT / cfg["data"]["processed_dir"],
        )

    X, y = preprocess(
        df,
        cols_to_drop=cfg["features"]["cols_to_drop"],
        skewed_cols=cfg["features"]["skewed_cols"],
        cat_cols=cfg["features"]["cat_cols"],
        drop_from_features=cfg["features"]["drop_from_features"],
    )

    # Retrain with stored best params
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=rs,
    )

    te = TargetEncoderWrapper(cfg["features"]["cat_cols"], **cfg["encoder"])
    X_train_enc = te.fit_transform(X_train, y_train)
    X_cal_enc = te.transform(X_cal)

    scale_pos = (len(y_train) - sum(y_train)) / sum(y_train)
    weights = compute_normalized_weights(X_train_enc, y_train, scale_pos, **cfg["weights"])

    name = current["model_name"]
    params = current["best_params"]

    if name == "XGBoost":
        model = XGBClassifier(**params, monotone_constraints=mono, n_jobs=-1, eval_metric="auc", random_state=42)
    elif name == "LightGBM":
        model = LGBMClassifier(**params, n_jobs=-1, verbose=-1, random_state=42)
    elif name == "CatBoost":
        model = CatBoostClassifier(**params, eval_metric="AUC", random_seed=42, verbose=0)

    model.fit(X_train_enc, y_train, sample_weight=weights)
    calibrator = calibrate_model(model, X_cal_enc, y_cal)

    # Evaluate on calibration set
    preds = calibrator.predict_proba(X_cal_enc)[:, 1]
    new_metrics = compute_all_metrics(y_cal, preds)
    print(f"Retrained  : PR-AUC: {new_metrics['pr_auc']:.4f} | F2: {new_metrics['f2']:.4f}")

    # Compare and promote
    if new_metrics["pr_auc"] > current["metrics"]["pr_auc"]:
        print("New model is better. Saving...")
        pipeline = {
            "model_name": name,
            "best_params": params,
            "target_encoder": te.encoder,
            "calibrated_model": calibrator,
            "cat_cols": cfg["features"]["cat_cols"],
            "skewed_cols": cfg["features"]["skewed_cols"],
            "feature_columns": list(X_train_enc.columns),
            "optimal_f2_threshold": new_metrics["f2_threshold"],
            "metrics": new_metrics,
        }
        with open(pickle_path, "wb") as f:
            pickle.dump(pipeline, f)
        print(f"Updated {pickle_path}")

        # Log to MLflow if available
        try:
            import mlflow
            from mlflow_config import setup_mlflow
            setup_mlflow(PROJECT_ROOT / "config" / "config.yaml")
            with mlflow.start_run(run_name=f"retrain_{name}"):
                mlflow.log_params(params)
                mlflow.log_metrics(new_metrics)
                mlflow.log_artifact(str(pickle_path))
            print("Logged to MLflow.")
        except ImportError:
            pass
    else:
        print("Current model is still better. No update.")


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    retrain(path)
