"""Retrain the production model's architecture on fresh data and promote it only if PR-AUC improves."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split

from src.config import PROJECT_ROOT, load_config
from src.data.load import load_data
from src.data.preprocess import preprocess
from src.evaluation.metrics import compute_all_metrics
from src.inference.predict import build_bundle, load_pipeline, save_pipeline
from src.tracking import log_run
from src.training.train import create_classifier, fit_calibrated


def retrain(new_data_path: Path | None = None) -> None:
    cfg = load_config()
    pickle_path = PROJECT_ROOT / cfg["output"]["models_dir"] / cfg["output"]["pickle_name"]

    current = load_pipeline(pickle_path)
    name, params = current["model_name"], current["best_params"]
    print(f"Current model: {name} | PR-AUC: {current['metrics']['pr_auc']:.4f}")

    raw_path = new_data_path or PROJECT_ROOT / cfg["data"]["raw_path"]
    df = load_data(raw_path)
    X, y = preprocess(df, cfg["features"])
    X_train, X_cal, y_train, y_cal = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=cfg["training"]["random_state"],
    )

    model = create_classifier(name, params, cfg["features"]["monotonic_constraints"])
    encoder, calibrator = fit_calibrated(model, X_train, y_train, X_cal, y_cal, cfg)
    new_metrics = compute_all_metrics(y_cal, calibrator.predict_proba(encoder.transform(X_cal))[:, 1])
    print(f"Retrained  : PR-AUC: {new_metrics['pr_auc']:.4f} | F2: {new_metrics['f2']:.4f}")

    if new_metrics["pr_auc"] <= current["metrics"]["pr_auc"]:
        print("Current production model remains superior. No update made.")
        return

    print("New model outperformed current. Promoting to production...")
    bundle = build_bundle(
        name, params, encoder.encoder, calibrator, list(X.columns),
        new_metrics["f2_threshold"], new_metrics, cfg["features"],
    )
    save_pipeline(bundle, pickle_path)
    print(f"Updated {pickle_path}")
    log_run(f"retrain_{name}", params, new_metrics, [pickle_path])


if __name__ == "__main__":
    retrain(Path(sys.argv[1]) if len(sys.argv) > 1 else None)
