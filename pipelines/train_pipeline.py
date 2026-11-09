"""Full training pipeline: load -> preprocess -> OOF compare -> tune -> final fit + calibrate -> export."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split

from src.config import PROJECT_ROOT, load_config
from src.data.load import load_data
from src.data.preprocess import preprocess
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.plots import plot_feature_importance, plot_reliability_diagram
from src.features.encoder import TargetEncoderWrapper
from src.inference.predict import build_bundle, save_pipeline
from src.tracking import log_run
from src.training.train import create_classifier, fit_calibrated, train_oof
from src.training.tune import tune_models
from src.training.weights import compute_normalized_weights

BASE_PARAMS = {
    "XGBoost": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    "LightGBM": {"n_estimators": 300, "max_depth": 5, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
    "CatBoost": {"iterations": 300, "depth": 5, "learning_rate": 0.05},
}


def run() -> None:
    cfg = load_config()
    rs = cfg["training"]["random_state"]
    mono = cfg["features"]["monotonic_constraints"]
    charts_dir = PROJECT_ROOT / cfg["output"]["charts_dir"]

    print("Loading data...")
    df = load_data(PROJECT_ROOT / cfg["data"]["raw_path"])
    X, y = preprocess(df, cfg["features"])
    print(f"Features: {X.shape[1]} | Rows: {len(X)} | Default rate: {y.mean():.2%}")

    # 1. Compare base models with out-of-fold predictions
    print("\n=== OOF Training ===")
    models = {name: create_classifier(name, params, mono) for name, params in BASE_PARAMS.items()}
    oof_preds = train_oof(models, X, y, cfg)

    scores = {name: compute_all_metrics(y, preds) for name, preds in oof_preds.items()}
    for name, m in scores.items():
        print(f"{name:<12} PR: {m['pr_auc']:.4f} | F2: {m['f2']:.4f} | Brier: {m['brier']:.4f}")
    ranked = sorted(scores, key=lambda name: scores[name]["pr_auc"], reverse=True)
    plot_reliability_diagram(
        y, {name: oof_preds[name] for name in ranked},
        "Base Models Reliability Diagram", charts_dir / "12a_base_calibration.png",
    )

    # 2. Tune every model, pick the best on holdout PR-AUC
    print("\n=== Hyperparameter Tuning ===")
    X_tune, X_holdout, y_tune, y_holdout = train_test_split(X, y, test_size=0.20, stratify=y, random_state=rs)
    encoder = TargetEncoderWrapper(cfg["features"]["cat_cols"], **cfg["encoder"])
    X_tune_enc = encoder.fit_transform(X_tune, y_tune)
    X_holdout_enc = encoder.transform(X_holdout)
    scale_pos_weight = (len(y_tune) - y_tune.sum()) / y_tune.sum()
    weights = compute_normalized_weights(X_tune_enc, y_tune, scale_pos_weight, **cfg["weights"])

    tuned = tune_models(ranked, X_tune_enc, y_tune, X_holdout_enc, y_holdout, weights, mono, cfg["tuning"], rs)
    best_name = max(tuned, key=lambda name: tuned[name]["holdout_metrics"]["pr_auc"])
    best = tuned[best_name]
    print(f"\nBest model: {best_name}")

    # 3. Final fit on 85%, calibrate on the remaining 15%
    print("\n=== Final Training & Calibration ===")
    X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.15, stratify=y, random_state=rs)
    final_model = create_classifier(best_name, best["best_params"], mono)
    final_encoder, calibrator = fit_calibrated(final_model, X_train, y_train, X_cal, y_cal, cfg)
    plot_feature_importance(
        final_model, list(X.columns), f"{best_name} — Feature Importance",
        charts_dir / "13_feature_importance.png",
    )

    # 4. Export + track
    pickle_path = PROJECT_ROOT / cfg["output"]["models_dir"] / cfg["output"]["pickle_name"]
    bundle = build_bundle(
        best_name, best["best_params"], final_encoder.encoder, calibrator, list(X.columns),
        best["holdout_metrics"]["f2_threshold"], best["holdout_metrics"], cfg["features"],
    )
    save_pipeline(bundle, pickle_path)
    print(f"Saved to {pickle_path} ({pickle_path.stat().st_size / 1024**2:.1f} MB)")

    log_run(f"train_{best_name}", best["best_params"], best["holdout_metrics"],
            [pickle_path, *charts_dir.glob("*.png")])
    print("Done.")


if __name__ == "__main__":
    run()
