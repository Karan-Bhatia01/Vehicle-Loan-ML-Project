"""Full training pipeline: load → preprocess → OOF → tune → calibrate → export."""
import pickle
import sys
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.load import load_data
from src.data.preprocess import preprocess
from src.features.encoder import TargetEncoderWrapper
from src.training.weights import compute_normalized_weights
from src.training.train import train_oof, calibrate_model
from src.training.tune import tune_models
from src.evaluation.metrics import compute_all_metrics
from src.evaluation.plots import plot_reliability_diagram, plot_feature_importance


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def get_base_models(mono: dict) -> dict:
    return {
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            monotone_constraints=mono, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, eval_metric="auc", random_state=42,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1, verbose=-1, random_state=42,
        ),
        "CatBoost": CatBoostClassifier(
            iterations=300, depth=5, learning_rate=0.05,
            eval_metric="AUC", random_seed=42, verbose=0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=50, n_jobs=-1, random_state=42,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=200, max_depth=15, min_samples_leaf=50, n_jobs=-1, random_state=42,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42,
        ),
    }


def run():
    cfg = load_config()
    rs = cfg["training"]["random_state"]

    # Optional: MLflow logging
    try:
        import mlflow
        from mlflow_config import setup_mlflow
        setup_mlflow(PROJECT_ROOT / "config" / "config.yaml")
        use_mlflow = True
    except ImportError:
        use_mlflow = False

    # 1. Load & preprocess
    print("Loading data...")
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
    print(f"Features: {X.shape[1]} | Rows: {len(X)} | Default rate: {y.mean():.2%}")

    mono = cfg["features"]["monotonic_constraints"]
    models = get_base_models(mono)

    # 2. OOF Training
    print("\n=== OOF Training ===")
    oof_preds = train_oof(
        models, X, y,
        cat_cols=cfg["features"]["cat_cols"],
        encoder_cfg=cfg["encoder"],
        weight_cfg=cfg["weights"],
        n_folds=cfg["training"]["cv_folds"],
        cal_split=cfg["training"]["calibration_split"],
        random_state=rs,
    )

    # 3. Evaluate & rank
    print("\n=== Base Model Results ===")
    scores = {}
    for name, preds in oof_preds.items():
        m = compute_all_metrics(y, preds)
        scores[name] = m
        print(f"{name:<18} PR: {m['pr_auc']:.4f} | F2: {m['f2']:.4f} | Brier: {m['brier']:.4f}")

    top3 = sorted(scores, key=lambda k: scores[k]["pr_auc"], reverse=True)[:3]
    print(f"\nTop 3: {top3}")

    # 4. Calibration plot
    charts_dir = PROJECT_ROOT / cfg["output"]["charts_dir"]
    plot_reliability_diagram(
        y, {n: oof_preds[n] for n in top3},
        "Base Models Reliability Diagram",
        charts_dir / "12a_base_calibration.png",
    )

    # 5. Tune top 3
    print("\n=== Hyperparameter Tuning ===")
    X_tune, X_holdout, y_tune, y_holdout = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=rs,
    )
    te_tune = TargetEncoderWrapper(cfg["features"]["cat_cols"], **cfg["encoder"])
    X_tune_enc = te_tune.fit_transform(X_tune, y_tune)
    X_holdout_enc = te_tune.transform(X_holdout)

    scale_pos = (len(y_tune) - sum(y_tune)) / sum(y_tune)
    sw_tune = compute_normalized_weights(X_tune_enc, y_tune, scale_pos, **cfg["weights"])

    tuned = tune_models(
        top3, X_tune_enc, y_tune, X_holdout_enc, y_holdout, sw_tune, mono,
        n_iter=cfg["tuning"]["n_iter"],
        cv_folds=cfg["tuning"]["cv_folds"],
        random_state=rs,
    )

    best_name = max(tuned, key=lambda k: tuned[k]["holdout_metrics"]["pr_auc"])
    print(f"\nBest model: {best_name}")

    # 6. Final retrain + calibrate
    print("\n=== Final Pipeline ===")
    X_ft, X_fc, y_ft, y_fc = train_test_split(X, y, test_size=0.15, stratify=y, random_state=rs)

    te_final = TargetEncoderWrapper(cfg["features"]["cat_cols"], **cfg["encoder"])
    X_ft_enc = te_final.fit_transform(X_ft, y_ft)
    X_fc_enc = te_final.transform(X_fc)

    sw_final = compute_normalized_weights(
        X_ft_enc, y_ft, (len(y) - sum(y)) / sum(y), **cfg["weights"],
    )

    best_params = tuned[best_name]["best_params"]
    if best_name == "XGBoost":
        final_model = XGBClassifier(**best_params, monotone_constraints=mono, n_jobs=-1, eval_metric="auc", random_state=42)
    elif best_name == "LightGBM":
        final_model = LGBMClassifier(**best_params, n_jobs=-1, verbose=-1, random_state=42)
    elif best_name == "CatBoost":
        final_model = CatBoostClassifier(**best_params, eval_metric="AUC", random_seed=42, verbose=0)

    final_model.fit(X_ft_enc, y_ft, sample_weight=sw_final)
    calibrator = calibrate_model(final_model, X_fc_enc, y_fc)

    # Feature importance
    plot_feature_importance(
        final_model, list(X_ft_enc.columns),
        f"{best_name} — Feature Importance",
        charts_dir / "13_feature_importance.png",
    )

    # 7. Export
    models_dir = PROJECT_ROOT / cfg["output"]["models_dir"]
    models_dir.mkdir(parents=True, exist_ok=True)
    pickle_path = models_dir / cfg["output"]["pickle_name"]

    pipeline = {
        "model_name": best_name,
        "best_params": best_params,
        "target_encoder": te_final.encoder,
        "calibrated_model": calibrator,
        "cat_cols": cfg["features"]["cat_cols"],
        "skewed_cols": cfg["features"]["skewed_cols"],
        "feature_columns": list(X_ft_enc.columns),
        "optimal_f2_threshold": tuned[best_name]["holdout_metrics"]["f2_threshold"],
        "metrics": tuned[best_name]["holdout_metrics"],
    }

    with open(pickle_path, "wb") as f:
        pickle.dump(pipeline, f)
    print(f"Saved to {pickle_path} ({pickle_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # 8. Log to MLflow
    if use_mlflow:
        with mlflow.start_run(run_name=f"train_{best_name}"):
            mlflow.log_params(best_params)
            mlflow.log_metrics(tuned[best_name]["holdout_metrics"])
            mlflow.log_artifact(str(pickle_path))
            for chart in charts_dir.glob("*.png"):
                mlflow.log_artifact(str(chart))
        print("Logged to MLflow.")

    print("Done.")


if __name__ == "__main__":
    run()
