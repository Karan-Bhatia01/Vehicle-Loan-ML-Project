import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, FrozenEstimator
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBClassifier

from src.evaluation.metrics import compute_all_metrics
from src.features.encoder import TargetEncoderWrapper
from src.training.weights import compute_normalized_weights


def create_classifier(name: str, params: dict, mono_constraints: dict | None = None, n_jobs: int = -1):
    """Build an untrained classifier. Monotonic constraints only apply to XGBoost."""
    if name == "XGBoost":
        return XGBClassifier(
            **params, monotone_constraints=mono_constraints,
            n_jobs=n_jobs, eval_metric="auc", random_state=42,
        )
    if name == "LightGBM":
        return LGBMClassifier(**params, n_jobs=n_jobs, verbose=-1, random_state=42)
    if name == "CatBoost":
        return CatBoostClassifier(
            **params, eval_metric="AUC", random_seed=42, verbose=0,
            thread_count=n_jobs, allow_writing_files=False,
        )
    raise ValueError(f"Unsupported model: {name}")


def calibrate_model(fitted_model, X_cal: pd.DataFrame, y_cal: pd.Series) -> CalibratedClassifierCV:
    """Fit isotonic calibration on top of an already-trained model."""
    calibrator = CalibratedClassifierCV(FrozenEstimator(fitted_model), method="isotonic")
    return calibrator.fit(X_cal, y_cal)


def fit_calibrated(model, X_train, y_train, X_cal, y_cal, cfg: dict) -> tuple[TargetEncoderWrapper, CalibratedClassifierCV]:
    """Encode -> weight -> fit -> calibrate. Encoder and model only ever see the training split."""
    encoder = TargetEncoderWrapper(cfg["features"]["cat_cols"], **cfg["encoder"])
    X_train_enc = encoder.fit_transform(X_train, y_train)
    X_cal_enc = encoder.transform(X_cal)

    scale_pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()
    weights = compute_normalized_weights(X_train_enc, y_train, scale_pos_weight, **cfg["weights"])

    model.fit(X_train_enc, y_train, sample_weight=weights)
    return encoder, calibrate_model(model, X_cal_enc, y_cal)


def train_oof(models: dict, X: pd.DataFrame, y: pd.Series, cfg: dict) -> dict[str, np.ndarray]:
    """Stratified K-fold with a nested calibration split per fold. Returns out-of-fold predictions per model."""
    train_cfg = cfg["training"]
    rs = train_cfg["random_state"]
    cv = StratifiedKFold(n_splits=train_cfg["cv_folds"], shuffle=True, random_state=rs)
    oof_preds = {name: np.zeros(len(X)) for name in models}

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\n--- Fold {fold}/{train_cfg['cv_folds']} ---")
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        X_t, X_cal, y_t, y_cal = train_test_split(
            X_train, y_train, test_size=train_cfg["calibration_split"], stratify=y_train, random_state=rs,
        )

        for name, template in models.items():
            encoder, calibrator = fit_calibrated(clone(template), X_t, y_t, X_cal, y_cal, cfg)
            preds = calibrator.predict_proba(encoder.transform(X_val))[:, 1]
            oof_preds[name][val_idx] = preds

            m = compute_all_metrics(y_val, preds)
            print(f"  {name:<10} PR: {m['pr_auc']:.3f} | F2: {m['f2']:.3f} (@{m['f2_threshold']:.2f})")

    return oof_preds
