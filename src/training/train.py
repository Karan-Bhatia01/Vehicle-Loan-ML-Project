import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split

from src.features.encoder import TargetEncoderWrapper
from src.training.weights import compute_normalized_weights
from src.evaluation.metrics import compute_all_metrics

# Handle sklearn version differences
try:
    from sklearn.calibration import FrozenEstimator
    _USE_FROZEN = True
except ImportError:
    _USE_FROZEN = False


def calibrate_model(fitted_model, X_cal: pd.DataFrame, y_cal: pd.Series):
    """Wrap a fitted model with isotonic calibration."""
    if _USE_FROZEN:
        calibrator = CalibratedClassifierCV(FrozenEstimator(fitted_model), method="isotonic")
    else:
        calibrator = CalibratedClassifierCV(fitted_model, cv="prefit", method="isotonic")
    calibrator.fit(X_cal, y_cal)
    return calibrator


def train_oof(
    models: dict,
    X: pd.DataFrame,
    y: pd.Series,
    cat_cols: list[str],
    encoder_cfg: dict,
    weight_cfg: dict,
    n_folds: int = 5,
    cal_split: float = 0.20,
    random_state: int = 42,
) -> dict[str, np.ndarray]:
    """Run OOF training with nested calibration. Returns dict of OOF predictions per model."""
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    oof_preds = {name: np.zeros(len(X)) for name in models}
    scale_pos_weight = (len(y) - sum(y)) / sum(y)

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

        X_train, y_train = X.iloc[train_idx].copy(), y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx].copy(), y.iloc[val_idx]

        # Nested split for calibration
        X_t, X_cal, y_t, y_cal = train_test_split(
            X_train, y_train, test_size=cal_split, stratify=y_train, random_state=random_state,
        )

        # Encode (fit on training subset only)
        te = TargetEncoderWrapper(cat_cols, **encoder_cfg)
        X_t = te.fit_transform(X_t, y_t)
        X_cal = te.transform(X_cal)
        X_val_enc = te.transform(X_val)

        # Sample weights
        weights = compute_normalized_weights(X_t, y_t, scale_pos_weight, **weight_cfg)

        for name, template in models.items():
            model = clone(template)
            try:
                model.fit(X_t, y_t, sample_weight=weights)
                calibrator = calibrate_model(model, X_cal, y_cal)
                preds = calibrator.predict_proba(X_val_enc)[:, 1]
                oof_preds[name][val_idx] = preds

                m = compute_all_metrics(y_val, preds)
                print(f"  {name:<18} PR: {m['pr_auc']:.3f} | F2: {m['f2']:.3f} (@{m['f2_threshold']:.2f})")
            except Exception as e:
                oof_preds[name][val_idx] = 0.5
                print(f"  {name} failed: {str(e)[:60]}")

    return oof_preds
