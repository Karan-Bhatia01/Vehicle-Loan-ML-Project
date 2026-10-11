import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
)


def sweep_f2(y_true: np.ndarray, y_proba: np.ndarray) -> tuple[float, float]:
    """Find the threshold that maximizes F2 score. Returns (best_f2, best_threshold)."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    denom = 4 * precision + recall
    f2 = np.divide(
        5 * precision * recall, denom,
        out=np.zeros_like(denom), where=denom != 0,
    )
    best_idx = np.argmax(f2)
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 1.0
    return float(f2[best_idx]), float(best_thresh)


def compute_all_metrics(y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    """Compute the full metric suite on a set of predictions."""
    f2, f2_thresh = sweep_f2(y_true, y_proba)
    return {
        "pr_auc": average_precision_score(y_true, y_proba),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "f2": f2,
        "f2_threshold": f2_thresh,
        "brier": brier_score_loss(y_true, y_proba),
    }
