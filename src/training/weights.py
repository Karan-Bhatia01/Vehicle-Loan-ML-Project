import numpy as np
import pandas as pd


def compute_normalized_weights(
    X: pd.DataFrame,
    y: pd.Series,
    scale_pos_weight: float,
    clip_min: float = 0.5,
    clip_max: float = 3.0,
) -> np.ndarray:
    """Cost-sensitive sample weights: penalize missing large defaults more heavily.

    Returns weights normalized to mean=1.0 to prevent gradient drift.
    """
    mean_disbursed = np.expm1(X["disbursed_amount"]).mean()
    financial_penalty = np.clip(
        np.expm1(X["disbursed_amount"]) / mean_disbursed,
        clip_min, clip_max,
    )
    base_weight = np.where(y == 1, scale_pos_weight, 1.0)
    weights = base_weight * financial_penalty
    return weights / weights.mean()
