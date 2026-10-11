import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def load_pipeline(path: Path) -> dict:
    """Load the exported pipeline pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def _engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    """Recreate the same interaction features used during training."""
    X = X.copy()
    X["ltv_x_bureau"] = X.get("ltv", 0) * X.get("bureau_score", 0)
    X["installment_burden"] = X.get("total_installment_amount", 0) / (X.get("disbursed_amount", 0) + 1)
    X["overdue_per_account"] = X.get("total_overdue_accounts", 0) / (X.get("total_account_count", 0) + 1)
    X["balance_to_sanctioned"] = X.get("total_current_balance", 0) / (X.get("total_sanctioned_amount", 0) + 1)
    X["inquiry_per_month"] = X.get("inquiry_count", 0) / (X.get("credit_history_months", 0) + 1)
    X["age_x_credit"] = X.get("age_at_disbursal", 0) * X.get("credit_history_months", 0)
    X["new_acct_velocity"] = X.get("new_accounts_6m", 0) / (X.get("total_account_count", 0) + 1)
    return X


def predict(pipeline: dict, raw_input: pd.DataFrame) -> dict:
    """Run inference on a raw loan application DataFrame.

    Returns dict with 'probability', 'decision', and 'threshold'.
    """
    X = raw_input.copy()

    # Format categoricals
    for c in pipeline["cat_cols"]:
        if c in X.columns:
            X[c] = X[c].fillna(-1).astype(int).astype(str)

    # Log-transform skewed columns
    for c in pipeline["skewed_cols"]:
        if c in X.columns:
            X[c] = np.log1p(np.clip(X[c], 0, None))

    # Create interaction features (must happen after log-transform of disbursed_amount)
    X = _engineer_features(X)

    # Align columns to what the encoder expects, fill missing with 0
    expected = pipeline["feature_columns"]
    X = X.reindex(columns=expected, fill_value=0)

    # Patch for category_encoders version mismatch (older pickle missing newer attributes)
    encoder = pipeline["target_encoder"]
    if not hasattr(encoder, "min_group_lumping_"):
        encoder.min_group_lumping_ = {}
    if not hasattr(encoder, "keep_components"):
        encoder.keep_components = False
    if not hasattr(encoder, "composite_members_"):
        encoder.composite_members_ = []
        
    if hasattr(encoder, "ordinal_encoder"):
        if not hasattr(encoder.ordinal_encoder, "min_group_lumping_"):
            encoder.ordinal_encoder.min_group_lumping_ = {}
        if not hasattr(encoder.ordinal_encoder, "index_start"):
            encoder.ordinal_encoder.index_start = 1

    # Target encode
    X = encoder.transform(X).fillna(0)

    # Final column alignment
    X = X.reindex(columns=expected, fill_value=0)

    # Predict
    proba = pipeline["calibrated_model"].predict_proba(X)[:, 1]
    threshold = pipeline["optimal_f2_threshold"]

    return {
        "probability": proba.tolist(),
        "decision": ["REJECT" if p > threshold else "APPROVE" for p in proba],
        "threshold": threshold,
    }
