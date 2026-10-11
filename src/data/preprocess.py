import numpy as np
import pandas as pd


def preprocess(
    df: pd.DataFrame,
    cols_to_drop: list[str],
    skewed_cols: list[str],
    cat_cols: list[str],
    drop_from_features: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    """Prune, engineer, transform. Returns (X, y)."""
    df = df.copy()

    # Drop low-value columns
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Interaction features
    df["ltv_x_bureau"] = df["ltv"] * df["bureau_score"]
    df["installment_burden"] = df["total_installment_amount"] / (df["disbursed_amount"] + 1)
    df["overdue_per_account"] = df["total_overdue_accounts"] / (df["total_account_count"] + 1)
    df["balance_to_sanctioned"] = df["total_current_balance"] / (df["total_sanctioned_amount"] + 1)
    df["inquiry_per_month"] = df["inquiry_count"] / (df["credit_history_months"] + 1)
    df["age_x_credit"] = df["age_at_disbursal"] * df["credit_history_months"]
    df["new_acct_velocity"] = df["new_accounts_6m"] / (df["total_account_count"] + 1)

    # Split target before transforms
    y = df["loan_default"]
    X = df.drop(columns=drop_from_features, errors="ignore")

    # Format high-cardinality categoricals as strings
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna(-1).astype(int).astype(str)

    # Log-transform skewed currency columns
    for c in skewed_cols:
        if c in X.columns:
            X[c] = np.log1p(np.clip(X[c], 0, None))

    return X, y
