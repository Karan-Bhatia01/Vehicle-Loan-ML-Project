import numpy as np
import pandas as pd


def build_features(X: pd.DataFrame, cat_cols: list[str], skewed_cols: list[str]) -> pd.DataFrame:
    """Feature steps shared by training and inference: ratios, categorical IDs as strings, log amounts."""
    X = X.copy()
    # Missing raw columns (e.g. partial API input) count as 0.
    col = lambda name: X.get(name, 0)

    # Ratios use raw amounts, so they must be computed before the log transform below.
    X["ltv_x_bureau"] = col("ltv") * col("bureau_score")
    X["installment_burden"] = col("total_installment_amount") / (col("disbursed_amount") + 1)
    X["overdue_per_account"] = col("total_overdue_accounts") / (col("total_account_count") + 1)
    X["balance_to_sanctioned"] = col("total_current_balance") / (col("total_sanctioned_amount") + 1)
    X["inquiry_per_month"] = col("inquiry_count") / (col("credit_history_months") + 1)
    X["age_x_credit"] = col("age_at_disbursal") * col("credit_history_months")
    X["new_acct_velocity"] = col("new_accounts_6m") / (col("total_account_count") + 1)

    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna(-1).astype(int).astype(str)

    for c in skewed_cols:
        if c in X.columns:
            X[c] = np.log1p(np.clip(X[c], 0, None))

    return X


def preprocess(df: pd.DataFrame, features_cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """Turn the merged table into (X, y) using the `features` section of config.yaml."""
    y = df[features_cfg["target_col"]]
    X = df.drop(columns=features_cfg["cols_to_drop"] + features_cfg["drop_from_features"], errors="ignore")
    X = build_features(X, features_cfg["cat_cols"], features_cfg["skewed_cols"])
    return X, y
