import pandas as pd
from pathlib import Path

def load_data(raw_path: Path, processed_dir: Path) -> pd.DataFrame:
    """Load and merge all processed CSVs, add temporal features from raw data."""
    df_loan = pd.read_csv(processed_dir / "loan_details.csv")
    df_customer = pd.read_csv(processed_dir / "customer_profile.csv")
    df_bureau = pd.read_csv(processed_dir / "bureau_history.csv")
    df_risk = pd.read_csv(processed_dir / "engineered_risk.csv")

    df = (
        df_loan
        .merge(df_customer, on="customer_id")
        .merge(df_bureau, on="customer_id")
        .merge(df_risk, on="customer_id")
    )

    # Temporal features from raw disbursal date
    df_raw = pd.read_csv(raw_path)
    disbursal = pd.to_datetime(df_raw["DisbursalDate"], format="%d-%m-%y", errors="coerce")
    df["Disbursal_Quarter"] = disbursal.dt.quarter.values
    df["Disbursal_Month"] = disbursal.dt.month.values
    df["Is_End_Of_Month"] = disbursal.dt.is_month_end.astype(int).values

    # One-hot encode employment type
    df = pd.get_dummies(df, columns=["employment_type"], drop_first=True, dtype=int)

    return df
