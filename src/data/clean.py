import re

import numpy as np
import pandas as pd

RAW_TO_CLEAN = {
    "UniqueID": "loan_id",
    "Current_pincode_ID": "pincode_id",
    "Employment.Type": "employment_type",
    "State_ID": "state_id",
    "Employee_code_ID": "employee_id",
    "Aadhar_flag": "has_aadhar",
    "PAN_flag": "has_pan",
    "VoterID_flag": "has_voter_id",
    "Driving_flag": "has_driving_license",
    "Passport_flag": "has_passport",
    "PERFORM_CNS.SCORE": "bureau_score",
    "NEW.ACCTS.IN.LAST.SIX.MONTHS": "new_accounts_6m",
    "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS": "delinquent_accounts_6m",
    "AVERAGE.ACCT.AGE": "avg_account_age_months",
    "CREDIT.HISTORY.LENGTH": "credit_history_months",
    "NO.OF_INQUIRIES": "inquiry_count",
}

# Primary + secondary bureau accounts are summed into one "total_*" column.
PRIMARY_SECONDARY_TOTALS = {
    "total_account_count": ("PRI.NO.OF.ACCTS", "SEC.NO.OF.ACCTS"),
    "total_active_accounts": ("PRI.ACTIVE.ACCTS", "SEC.ACTIVE.ACCTS"),
    "total_overdue_accounts": ("PRI.OVERDUE.ACCTS", "SEC.OVERDUE.ACCTS"),
    "total_current_balance": ("PRI.CURRENT.BALANCE", "SEC.CURRENT.BALANCE"),
    "total_sanctioned_amount": ("PRI.SANCTIONED.AMOUNT", "SEC.SANCTIONED.AMOUNT"),
    "total_installment_amount": ("PRIMARY.INSTAL.AMT", "SEC.INSTAL.AMT"),
}

# Primary-only bureau columns that are kept (renamed) for reference.
PRIMARY_ONLY = {
    "PRI.NO.OF.ACCTS": "pri_account_count",
    "PRI.ACTIVE.ACCTS": "pri_active_accounts",
    "PRI.OVERDUE.ACCTS": "pri_overdue_accounts",
    "PRI.CURRENT.BALANCE": "pri_current_balance",
    "PRI.SANCTIONED.AMOUNT": "pri_sanctioned_amount",
    "PRI.DISBURSED.AMOUNT": "pri_disbursed_amount",
    "PRIMARY.INSTAL.AMT": "pri_installment_amount",
}

DATE_FORMAT = "%d-%m-%y"
LAST_VALID_BIRTH_YEAR = 2026
YOUNG_BORROWER_AGE = 25
MIN_BUREAU_SCORE_WITH_HISTORY = 10


def duration_to_months(text) -> int:
    """'2yrs 3mon' -> 27. Anything unparseable counts as 0."""
    match = re.search(r"(\d+)yrs (\d+)mon", str(text))
    return int(match.group(1)) * 12 + int(match.group(2)) if match else 0


def _parse_birth_dates(dob: pd.Series) -> pd.Series:
    """Two-digit years like '64' parse as 2064; shift those back a century."""
    dob = pd.to_datetime(dob, format=DATE_FORMAT)
    return dob.where(dob.dt.year <= LAST_VALID_BIRTH_YEAR, dob - pd.DateOffset(years=100))


def clean_raw(raw: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Raw train.csv-style rows -> one clean, snake_case table with all derived columns.

    Also works on rows without UniqueID / loan_default (single applications at inference time).
    """
    df = raw.copy()

    if "UniqueID" in df.columns:
        before = len(df)
        df = df.drop_duplicates(subset="UniqueID")
        if verbose:
            print(f"Duplicates removed: {before - len(df)}")

    missing_employment = df["Employment.Type"].isna().sum()
    df["Employment.Type"] = df["Employment.Type"].fillna("Salaried")
    if verbose:
        print(f"Missing employment type filled with 'Salaried': {missing_employment}")

    for col in ("AVERAGE.ACCT.AGE", "CREDIT.HISTORY.LENGTH"):
        df[col] = df[col].map(duration_to_months)

    disbursal = pd.to_datetime(df["DisbursalDate"], format=DATE_FORMAT, errors="coerce")
    df["Disbursal_Quarter"] = disbursal.dt.quarter
    df["Disbursal_Month"] = disbursal.dt.month
    df["Is_End_Of_Month"] = disbursal.dt.is_month_end.astype(int)
    df["age_at_disbursal"] = (disbursal - _parse_birth_dates(df["Date.of.Birth"])).dt.days // 365

    for total, (primary, secondary) in PRIMARY_SECONDARY_TOTALS.items():
        df[total] = df[primary] + df[secondary]

    df = df.rename(columns={**RAW_TO_CLEAN, **PRIMARY_ONLY})

    # "+ 1" in the denominators avoids division by zero for borrowers with no accounts.
    df["overdue_ratio"] = df["total_overdue_accounts"] / (df["total_account_count"] + 1)
    df["credit_utilization"] = df["total_current_balance"] / (df["total_sanctioned_amount"] + 1)
    df["active_ratio"] = df["total_active_accounts"] / (df["total_account_count"] + 1)
    df["downpayment"] = df["asset_cost"] - df["disbursed_amount"]
    df["downpayment_ratio"] = df["downpayment"] / df["asset_cost"]
    id_flags = ["has_aadhar", "has_pan", "has_voter_id", "has_driving_license", "has_passport"]
    df["total_ids_provided"] = df[id_flags].sum(axis=1)
    df["credit_hunger_delta"] = df["credit_history_months"] - df["avg_account_age_months"]
    df["credit_hunger_ratio"] = df["avg_account_age_months"] / (df["credit_history_months"] + 1)
    df["has_credit_history"] = (df["bureau_score"] > MIN_BUREAU_SCORE_WITH_HISTORY).astype(int)
    df["has_overdue_accounts"] = (df["total_overdue_accounts"] > 0).astype(int)
    df["recent_new_account"] = (df["new_accounts_6m"] > 0).astype(int)
    df["is_young_borrower"] = (df["age_at_disbursal"] < YOUNG_BORROWER_AGE).astype(int)

    drop = ["Date.of.Birth", "DisbursalDate", "MobileNo_Avl_Flag", "PERFORM_CNS.SCORE.DESCRIPTION"]
    drop += [c for pair in PRIMARY_SECONDARY_TOTALS.values() for c in pair if c in df.columns]
    drop += ["SEC.NO.OF.ACCTS", "SEC.ACTIVE.ACCTS", "SEC.OVERDUE.ACCTS", "SEC.CURRENT.BALANCE",
             "SEC.SANCTIONED.AMOUNT", "SEC.DISBURSED.AMOUNT", "SEC.INSTAL.AMT"]
    return df.drop(columns=drop, errors="ignore").replace([np.inf, -np.inf], np.nan)
