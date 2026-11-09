from pathlib import Path

import pandas as pd

from src.data.clean import clean_raw


def load_data(raw_path: Path, verbose: bool = True) -> pd.DataFrame:
    """Read a raw train.csv-style file and return the cleaned, model-ready table."""
    df = clean_raw(pd.read_csv(raw_path), verbose=verbose)
    return pd.get_dummies(df, columns=["employment_type"], drop_first=True, dtype=int)
