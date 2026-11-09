"""Clean the raw loan data with Pandas and load the result into the `loans_clean` table.

Usage (PowerShell):
    $env:DB_URL = "postgresql+psycopg2://postgres:<password>@localhost:5432/loan_db"
    python db/load_to_postgres.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sqlalchemy import create_engine

from src.config import PROJECT_ROOT, load_config
from src.data.clean import clean_raw

TABLE_NAME = "loans_clean"
CHUNK_SIZE = 10_000


def get_engine():
    url = os.environ.get("DB_URL")
    if not url:
        sys.exit("DB_URL is not set. Example: postgresql+psycopg2://postgres:<password>@localhost:5432/loan_db")
    return create_engine(url)


def main() -> None:
    raw_path = PROJECT_ROOT / load_config()["data"]["raw_path"]
    df = clean_raw(pd.read_csv(raw_path), verbose=True)

    # chunksize writes in batches of 10k rows instead of one giant INSERT, keeping memory flat.
    df.to_sql(TABLE_NAME, get_engine(), if_exists="replace", index=False, chunksize=CHUNK_SIZE)
    print(f"Rows loaded into {TABLE_NAME}: {len(df):,}")


if __name__ == "__main__":
    main()
