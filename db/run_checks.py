"""Run db/data_checks.sql against the database and compare each result with what it must be.

Exits with a non-zero code if any check fails.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sqlalchemy import text

from db.load_to_postgres import get_engine
from src.config import PROJECT_ROOT, load_config

SQL_FILE = Path(__file__).with_name("data_checks.sql")


def read_queries() -> dict[str, str]:
    """Split data_checks.sql into {check_name: query} using the '-- name:' markers."""
    parts = re.split(r"^-- name: (\w+)\s*$", SQL_FILE.read_text(), flags=re.MULTILINE)
    return {name: sql.strip() for name, sql in zip(parts[1::2], parts[2::2])}


def evaluate(results: dict[str, dict], expected_rows: int) -> list[tuple[str, bool, str]]:
    """(check, passed, detail) for every rule. `results` maps check name -> its single result row."""
    r = results
    nulls = {k: v for k, v in r["null_counts"].items() if v}
    rate = float(r["target_balance"]["default_rate"])
    ranges = r["value_ranges"]
    return [
        ("row count matches raw file", r["row_count"]["row_count"] == expected_rows,
         f"{r['row_count']['row_count']:,} rows (expected {expected_rows:,})"),
        ("no nulls in model columns", not nulls, f"nulls: {nulls or 0}"),
        ("no duplicate loan_id", r["duplicate_ids"]["duplicate_ids"] == 0,
         f"{r['duplicate_ids']['duplicate_ids']} duplicates"),
        ("LTV within 0-100", 0 <= ranges["min_ltv"] and ranges["max_ltv"] <= 100,
         f"{ranges['min_ltv']} - {ranges['max_ltv']}"),
        ("age within 18-75", 18 <= ranges["min_age"] and ranges["max_age"] <= 75,
         f"{ranges['min_age']} - {ranges['max_age']}"),
        ("bureau score within 0-900", 0 <= ranges["min_bureau_score"] and ranges["max_bureau_score"] <= 900,
         f"{ranges['min_bureau_score']} - {ranges['max_bureau_score']}"),
        ("disbursed amount never above asset cost", ranges["loans_above_asset_cost"] == 0,
         f"{ranges['loans_above_asset_cost']} loans"),
        ("target is only 0 or 1", r["target_balance"]["invalid_target_values"] == 0,
         f"{r['target_balance']['invalid_target_values']} invalid"),
        ("default rate between 20% and 23%", 0.20 <= rate <= 0.23, f"{rate:.2%}"),
    ]


def main() -> None:
    raw_path = PROJECT_ROOT / load_config()["data"]["raw_path"]
    expected_rows = pd.read_csv(raw_path, usecols=["UniqueID"])["UniqueID"].nunique()

    with get_engine().connect() as conn:
        results = {name: dict(conn.execute(text(sql)).mappings().one()) for name, sql in read_queries().items()}

    outcomes = evaluate(results, expected_rows)
    for check, passed, detail in outcomes:
        print(f"[{'PASS' if passed else 'FAIL'}] {check}: {detail}")

    failed = sum(not passed for _, passed, _ in outcomes)
    print(f"\n{len(outcomes) - failed}/{len(outcomes)} checks passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
