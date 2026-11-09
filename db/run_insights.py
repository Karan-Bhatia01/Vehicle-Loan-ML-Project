"""Run every query in db/insights.sql and print the results as tables."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from db.load_to_postgres import get_engine

SQL_FILE = Path(__file__).with_name("insights.sql")


def read_queries() -> dict[str, tuple[str, str]]:
    """{name: (question, sql)} - the comment lines after '-- name:' are the question."""
    parts = re.split(r"^-- name: (\w+)\s*$", SQL_FILE.read_text(), flags=re.MULTILINE)
    queries = {}
    for name, body in zip(parts[1::2], parts[2::2]):
        lines = body.strip().splitlines()
        question = " ".join(l.lstrip("- ").strip() for l in lines if l.startswith("--"))
        queries[name] = (question, "\n".join(l for l in lines if not l.startswith("--")))
    return queries


def main() -> None:
    engine = get_engine()
    for name, (question, sql) in read_queries().items():
        print(f"\n=== {name} ===\n{question}")
        print(pd.read_sql(sql, engine).to_string(index=False))


if __name__ == "__main__":
    main()
