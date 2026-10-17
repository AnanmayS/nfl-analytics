"""DuckDB connection helper. All transforms run through this database."""

from contextlib import contextmanager

import duckdb

from src.config import DB_PATH


@contextmanager
def connect(read_only: bool = False):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH), read_only=read_only)
    try:
        yield con
    finally:
        con.close()
