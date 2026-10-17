"""DuckDB connection helper. All transforms run through this database."""

import time
from contextlib import contextmanager

import duckdb

from src.config import DB_PATH

LOCK_WAIT_SECONDS = 60


@contextmanager
def connect(read_only: bool = False):
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # The dashboard opens the file briefly for each read; wait those out instead of failing.
    deadline = time.monotonic() + LOCK_WAIT_SECONDS
    while True:
        try:
            con = duckdb.connect(str(DB_PATH), read_only=read_only)
            break
        except duckdb.IOException as e:
            if "lock" not in str(e).lower() or time.monotonic() > deadline:
                raise
            time.sleep(1)
    try:
        yield con
    finally:
        con.close()
