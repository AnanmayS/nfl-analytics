"""Project-wide paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
OUT = DATA / "out"
DB_PATH = DATA / "nfl.duckdb"

FIRST_SEASON = 2016
