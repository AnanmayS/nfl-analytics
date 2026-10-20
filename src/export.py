"""Slim copy of the database with only what the dashboard reads, for the hosted app.

The full database is ~800 MB, mostly raw play-by-play; the dashboard needs a few thousand
rows of derived tables. Anything derived from the opening-lines file (src/ingest/odds.py,
personal use only, no republishing) is left out, so the copy is safe to publish.
"""

from __future__ import annotations

import logging

import duckdb

from src.archetypes.pipeline import SPECS
from src.config import DATA, DB_PATH

log = logging.getLogger(__name__)
APP_DB = DATA / "app.duckdb"

TABLES = [
    "game_predictions", "raw_teams", "refresh_info", "team_profile", "team_key_players",
    "game_metrics", "game_calibration", "game_vs_vegas", "game_model_lift",
    "archetype_mix_tests", "qb_wr_pairings",
    *[f"archetype_{p.lower()}" for p in SPECS],
    *[f"archetype_centroids_{p.lower()}" for p in SPECS],
    *[f"archetype_selection_{p.lower()}" for p in SPECS],
]
# Columns built from the opening-lines file (Tuesday picks and the lines themselves)
ODDS_COLUMNS = {"open_spread", "p_open", "p_tue_stats", "p_tuesday"}


def run() -> None:
    tmp = APP_DB.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    con = duckdb.connect(str(tmp))
    try:
        con.execute(f"ATTACH '{DB_PATH}' AS src (READ_ONLY)")
        have = {r[0] for r in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'src'"
        ).fetchall()}
        for t in TABLES:
            if t not in have:
                log.warning("skipping %s: not in %s", t, DB_PATH.name)
                continue
            cols = [r[0] for r in con.execute(f"DESCRIBE src.{t}").fetchall()]
            keep = ", ".join(f'"{c}"' for c in cols if c not in ODDS_COLUMNS)
            con.execute(f"CREATE TABLE {t} AS SELECT {keep} FROM src.{t}")
        con.execute("DETACH src")
    finally:
        con.close()
    tmp.replace(APP_DB)
    log.info("wrote %s (%.1f MB)", APP_DB, APP_DB.stat().st_size / 1e6)
