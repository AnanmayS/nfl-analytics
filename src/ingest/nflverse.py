"""Pull nflverse datasets via nflreadpy, cache per-season parquet, load into DuckDB.

Layout: data/raw/<dataset>/<season>.parquet. Completed seasons are cached forever;
the current season is re-pulled on every run (or any season with force=True).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import nflreadpy as nfl
import polars as pl

from src.config import FIRST_SEASON, RAW
from src.db import connect

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Dataset:
    name: str
    loader: Callable[[int], pl.DataFrame]
    first_season: int = FIRST_SEASON
    seasonal: bool = True  # False: one file (data/raw/<name>/all.parquet), refreshed each run


DATASETS: list[Dataset] = [
    Dataset("pbp", lambda s: nfl.load_pbp(s)),
    Dataset("player_stats", lambda s: nfl.load_player_stats(s, summary_level="week")),
    Dataset("rosters", lambda s: nfl.load_rosters(s)),
    Dataset("snap_counts", lambda s: nfl.load_snap_counts(s)),
    Dataset("participation", lambda s: nfl.load_participation(s)),
    Dataset("ngs_passing", lambda s: nfl.load_nextgen_stats(s, "passing")),
    Dataset("ngs_receiving", lambda s: nfl.load_nextgen_stats(s, "receiving")),
    Dataset("ngs_rushing", lambda s: nfl.load_nextgen_stats(s, "rushing")),
    Dataset("schedules", lambda s: nfl.load_schedules(s)),
    Dataset("ftn_charting", lambda s: nfl.load_ftn_charting(s), first_season=2022),
    Dataset("injuries", lambda s: nfl.load_injuries(s)),  # weekly official injury reports
    Dataset("rosters_weekly", lambda s: nfl.load_rosters_weekly(s)),  # incl. injured reserve
    Dataset("players", lambda _: nfl.load_players(), seasonal=False),  # id crosswalk
    Dataset("teams", lambda _: nfl.load_teams(), seasonal=False),  # names, colors, logos
]


def seasons(last: int | None = None) -> list[int]:
    return list(range(FIRST_SEASON, (last or nfl.get_current_season()) + 1))


def pull(ds: Dataset, season: int, force: bool = False) -> bool:
    """Fetch one dataset-season to parquet. Returns True if a file exists afterwards."""
    path = RAW / ds.name / (f"{season}.parquet" if ds.seasonal else "all.parquet")
    current = season == nfl.get_current_season() or not ds.seasonal
    if path.exists() and not (force or current):
        return True
    try:
        df = ds.loader(season)
    except Exception as e:  # nflverse returns errors for seasons not yet published
        log.warning("skip %s %s: %s", ds.name, season, str(e)[:120])
        return path.exists()
    if df.height == 0:
        log.warning("skip %s %s: empty", ds.name, season)
        return path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Normalise null-typed columns so per-season files union cleanly.
    df = df.with_columns(pl.col(c).cast(pl.String) for c, t in df.schema.items() if t == pl.Null)
    df.write_parquet(path)
    log.info("wrote %s (%d rows)", path.relative_to(RAW.parent.parent), df.height)
    return True


def load_duckdb(names: list[str] | None = None) -> None:
    """(Re)create raw_<dataset> tables from cached parquet."""
    with connect() as con:
        for ds in DATASETS:
            if names and ds.name not in names:
                continue
            files = sorted((RAW / ds.name).glob("*.parquet"))
            if not files:
                continue
            glob = str(RAW / ds.name / "*.parquet")
            con.execute(
                f"CREATE OR REPLACE TABLE raw_{ds.name} AS "
                f"SELECT * FROM read_parquet('{glob}', union_by_name=true)"
            )
            n = con.execute(f"SELECT count(*) FROM raw_{ds.name}").fetchone()[0]
            log.info("raw_%s: %d rows from %d files", ds.name, n, len(files))


def run(names: list[str] | None = None, last_season: int | None = None, force: bool = False):
    for ds in DATASETS:
        if names and ds.name not in names:
            continue
        if not ds.seasonal:
            pull(ds, 0, force=force)
            continue
        for s in seasons(last_season):
            if s >= ds.first_season:
                pull(ds, s, force=force)
    load_duckdb(names)
