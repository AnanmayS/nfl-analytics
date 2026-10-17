"""Historical opening and closing lines from the Australia Sports Betting NFL file.

Download nfl.xlsx yourself from https://www.aussportsbetting.com/data/ (it sits behind a
bot check) and save it to data/external/nfl.xlsx. Its terms allow personal use only and
forbid republishing, so the file and everything derived from it stay under data/, which
is git-ignored.

Writes game_lines: one row per nflverse game_id with
  open_spread / book_close_spread   home team's expected margin (nflverse convention:
                                    positive = home favored), at open and at close
  p_open_ml / p_close_ml            de-vigged moneyline home win probability
  open_total / close_total
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.config import DATA
from src.db import connect

log = logging.getLogger(__name__)
XLSX = DATA / "external" / "nfl.xlsx"

# Names in the file that differ from today's nflverse team names
OLD_NAMES = {
    "Oakland Raiders": "LV", "San Diego Chargers": "LAC", "St. Louis Rams": "LA",
    "Washington Redskins": "WAS", "Washington Football Team": "WAS",
}


def read_lines(path: Path) -> pl.DataFrame:
    d = pl.read_excel(path, sheet_name="Data")

    def devig(home: str, away: str) -> pl.Expr:  # decimal odds -> home win probability
        ih, ia = 1 / pl.col(home), 1 / pl.col(away)
        return ih / (ih + ia)

    return d.select(
        pl.col("Date").cast(pl.Date).alias("date"),
        pl.col("Home Team").alias("home_name"), pl.col("Away Team").alias("away_name"),
        (-pl.col("Home Line Open")).alias("open_spread"),
        (-pl.col("Home Line Close")).alias("book_close_spread"),
        devig("Home Odds Open", "Away Odds Open").alias("p_open_ml"),
        devig("Home Odds Close", "Away Odds Close").alias("p_close_ml"),
        pl.col("Total Score Open").alias("open_total"),
        pl.col("Total Score Close").alias("close_total"),
    )


def match_games(lines: pl.DataFrame, sched: pl.DataFrame, teams: pl.DataFrame) -> pl.DataFrame:
    """Attach nflverse game_ids by team pair and date (±1 day for time zones). Neutral-site
    games can list either team as home, so a swapped pair is accepted with the line flipped."""
    # nflverse lists some teams under two codes (e.g. LA and LAR); keep the schedule's.
    used = set(sched["home_team"]) | set(sched["away_team"])
    t = teams.filter(pl.col("team_abbr").is_in(used))
    name_to_abbr = dict(zip(t["team_name"], t["team_abbr"], strict=True)) | OLD_NAMES
    ln = lines.with_columns(
        home_team=pl.col("home_name").replace_strict(name_to_abbr, default=None),
        away_team=pl.col("away_name").replace_strict(name_to_abbr, default=None),
    )
    s = sched.select("game_id", "game_date", "home_team", "away_team")
    vals = ["open_spread", "book_close_spread", "p_open_ml", "p_close_ml"]
    out = []
    for swapped in (False, True):
        left = ln if not swapped else ln.rename(
            {"home_team": "away_team", "away_team": "home_team"}).with_columns(
            *[(-pl.col(c)).alias(c) for c in vals[:2]],
            *[(1 - pl.col(c)).alias(c) for c in vals[2:]])
        for shift in (0, -1, 1):
            j = left.with_columns(game_date=pl.col("date") + pl.duration(days=shift)).join(
                s, on=["game_date", "home_team", "away_team"], how="inner")
            out.append(j.select("game_id", *vals, "open_total", "close_total").with_columns(
                swapped=pl.lit(swapped), shift=pl.lit(abs(shift))))
    m = pl.concat(out).sort("swapped", "shift").unique("game_id", keep="first")
    return m.select("game_id", *vals, "open_total", "close_total")


def run(path: Path = XLSX) -> pl.DataFrame | None:
    if not path.exists():
        log.warning("no opening-line file at %s; see src/ingest/odds.py", path)
        return None
    lines = read_lines(path)
    with connect() as con:
        from src.features.team import TEAM_FIX
        sched = con.execute("""SELECT game_id, CAST(gameday AS DATE) AS game_date,
                                      home_team, away_team FROM raw_schedules""").pl().with_columns(
            pl.col("home_team", "away_team").replace(TEAM_FIX))
        teams = con.execute("SELECT team_abbr, team_name FROM raw_teams").pl()
        m = match_games(lines, sched, teams)
        con.register("_gl", m.to_arrow())
        con.execute("CREATE OR REPLACE TABLE game_lines AS SELECT * FROM _gl")
        con.unregister("_gl")
    log.info("game_lines: matched %d of %d rows in %s", m.height, lines.height, path.name)
    return m
