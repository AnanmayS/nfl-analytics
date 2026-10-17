"""Who is actually starting at QB, and how much better or worse that is than recent form.

The team's rolling stats reflect whoever has been playing QB lately (the "recent starter":
most offensive snaps over the last 3 games). The injury report alone misses QB changes:
players on injured reserve aren't listed, benchings and offseason moves never appear. So:

expected starter for a game, in order of preference
  1. the starting QB listed on the schedule (nflverse fills this for played games and
     usually a week or two ahead);
  2. otherwise, if the recent starter is unavailable (not active on that week's roster,
     e.g. injured reserve, or listed Out / Doubtful), the next QB by recent snaps;
  3. otherwise the recent starter.

QB rating = shrunken EPA per dropback from games strictly before kickoff: last seasons
decay by DECAY per season, and RATING_K dropbacks of replacement-level play (PRIOR) are
mixed in, so rookies and little-used backups rate near replacement.

Features (per team-game):
  qb_change   1 if the expected starter isn't the recent starter
  qb_delta    rating(expected) - rating(recent): the QB-quality swing vs. recent form
"""

from __future__ import annotations

import duckdb
import polars as pl

from src.features.team import TEAM_FIX

RATING_K = 300.0     # dropbacks of replacement-level play mixed into every rating
PRIOR = -0.10        # replacement-level EPA per dropback
DECAY = 0.6          # weight per season back
AVAILABLE = {"ACT"}  # weekly-roster statuses that mean "can play"

QB_GAMES_SQL = """
SELECT coalesce(passer_player_id, rusher_player_id) AS qb_id, game_id, season,
       CAST(min(game_date) AS DATE) AS game_date,
       sum(qb_epa) AS epa, count(*) AS dropbacks
FROM {pbp}
WHERE qb_dropback = 1 AND qb_epa IS NOT NULL
  AND coalesce(passer_player_id, rusher_player_id) IS NOT NULL
GROUP BY ALL
"""


def qb_states(qb_games: pl.DataFrame) -> pl.DataFrame:
    """Cumulative (decayed) EPA and dropbacks for each QB *after* each of his games.

    Within a season everything counts fully; each earlier season counts DECAY^(gap).
    """
    g = qb_games.sort("qb_id", "game_date").with_columns(
        c_epa=pl.col("epa").cum_sum().over("qb_id", "season"),
        c_db=pl.col("dropbacks").cum_sum().over("qb_id", "season"),
    )
    tot = g.group_by("qb_id", "season").agg(pl.col("epa").sum().alias("s_epa"),
                                            pl.col("dropbacks").sum().alias("s_db"))
    seasons = tot.select("qb_id", pl.col("season").alias("t"))
    prior = (seasons.join(tot, on="qb_id").filter(pl.col("season") < pl.col("t"))
             .with_columns(w=DECAY ** (pl.col("t") - pl.col("season")))
             .group_by("qb_id", "t").agg((pl.col("w") * pl.col("s_epa")).sum().alias("p_epa"),
                                        (pl.col("w") * pl.col("s_db")).sum().alias("p_db"))
             .rename({"t": "season"}))
    return g.join(prior, on=["qb_id", "season"], how="left").select(
        "qb_id", "season", "game_date",
        (pl.col("p_epa").fill_null(0) + pl.col("c_epa")).alias("e"),
        (pl.col("p_db").fill_null(0) + pl.col("c_db")).alias("d"),
    )


def rating_asof(lookups: pl.DataFrame, states: pl.DataFrame, qb_col: str) -> pl.Series:
    """Rating of the QB in `qb_col` for each row of `lookups` (needs game_date, season),
    using only games strictly before that date. Missing QB -> replacement level."""
    left = lookups.select(pl.int_range(pl.len()).alias("_row"),
                          pl.col(qb_col).cast(pl.String).alias("qb_id"),
                          "game_date", "season").sort("game_date")
    right = states.rename({"season": "s_season", "game_date": "s_date"}).sort("s_date")
    j = left.join_asof(right, left_on="game_date", right_on="s_date", by="qb_id",
                       strategy="backward", allow_exact_matches=False, check_sortedness=False)
    decay = pl.lit(DECAY) ** (pl.col("season") - pl.col("s_season")).fill_null(0)
    e = (pl.col("e") * decay).fill_null(0)
    d = (pl.col("d") * decay).fill_null(0)
    return j.with_columns(r=(e + RATING_K * PRIOR) / (d + RATING_K)).sort("_row")["r"]


def expected_starters(games: pl.DataFrame, roster_status: pl.DataFrame,
                      states: pl.DataFrame) -> pl.DataFrame:
    """Pick each team-game's expected starting QB and compute qb_change / qb_delta.

    games: game_id, team, season, week, game_date, listed_qb_id/name, recent_qb_id/name,
           backup_qb_id/name, recent_qb_sit (injury-report sit chance of the recent starter)
    roster_status: season, week, team, player_id, status (weekly roster)
    """
    # Latest roster status for the recent starter at or before this week (future weeks
    # use the newest roster available).
    rs = roster_status.sort("week")
    st = games.select("game_id", "team", "season", "week", "recent_qb_id").sort("week").join_asof(
        rs.rename({"player_id": "recent_qb_id", "week": "r_week"}), left_on="week",
        right_on="r_week", by=["season", "team", "recent_qb_id"], strategy="backward",
        check_sortedness=False,
    ).select("game_id", "team", "status")
    g = games.join(st, on=["game_id", "team"], how="left")
    recent_unavailable = ((pl.col("status").is_not_null() & ~pl.col("status").is_in(AVAILABLE))
                          | (pl.col("recent_qb_sit").fill_null(0) >= 0.9))
    g = g.with_columns(
        source=pl.when(pl.col("listed_qb_id").is_not_null()).then(pl.lit("listed"))
        .when(recent_unavailable).then(pl.lit("recent starter unavailable"))
        .otherwise(pl.lit("recent starter")),
    ).with_columns(
        expected_qb_id=pl.when(pl.col("source") == "listed").then(pl.col("listed_qb_id"))
        .when(pl.col("source") == "recent starter").then(pl.col("recent_qb_id"))
        .otherwise(pl.col("backup_qb_id")),
        expected_qb=pl.when(pl.col("source") == "listed").then(pl.col("listed_qb"))
        .when(pl.col("source") == "recent starter").then(pl.col("recent_qb"))
        .otherwise(pl.col("backup_qb")),
    )
    g = g.with_columns(
        recent_rating=rating_asof(g, states, "recent_qb_id"),
        expected_rating=rating_asof(g, states, "expected_qb_id"),
        qb_change=(pl.col("expected_qb_id").fill_null("") != pl.col("recent_qb_id").fill_null(""))
        .cast(pl.Float64),
    )
    return g.with_columns(
        qb_delta=pl.when(pl.col("qb_change") > 0)
        .then(pl.col("expected_rating") - pl.col("recent_rating")).otherwise(0.0),
    )


def build_qb_features(con: duckdb.DuckDBPyConnection) -> None:
    """Writes game_qb. Requires game_availability (src.features.availability)."""
    states = qb_states(con.execute(QB_GAMES_SQL.format(pbp="raw_pbp")).pl())
    fix = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in TEAM_FIX.items())
    games = con.execute(f"""
        WITH tg AS (
            SELECT game_id, season, week, CAST(gameday AS DATE) AS game_date,
                   CASE home_team {fix} ELSE home_team END AS team,
                   home_qb_id AS listed_qb_id, home_qb_name AS listed_qb FROM raw_schedules
            UNION ALL
            SELECT game_id, season, week, CAST(gameday AS DATE) AS game_date,
                   CASE away_team {fix} ELSE away_team END AS team,
                   away_qb_id AS listed_qb_id, away_qb_name AS listed_qb FROM raw_schedules
        ),
        inj AS (
            SELECT season, week, CASE team {fix} ELSE team END AS team, gsis_id,
                   max(CASE report_status WHEN 'Out' THEN 1.0 WHEN 'Doubtful' THEN 0.9
                       WHEN 'Questionable' THEN 0.2 ELSE 0.0 END) AS sit
            FROM raw_injuries GROUP BY ALL
        )
        SELECT tg.*, a.starting_qb_id AS recent_qb_id, a.starting_qb AS recent_qb,
               a.backup_qb_id, a.backup_qb, i.sit AS recent_qb_sit
        FROM tg
        LEFT JOIN game_availability a ON a.game_id = tg.game_id AND a.team = tg.team
        LEFT JOIN inj i ON i.season = tg.season AND i.week = tg.week AND i.team = tg.team
                       AND i.gsis_id = a.starting_qb_id
    """).pl()
    roster = con.execute(f"""
        SELECT season, week, CASE team {fix} ELSE team END AS team, gsis_id AS player_id,
               any_value(status) AS status
        FROM raw_rosters_weekly WHERE position = 'QB' AND gsis_id IS NOT NULL
        GROUP BY ALL
    """).pl()
    out = expected_starters(games, roster, states)
    con.register("_qb", out.to_arrow())
    con.execute("CREATE OR REPLACE TABLE game_qb AS SELECT * FROM _qb")
    con.unregister("_qb")
