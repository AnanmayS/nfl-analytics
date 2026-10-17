"""Strictly pre-kickoff team features for the game model.

1. team_game_stats (DuckDB): per team-game offensive and defensive rates from pbp.
2. Rolling state (polars): within each team-season, an exponentially weighted mean of game
   values where the previous season's final value — regressed toward the league mean —
   enters as `prior_games` pseudo-games that decay like any other game. A game's feature
   only uses games played strictly before its kickoff date.
3. game_features (DuckDB): one row per scheduled game (played or not) with home/away
   features, rest, venue, Vegas lines and the outcome.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import polars as pl

# Old schedule/snap/roster rows use pre-relocation codes; pbp uses current ones.
TEAM_FIX = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

OFF_METRICS = ["pass_epa", "rush_epa", "success_rate", "early_down_epa", "turnover_rate",
               "pressure_rate"]
METRICS = [f"off_{m}" for m in OFF_METRICS] + [f"def_{m}" for m in OFF_METRICS]

TEAM_GAME_SQL = """
CREATE OR REPLACE TABLE team_game_stats AS
WITH p AS (
    SELECT game_id, season, game_date, posteam, defteam, epa, success, pass, rush, down,
           qb_dropback, sack, qb_hit, interception, fumble_lost
    FROM {pbp}
    WHERE play_type IN ('pass', 'run') AND epa IS NOT NULL
      AND coalesce(two_point_attempt, 0) = 0 AND posteam IS NOT NULL
),
off AS (
    SELECT game_id, season, CAST(min(game_date) AS DATE) AS game_date,
           posteam AS team, any_value(defteam) AS opp,
           count(*)                                    AS plays,
           avg(epa) FILTER (WHERE pass = 1)            AS pass_epa,
           avg(epa) FILTER (WHERE rush = 1)            AS rush_epa,
           avg(success)                                AS success_rate,
           avg(epa) FILTER (WHERE down IN (1, 2))      AS early_down_epa,
           avg(coalesce(interception, 0) + coalesce(fumble_lost, 0)) AS turnover_rate,
           -- pressure proxy: sack or QB hit per dropback (participation pressure data
           -- stops in 2025, so this keeps one definition across all seasons)
           avg(greatest(coalesce(sack, 0), coalesce(qb_hit, 0)))
               FILTER (WHERE qb_dropback = 1)          AS pressure_rate
    FROM p GROUP BY game_id, season, posteam
)
SELECT o.game_id, o.season, o.game_date, o.team, o.opp, o.plays,
       {off_cols}, {def_cols}
FROM off o
JOIN off d ON d.game_id = o.game_id AND d.team = o.opp
"""


@dataclass(frozen=True)
class RollingParams:
    halflife: float = 5.0     # games
    prior_games: float = 4.0  # weight of last season's (regressed) value at week 1
    regress: float = 0.5      # share of last season's deviation from league mean kept


def build_team_game_stats(con: duckdb.DuckDBPyConnection, pbp: str = "raw_pbp") -> None:
    con.execute(TEAM_GAME_SQL.format(
        pbp=pbp,
        off_cols=", ".join(f"o.{m} AS off_{m}" for m in OFF_METRICS),
        def_cols=", ".join(f"d.{m} AS def_{m}" for m in OFF_METRICS),
    ))


def rolling_state(
    games: pl.DataFrame, priors: pl.DataFrame, metrics: list[str], p: RollingParams
) -> pl.DataFrame:
    """Post-game feature state for each played team-game.

    games: one row per played team-game with columns team, season, game_date, *metrics.
    Returns team, season, game_date, n_games (games played this season through this one),
    and each metric's weighted mean *including* that game. Callers must join it to a later
    kickoff (see pregame_features) — never to the same game.
    """
    d = 0.5 ** (1 / p.halflife)
    g = games.sort("team", "season", "game_date").with_columns(
        k=pl.int_range(pl.len()).over("team", "season")  # 0-based game index in season
    )
    g = g.join(priors, on=["team", "season"], how="left")

    # After game k (0-based), weights: game i -> d^(k-i), prior -> prior_games * d^(k+1).
    # Sum_i d^(k-i) x_i = d^k * cumsum(x_i d^-i); fine numerically for <= ~22 games.
    exprs = []
    for m in metrics:
        x = pl.col(m)
        w_obs = pl.when(x.is_not_null()).then(d ** -pl.col("k")).otherwise(0.0)
        s = (x.fill_null(0.0) * w_obs).cum_sum().over("team", "season") * d ** pl.col("k")
        w = w_obs.cum_sum().over("team", "season") * d ** pl.col("k")
        pw = pl.when(pl.col(f"prior_{m}").is_not_null()).then(
            p.prior_games * d ** (pl.col("k") + 1)).otherwise(0.0)
        exprs.append(
            ((s + pw * pl.col(f"prior_{m}").fill_null(0.0)) / (w + pw)).alias(m)
        )
    return g.select("team", "season", "game_date", (pl.col("k") + 1).alias("n_games"), *exprs)


def season_priors(games: pl.DataFrame, metrics: list[str], p: RollingParams) -> pl.DataFrame:
    """Prior for season s = league mean(s-1) + regress * (team's season-(s-1) mean - league).

    Only uses season s-1 games, so it is known before season s kicks off. A team's first
    season in the data gets no prior (null).
    """
    team_season = games.group_by("team", "season").agg(pl.col(m).mean() for m in metrics)
    league = team_season.group_by("season").agg(pl.col(m).mean().alias(f"lg_{m}")
                                                for m in metrics)
    prior = team_season.join(league, on="season").select(
        "team",
        (pl.col("season") + 1).alias("season"),
        *[(pl.col(f"lg_{m}") + p.regress * (pl.col(m) - pl.col(f"lg_{m}"))).alias(f"prior_{m}")
          for m in metrics],
    )
    return prior


def pregame_features(
    schedule: pl.DataFrame, state: pl.DataFrame, priors: pl.DataFrame, metrics: list[str]
) -> pl.DataFrame:
    """Feature values for each scheduled (team, season, game_date) as of kickoff.

    Uses the latest post-game state from a game *strictly before* game_date in the same
    season; with no prior game this season, the season prior.
    """
    sched = schedule.select("game_id", "team", "season", "game_date").sort("game_date")
    st = state.sort("game_date")
    joined = sched.join_asof(
        st.rename({"game_date": "state_date"}), left_on="game_date", right_on="state_date",
        by=["team", "season"], strategy="backward", allow_exact_matches=False,
        check_sortedness=False,
    ).join(priors, on=["team", "season"], how="left")
    return joined.select(
        "game_id", "team", "season", "game_date",
        pl.col("n_games").fill_null(0).alias("games_played"),
        *[pl.coalesce(m, f"prior_{m}").alias(m) for m in metrics],
    )


def build_game_features(con: duckdb.DuckDBPyConnection, p: RollingParams | None = None) -> None:
    p = p or RollingParams()
    build_team_game_stats(con)
    tg = con.execute("SELECT * FROM team_game_stats").pl()

    sched = con.execute(
        """SELECT game_id, season, game_type, week, CAST(gameday AS DATE) AS game_date,
                  gametime, home_team, away_team, home_score, away_score, result, total,
                  spread_line, total_line, home_moneyline, away_moneyline,
                  home_rest, away_rest, location, div_game, roof
           FROM raw_schedules"""
    ).pl().with_columns(pl.col("home_team", "away_team").replace(TEAM_FIX))

    priors = season_priors(tg, METRICS, p)
    state = rolling_state(tg, priors, METRICS, p)
    long = pl.concat([
        sched.select("game_id", "season", "game_date", pl.col("home_team").alias("team")),
        sched.select("game_id", "season", "game_date", pl.col("away_team").alias("team")),
    ])
    feats = pregame_features(long, state, priors, METRICS)

    def side(prefix: str, team_col: str) -> pl.DataFrame:
        return feats.rename({c: f"{prefix}_{c}" for c in [*METRICS, "games_played"]}).rename(
            {"team": team_col}).drop("season", "game_date")

    out = (
        sched.join(side("home", "home_team"), on=["game_id", "home_team"], how="left")
        .join(side("away", "away_team"), on=["game_id", "away_team"], how="left")
        .with_columns(
            neutral=(pl.col("location") == "Neutral").cast(pl.Int8),
            home_win=pl.when(pl.col("result") > 0).then(1)
            .when(pl.col("result") < 0).then(0).otherwise(None),
        )
    )
    con.register("_gf", out.to_arrow())
    con.execute("CREATE OR REPLACE TABLE game_features AS SELECT * FROM _gf")
    con.register("_ts", state.to_arrow())
    con.execute("CREATE OR REPLACE TABLE team_rolling_state AS SELECT * FROM _ts")
