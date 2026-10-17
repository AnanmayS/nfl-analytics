"""Opponent-adjusted team ratings, refit every week from games strictly before it.

Weighted ridge regressions over recent games (older games decay with halflife_days, so
last season fades out over the offseason):
  margin rating   home margin (capped at +/- MARGIN_CAP; blowout points are mostly noise)
                  = hfa * (not neutral) + r_home - r_away -- a decayed, schedule-adjusted
                  point-differential rating, like SRS
  EPA / success-rate ratings
                  a team's offensive EPA/play (or success rate) in a game
                  = mu + off_team + def_opponent (+ hfa if at home);
                  off > 0 is a good offense, def > 0 a bad defense

Settings were chosen on 2018-23 only and confirmed on 2024-26 (see README).

Raw rolling stats treat a strong game against a weak opponent the same as against a
strong one; these ratings adjust for who each team played.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

MARGIN_CAP = 21


@dataclass(frozen=True)
class RatingParams:
    halflife_days: float = 200.0
    margin_lambda: float = 8.0   # ridge penalty (in games of weight) for margin ratings
    epa_lambda: float = 8.0
    lookback_days: int = 900


DEFAULT_PARAMS = RatingParams()


def _ridge(X: np.ndarray, y: np.ndarray, w: np.ndarray, lam: float, n_free: int) -> np.ndarray:
    """Weighted ridge; the first n_free columns (intercept / home field) aren't penalized."""
    Xw = X * w[:, None]
    A = X.T @ Xw
    pen = np.full(X.shape[1], lam)
    pen[:n_free] = 1e-6
    A[np.diag_indices_from(A)] += pen
    return np.linalg.solve(A, Xw.T @ y)


def weekly_ratings(games: pl.DataFrame, team_games: pl.DataFrame, targets: pl.DataFrame,
                   p: RatingParams = DEFAULT_PARAMS) -> pl.DataFrame:
    """Ratings for every (season, week) in `targets`, fit on games before that week.

    games: played games with game_date, home_team, away_team, result (home margin), neutral
    team_games: game_date, team, opp, is_home, off_epa_play (one row per team-game)
    targets: season, week, week_start
    Returns season, week, team, margin_rating, off_rating, def_rating.
    """
    teams = sorted(set(games["home_team"]) | set(games["away_team"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    g_date = games["game_date"].to_numpy().astype("datetime64[D]")
    h = np.array([idx[t] for t in games["home_team"]])
    a = np.array([idx[t] for t in games["away_team"]])
    margin = games["result"].to_numpy().astype(float)
    neutral = games["neutral"].to_numpy().astype(float)
    tg_date = team_games["game_date"].to_numpy().astype("datetime64[D]")
    o = np.array([idx.get(t, -1) for t in team_games["team"]])
    d = np.array([idx.get(t, -1) for t in team_games["opp"]])
    is_home = team_games["is_home"].to_numpy().astype(float)
    epa = team_games["off_epa_play"].to_numpy().astype(float)
    ok_tg = (o >= 0) & (d >= 0) & np.isfinite(epa)

    out = []
    for season, week, start in targets.select("season", "week", "week_start").iter_rows():
        s = np.datetime64(start, "D")
        age = (s - g_date).astype(float)
        m = (age > 0) & (age <= p.lookback_days)
        if m.sum() < 20:
            continue
        w = 0.5 ** (age[m] / p.halflife_days)
        X = np.zeros((m.sum(), 1 + n))
        X[:, 0] = 1 - neutral[m]
        rows = np.arange(m.sum())
        X[rows, 1 + h[m]] = 1
        X[rows, 1 + a[m]] = -1
        r = _ridge(X, margin[m], w, p.margin_lambda, 1)[1:]
        r = r - r.mean()

        age2 = (s - tg_date).astype(float)
        m2 = ok_tg & (age2 > 0) & (age2 <= p.lookback_days)
        w2 = 0.5 ** (age2[m2] / p.halflife_days)
        X2 = np.zeros((m2.sum(), 2 + 2 * n))
        rows2 = np.arange(m2.sum())
        X2[:, 0] = 1
        X2[:, 1] = is_home[m2]
        X2[rows2, 2 + o[m2]] = 1
        X2[rows2, 2 + n + d[m2]] = 1
        b = _ridge(X2, epa[m2], w2, p.epa_lambda, 2)
        off, dfn = b[2:2 + n], b[2 + n:]
        out.append(pl.DataFrame({"season": season, "week": week, "team": teams,
                                 "margin_rating": r, "off_rating": off - off.mean(),
                                 "def_rating": dfn - dfn.mean()}))
    return pl.concat(out)


RATING_COLS = ["margin_rating", "net_epa_rating", "net_sr_rating"]


def build_ratings(con, p: RatingParams = DEFAULT_PARAMS) -> pl.DataFrame:
    """team_ratings table: each team's ratings entering every scheduled week."""
    from src.features.team import TEAM_FIX  # local import: team.py is heavy

    fix = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in TEAM_FIX.items())
    sched = con.execute(f"""
        SELECT game_id, season, week, CAST(gameday AS DATE) AS game_date,
               CASE home_team {fix} ELSE home_team END AS home_team,
               CASE away_team {fix} ELSE away_team END AS away_team,
               result, (location = 'Neutral')::INT AS neutral
        FROM raw_schedules""").pl()
    tg = con.execute("""
        SELECT t.game_date, t.team, t.opp, t.off_epa_play, t.off_success_rate,
               (t.team = s.home_team)::INT AS is_home
        FROM team_game_stats t JOIN (
            SELECT game_id, CASE home_team {fix} ELSE home_team END AS home_team
            FROM raw_schedules) s USING (game_id)""".replace("{fix}", fix)).pl()
    played = sched.filter(pl.col("result").is_not_null()).with_columns(
        pl.col("result").clip(-MARGIN_CAP, MARGIN_CAP))
    targets = sched.group_by("season", "week").agg(pl.col("game_date").min().alias("week_start"))
    epa = weekly_ratings(played, tg, targets, p)
    sr = weekly_ratings(played, tg.with_columns(off_epa_play=pl.col("off_success_rate")),
                        targets, p)
    out = epa.select("season", "week", "team", "margin_rating",
                     (pl.col("off_rating") - pl.col("def_rating")).alias("net_epa_rating")).join(
        sr.select("season", "week", "team",
                  (pl.col("off_rating") - pl.col("def_rating")).alias("net_sr_rating")),
        on=["season", "week", "team"])
    con.register("_r", out.to_arrow())
    con.execute("CREATE OR REPLACE TABLE team_ratings AS SELECT * FROM _r")
    con.unregister("_r")
    return out


def add_rating_features(df: pl.DataFrame, ratings: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Join home/away ratings for each game's week and add home-minus-away differences."""
    for side in ("home", "away"):
        df = df.join(ratings.select("season", "week", pl.col("team").alias(f"{side}_team"),
                                    *[pl.col(c).alias(f"{side}_{c}") for c in RATING_COLS]),
                     on=["season", "week", f"{side}_team"], how="left")
    names = [f"d_{c}" for c in RATING_COLS]
    return df.with_columns(
        (pl.col(f"home_{c}") - pl.col(f"away_{c}")).alias(f"d_{c}") for c in RATING_COLS
    ), names
