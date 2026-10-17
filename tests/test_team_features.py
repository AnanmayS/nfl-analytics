"""Leakage and correctness checks for the rolling pre-kickoff team features."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.features.team import RollingParams, pregame_features, rolling_state, season_priors

M = ["off_pass_epa"]
P = RollingParams(halflife=2.0, prior_games=3.0, regress=0.5)


def make_games(values_2023: list[float], values_2024: list[float]) -> pl.DataFrame:
    rows = []
    for season, vals in [(2023, values_2023), (2024, values_2024)]:
        for i, v in enumerate(vals):
            gd = date(season, 9, 7) + timedelta(weeks=i)
            rows.append(dict(team="AAA", season=season, game_date=gd, off_pass_epa=v))
            # league partner so league means are defined
            rows.append(dict(team="BBB", season=season, game_date=gd, off_pass_epa=0.0))
    return pl.DataFrame(rows)


def features(games: pl.DataFrame) -> pl.DataFrame:
    priors = season_priors(games, M, P)
    state = rolling_state(games, priors, M, P)
    sched = games.select(pl.lit("g").alias("game_id"), "team", "season", "game_date")
    sched = sched.with_columns(game_id=pl.format("{}_{}", "team", "game_date"))
    return pregame_features(sched, state, priors, M).sort("team", "game_date")


def aaa(df: pl.DataFrame, season: int) -> pl.DataFrame:
    return df.filter((pl.col("team") == "AAA") & (pl.col("season") == season))


def test_feature_ignores_current_and_future_games():
    base = make_games([0.1, 0.2, 0.3], [0.4, 0.5, 0.6, 0.7])
    f0 = aaa(features(base), 2024)
    for g in range(4):  # perturb game g and everything after it
        vals = [0.4, 0.5, 0.6, 0.7]
        vals[g:] = [99.0] * (4 - g)
        f1 = aaa(features(make_games([0.1, 0.2, 0.3], vals)), 2024)
        np.testing.assert_allclose(
            f1["off_pass_epa"][: g + 1].to_numpy(), f0["off_pass_epa"][: g + 1].to_numpy(),
            err_msg=f"feature for game {g} changed when game {g}+ outcomes changed",
        )


def test_week_one_uses_regressed_prior_only():
    f = features(make_games([0.2, 0.4], [5.0, 5.0]))
    wk1 = aaa(f, 2024).row(0, named=True)
    # AAA 2023 mean 0.3, BBB 0.0 -> league 0.15; prior = 0.15 + 0.5 * (0.3 - 0.15)
    assert wk1["games_played"] == 0
    assert wk1["off_pass_epa"] == pytest.approx(0.225)


def test_first_season_has_no_prior():
    f = features(make_games([0.2, 0.4], [0.0]))
    assert aaa(f, 2023)["off_pass_epa"][0] is None


def test_exponential_weighting_matches_manual():
    f = features(make_games([0.2, 0.4], [1.0, 2.0, 3.0]))
    d = 0.5 ** (1 / P.halflife)
    prior = 0.225
    # Game 3 of 2024 sees games 1 and 2: weights d (game 1), 1 (game 2), 3*d^2 (prior)
    num = 1.0 * d + 2.0 * 1 + P.prior_games * d**2 * prior
    den = d + 1 + P.prior_games * d**2
    assert aaa(f, 2024)["off_pass_epa"][2] == pytest.approx(num / den)


def test_same_day_state_is_not_used():
    games = make_games([0.2, 0.4], [1.0, 2.0])
    priors = season_priors(games, M, P)
    state = rolling_state(games, priors, M, P)
    # a schedule row on the same date as AAA's first 2024 game must not see that game
    sched = pl.DataFrame(dict(game_id=["x"], team=["AAA"], season=[2024],
                              game_date=[date(2024, 9, 7)]))
    f = pregame_features(sched, state, priors, M)
    assert f["games_played"][0] == 0
    assert f["off_pass_epa"][0] == pytest.approx(0.225)


def test_prior_independent_of_current_season():
    a = season_priors(make_games([0.2, 0.4], [1.0]), M, P)
    b = season_priors(make_games([0.2, 0.4], [-50.0]), M, P)
    key = (pl.col("team") == "AAA") & (pl.col("season") == 2024)
    assert a.filter(key)["prior_off_pass_epa"][0] == b.filter(key)["prior_off_pass_epa"][0]
