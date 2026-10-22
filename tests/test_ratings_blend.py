"""Opponent-adjusted ratings and the market blend use only earlier games."""

from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest

from src.features.ratings import RatingParams, weekly_ratings
from src.models.game import market_blend

P = RatingParams(halflife_days=100, margin_lambda=1, epa_lambda=1)
TEAMS = ["A", "B", "C", "D"]


def make_games(n_weeks: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    strength = {"A": 7, "B": 2, "C": -2, "D": -7}
    games, tgs = [], []
    rounds = [[("A", "B"), ("C", "D")], [("C", "A"), ("D", "B")], [("A", "D"), ("B", "C")],
              [("B", "A"), ("D", "C")], [("A", "C"), ("B", "D")], [("D", "A"), ("C", "B")]]
    for w in range(n_weeks):
        d = date(2025, 9, 7) + timedelta(weeks=w)
        for h, a in rounds[w % len(rounds)]:  # balanced home/away
            m = strength[h] - strength[a] + 2 + rng.normal(0, 3)
            games.append(dict(game_date=d, home_team=h, away_team=a, result=float(m), neutral=0))
            for t, o, home in [(h, a, 1), (a, h, 0)]:
                tgs.append(dict(game_date=d, team=t, opp=o, is_home=home,
                                off_epa_play=strength[t] / 50 - strength[o] / 100))
    return pl.DataFrame(games), pl.DataFrame(tgs)


def targets(weeks):
    return pl.DataFrame({"season": [2025] * len(weeks), "week": weeks,
                         "week_start": [date(2025, 9, 7) + timedelta(weeks=w - 1) for w in weeks]})


def test_ratings_recover_team_order():
    games, tg = make_games(12)
    r = weekly_ratings(games, tg, targets([13]), P).sort("margin_rating", descending=True)
    assert r["team"].to_list() == ["A", "B", "C", "D"]
    assert r["margin_rating"].sum() == pytest.approx(0, abs=1e-9)


def test_ratings_ignore_games_on_or_after_week_start():
    games, tg = make_games(16)
    base = weekly_ratings(games, tg, targets([12]), P).sort("team")
    # wreck every game from week 12 on: ratings entering week 12 must not move
    late = pl.col("game_date") >= date(2025, 9, 7) + timedelta(weeks=11)
    g2 = games.with_columns(result=pl.when(late).then(-99.0).otherwise(pl.col("result")))
    t2 = tg.with_columns(off_epa_play=pl.when(late).then(9.0).otherwise(pl.col("off_epa_play")))
    after = weekly_ratings(g2, t2, targets([12]), P).sort("team")
    np.testing.assert_allclose(base["margin_rating"], after["margin_rating"])
    np.testing.assert_allclose(base["off_rating"], after["off_rating"])


def test_market_blend_skips_games_without_a_line():
    rng = np.random.default_rng(1)
    n = 400
    p_true = rng.uniform(0.2, 0.8, n)
    df = pl.DataFrame({
        "game_id": [f"g{i}" for i in range(n)],
        "season": [2017] * 200 + [2018] * 200,
        "week": [i % 17 + 1 for i in range(n)],
        "game_date": [date(2017, 9, 1) + timedelta(days=i) for i in range(n)],
        "home_win": (rng.uniform(size=n) < p_true).astype(int),
        "p_logit": np.clip(p_true + rng.normal(0, 0.1, n), 0.05, 0.95),
        "p_vegas": [None if i % 50 == 0 else float(v) for i, v in enumerate(p_true)],
    }).with_columns(result=pl.when(pl.col("home_win") == 1).then(3.0).otherwise(-3.0))
    out = df.join(market_blend(df, 2018), on="game_id", how="left").filter(
        pl.col("season") == 2018)
    no_line = out.filter(pl.col("p_vegas").is_null())
    assert no_line.height > 0 and no_line["p_blend"].null_count() == no_line.height
    with_line = out.filter(pl.col("p_vegas").is_not_null())
    assert with_line["p_blend"].null_count() == 0
    assert {"blend_contrib_vegas_logit", "blend_contrib_stats_logit"} <= set(out.columns)
    # The stored unscaled coefficients reproduce the blend, so it can be redone with a new line
    lo = with_line.with_columns(
        sl=(pl.col("p_logit").clip(1e-4, 1 - 1e-4) / (1 - pl.col("p_logit").clip(1e-4, 1 - 1e-4)))
        .log(),
        vl=(pl.col("p_vegas").clip(1e-4, 1 - 1e-4) / (1 - pl.col("p_vegas").clip(1e-4, 1 - 1e-4)))
        .log(),
    ).select(pl.col("blend_coef_intercept") + pl.col("blend_coef_stats_logit") * pl.col("sl")
             + pl.col("blend_coef_vegas_logit") * pl.col("vl"), "p_blend")
    np.testing.assert_allclose(1 / (1 + np.exp(-lo.to_series(0).to_numpy())),
                               lo["p_blend"].to_numpy(), atol=1e-9)
