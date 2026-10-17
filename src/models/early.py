"""Tuesday picks vs. the opening line.

The game-time model is compared with the closing line, which already contains everything
known by kickoff. The fair test of "beating Vegas" is earlier in the week: a model that
only uses what's known on the Tuesday of game week, blended with the *opening* line, scored
against that same opening line.

Tuesday information: every earlier game (rolling form, opponent-adjusted ratings), rest
days, and the QB who has been starting unless last week's roster had him on injured
reserve. No injury reports (they start Wednesday), no announced starters, no closing line.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats
from scipy.stats import norm

from src.models.game import SPREAD_SIGMA, market_blend, walk_forward


def tuesday_predictions(games: pl.DataFrame, feats: list[str], models, start_season: int
                        ) -> pl.DataFrame:
    """games must carry the Tuesday-view QB columns and open_spread. Returns game_id,
    p_tue_stats (stats only) and p_tuesday (stats blended with the opening line)."""
    g = games.with_columns(p_open=pl.Series(
        norm.cdf(games["open_spread"].fill_null(np.nan).to_numpy() / SPREAD_SIGMA)).fill_nan(None))
    st = walk_forward(g, feats, start_season - 1, models, explain=None).select(
        "game_id", pl.col("p_logit").alias("p_tue_stats"))
    g = g.join(st, on="game_id", how="left")
    bl = market_blend(g, start_season, stats_col="p_tue_stats", line_col="p_open").select(
        "game_id", pl.col("p_blend").alias("p_tuesday"))
    return g.select("game_id", "p_open").join(st, on="game_id", how="left").join(
        bl, on="game_id", how="left")


def _ll(p: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def evaluate(out: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Summary rows (metric, value, games, p_value) and per-season pick rates."""
    o = out.filter(pl.col("home_win").is_not_null() & pl.col("p_tuesday").is_not_null()
                   & pl.col("p_open").is_not_null() & pl.col("p_vegas").is_not_null())
    y = o["home_win"].to_numpy()
    rows = []

    def add(metric, value, n, p=None):
        rows.append({"metric": metric, "value": float(value), "games": int(n),
                     "p_value": None if p is None else float(p)})

    for name, col in [("Tuesday pick", "p_tuesday"), ("Tuesday stats model", "p_tue_stats"),
                      ("Opening line", "p_open"), ("Closing line", "p_vegas")]:
        p = o[col].to_numpy()
        add(f"{name}: picks right", ((p > .5) == (y == 1)).mean(), len(y))
        add(f"{name}: log loss", _ll(p, y), len(y))
    a = (o["p_tuesday"].to_numpy() > .5) == (y == 1)
    b = (o["p_open"].to_numpy() > .5) == (y == 1)
    wins, losses = int((a & ~b).sum()), int((~a & b).sum())
    add("Games where only the Tuesday pick was right", wins, wins + losses,
        stats.binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else None)
    add("Games where only the opening line was right", losses, wins + losses)

    model_spread = SPREAD_SIGMA * norm.ppf(np.clip(o["p_tue_stats"].to_numpy(), 1e-4, 1 - 1e-4))
    edge = model_spread - o["open_spread"].to_numpy()
    move = o["spread_line"].to_numpy() - o["open_spread"].to_numpy()
    vs_open = o["result"].to_numpy() - o["open_spread"].to_numpy()
    for th in (3.0, 4.5):
        m = (np.abs(edge) >= th) & (move != 0)
        k = int((np.sign(move[m]) == np.sign(edge[m])).sum())
        add(f"Line moved toward our Tuesday number (disagree by {th:g}+ pts)",
            k / max(m.sum(), 1), m.sum(), stats.binomtest(k, int(m.sum()), 0.5).pvalue)
        mm = (np.abs(edge) >= th) & (vs_open != 0)
        c = int((np.sign(vs_open[mm]) == np.sign(edge[mm])).sum())
        add(f"Our side covered the opening spread (disagree by {th:g}+ pts)",
            c / max(mm.sum(), 1), mm.sum(),
            stats.binomtest(c, int(mm.sum()), 0.524, alternative="greater").pvalue)

    by_season = o.group_by("season").agg(
        pl.len().alias("games"),
        ((pl.col("p_tuesday") > .5) == (pl.col("home_win") == 1)).mean().alias("tuesday_pick"),
        ((pl.col("p_open") > .5) == (pl.col("home_win") == 1)).mean().alias("opening_line"),
        ((pl.col("p_vegas") > .5) == (pl.col("home_win") == 1)).mean().alias("closing_line"),
    ).sort("season")
    return pl.DataFrame(rows), by_season
