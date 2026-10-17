"""Train/evaluate the game models and write predictions, metrics and lift tables.

Runs the walk-forward twice: base team features, and base + pre-kickoff archetype-mix
features (Part 3). Both are scored on the identical set of games.
"""

from __future__ import annotations

import logging
from datetime import datetime

import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import LogisticRegression

from src.analysis import archetype_mix, team_profile
from src.config import OUT
from src.db import connect
from src.features.availability import build_availability
from src.features.team import ALL_METRICS, build_game_features
from src.models.game import (
    AVAILABILITY,
    MODELS,
    add_availability,
    base_features,
    calibration,
    defense_features,
    score,
    vegas_probs,
    walk_forward,
)

log = logging.getLogger(__name__)
START_SEASON = 2018  # 2016 is burn-in (no prior), 2017 gives the first training season
PROBS = ["p_logit", "p_xgb"]
MARGINS = ["margin_ridge", "margin_xgb_reg"]
# Only the starting-QB flag made the cut: the broader "starters missing" counts hurt on the
# 2018-21 selection period (see README), so they're shown on the dashboard but not modeled.
INJURY_FEATURES = ["d_qb_out"]


def add_mix_features(games: pl.DataFrame, mix: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Home-minus-away pre-kickoff archetype snap shares (missing -> 0)."""
    cols = [c for c in mix.columns if c.startswith("mix_") and not c.endswith("_unlabeled")]
    def side(prefix: str) -> pl.DataFrame:
        return mix.select("game_id", pl.col("team").alias(f"{prefix}_team"),
                          *[pl.col(c).alias(f"{prefix}_{c}") for c in cols])
    g = games.join(side("home"), on=["game_id", "home_team"], how="left").join(
        side("away"), on=["game_id", "away_team"], how="left")
    feats = [f"d_{c}" for c in cols]
    return g.with_columns(
        (pl.col(f"home_{c}").fill_null(0.0) - pl.col(f"away_{c}").fill_null(0.0)).alias(f"d_{c}")
        for c in cols
    ), feats


def log_loss_vec(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def lift_table(out: pl.DataFrame, base_suffix: str = "", new_suffix: str = "_mix",
               cols: list[str] = PROBS + MARGINS, n_boot: int = 2000) -> pl.DataFrame:
    """Paired comparison of two prediction sets (columns c+base_suffix vs c+new_suffix)
    with bootstrap 95% CIs over games. Negative delta = the new set is better."""
    s = out.filter(pl.col("home_win").is_not_null() & pl.col("p_vegas").is_not_null())
    y = s["home_win"].to_numpy().astype(float)
    r = s["result"].to_numpy().astype(float)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    rows = []
    for c in cols:
        a, b = s[f"{c}{base_suffix}"].to_numpy(), s[f"{c}{new_suffix}"].to_numpy()
        if c in PROBS:
            metric = "log_loss"
            da, db = log_loss_vec(a, y), log_loss_vec(b, y)
        else:
            metric = "mae"
            da, db = np.abs(a - r), np.abs(b - r)
        diff = db - da
        boots = diff[idx].mean(1)
        rows.append(dict(model=c, metric=metric, base=da.mean(), new=db.mean(),
                         delta=diff.mean(), ci_low=np.quantile(boots, 0.025),
                         ci_high=np.quantile(boots, 0.975)))
    return pl.DataFrame(rows)


def vs_vegas(out: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    """Does a model's probability carry information beyond the closing spread?
    Likelihood-ratio test of y ~ logit(p_vegas) vs y ~ logit(p_vegas) + logit(p_model)."""
    s = out.filter(pl.col("home_win").is_not_null() & pl.col("p_vegas").is_not_null())
    y = s["home_win"].to_numpy()
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))
    def ll(X):
        m = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
        return -log_loss_vec(m.predict_proba(X)[:, 1], y).sum(), m.coef_[0]
    lv, _ = ll(logit(s["p_vegas"].to_numpy())[:, None])
    rows = []
    for c in cols:
        lf, coef = ll(np.column_stack([logit(s["p_vegas"].to_numpy()), logit(s[c].to_numpy())]))
        lr = 2 * (lf - lv)
        rows.append(dict(model=c, coef_vegas=coef[0], coef_model=coef[1], lr_stat=lr,
                         p_value=float(stats.chi2.sf(lr, 1))))
    return pl.DataFrame(rows)


def _write(con, name: str, df: pl.DataFrame) -> None:
    con.register("_t", df.to_arrow())
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _t")
    con.unregister("_t")


def run(evaluate: bool = False) -> pl.DataFrame:
    """Weekly path: rebuild features, walk-forward the main models, write predictions.

    evaluate=True also reruns the comparison models (without whole-defense extras,
    without injury info, with archetype mix) and rewrites their lift tables. That roughly
    triples the runtime, so the weekly refresh skips it and keeps the last evaluation.
    """
    archetype_mix.build()
    with connect() as con:
        build_game_features(con)
        build_availability(con)
        games = con.execute("SELECT * FROM game_features").pl()
        mix = con.execute("SELECT * FROM pregame_archetype_mix").pl()
        avail = con.execute("SELECT * FROM game_availability").pl()
    games, core = base_features(games)
    games, def_feats = defense_features(games)
    games, _ = add_availability(games, avail)
    games, mix_feats = add_mix_features(games, mix)
    games = vegas_probs(games)
    # Main model: team form incl. whole-defense measures, plus this week's QB availability.
    feats = core + def_feats + INJURY_FEATURES

    log.info("walk-forward: main (%d features)", len(feats))
    main = walk_forward(games, feats, START_SEASON)
    out = games.select(
        "game_id", "season", "week", "game_type", "game_date", "gametime", "home_team",
        "away_team", "home_score", "away_score", "result", "home_win", "spread_line",
        "p_vegas", "p_moneyline", "home_rest", "away_rest", "neutral",
        *[f"{side}_{m}" for side in ("home", "away") for m in ALL_METRICS],
        *[f"{side}_{c}" for side in ("home", "away")
          for c in [*AVAILABILITY, "key_absences", "starting_qb", "report_available"]],
    ).join(main, on="game_id", how="inner")

    tables: list[tuple[str, pl.DataFrame]] = []
    if evaluate:
        probs_only = [m for m in MODELS if m.kind == "prob"]
        for suffix, fs, models in [("_nodef", core + INJURY_FEATURES, probs_only),
                                   ("_noinj", core + def_feats, probs_only),
                                   ("_mix", feats + mix_feats, MODELS)]:
            log.info("walk-forward comparison %s (%d features)", suffix, len(fs))
            alt = walk_forward(games, fs, START_SEASON, models, explain=None)
            out = out.join(alt.rename({c: f"{c}{suffix}" for c in alt.columns if c != "game_id"}),
                           on="game_id", how="inner")
        tables += [("game_model_lift", lift_table(out)),
                   ("game_defense_lift",
                    lift_table(out, base_suffix="_nodef", new_suffix="", cols=PROBS)),
                   ("game_injury_lift",
                    lift_table(out, base_suffix="_noinj", new_suffix="", cols=PROBS))]

    extra = [f"{c}_mix" for c in PROBS] if evaluate else []
    probs_all = [*PROBS, *extra, "p_vegas", "p_moneyline"]
    margins_all = [*MARGINS, *([f"{c}_mix" for c in MARGINS] if evaluate else []),
                   "spread_line"]
    metrics = score(out.filter(pl.col("p_moneyline").is_not_null()), probs_all, margins_all)
    seasons = sorted(out.filter(pl.col("home_win").is_not_null())["season"].unique())
    by_season = pl.concat([
        score(out.filter(pl.col("season") == s), [*PROBS, "p_vegas"], [])
        .with_columns(season=pl.lit(s)) for s in seasons
    ])
    tables += [("game_predictions", out), ("game_metrics", metrics),
               ("game_metrics_by_season", by_season),
               ("game_calibration", calibration(out, [*PROBS, "p_vegas"])),
               ("game_vs_vegas", vs_vegas(out, PROBS + extra))]

    with connect() as con:
        team_profile.build(con)
        for name, df in tables:
            _write(con, name, df)
        _write(con, "refresh_info", refresh_info(con))
    log.info("walk-forward metrics (%d-%d):\n%s", START_SEASON, out["season"].max(), metrics)
    for name, df in tables:
        if name.endswith("_lift"):
            log.info("%s:\n%s", name, df)
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT / "game_predictions.parquet")
    return metrics


def refresh_info(con) -> pl.DataFrame:
    """When the data was refreshed and how current each source is, for the dashboard."""
    row = con.execute("""
        SELECT max(CAST(gameday AS DATE)) FILTER (WHERE result IS NOT NULL) AS results_through,
               max(season) AS season
        FROM raw_schedules""").fetchone()
    inj_week = con.execute("SELECT max(week) FROM raw_injuries WHERE season = ?",
                           [row[1]]).fetchone()[0]
    return pl.DataFrame({"refreshed_at": [datetime.now()], "results_through": [row[0]],
                         "season": [row[1]],
                         "injury_report_week": [int(inj_week) if inj_week else None]})
