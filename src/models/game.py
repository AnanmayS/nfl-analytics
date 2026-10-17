"""Game outcome models: home win probability and home point margin.

Walk-forward by week: for every (season, week) in the test period, fit on all games that
kicked off before that week's first game, then predict that week. Unplayed games in the
current season get predictions too (shown in the app) but are never scored.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

from src.features.team import OFF_METRICS

log = logging.getLogger(__name__)

SPREAD_SIGMA = 13.45  # sd of (result - closing spread); standard NFL value
SEED = 42


def base_features(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    """Home-minus-away differences of every rolling metric, plus rest and venue."""
    exprs, names = [], []
    for side in ("off", "def"):
        for m in OFF_METRICS:
            n = f"d_{side}_{m}"
            exprs.append((pl.col(f"home_{side}_{m}") - pl.col(f"away_{side}_{m}")).alias(n))
            names.append(n)
    exprs += [
        (pl.col("home_rest") - pl.col("away_rest")).clip(-7, 7).alias("d_rest"),
        pl.col("neutral").cast(pl.Float64),
        pl.col("div_game").cast(pl.Float64),
    ]
    names += ["d_rest", "neutral", "div_game"]
    return df.with_columns(exprs), names


@dataclass
class ModelSpec:
    name: str
    kind: str  # "prob" or "margin"
    factory: Callable[[], object]


MODELS = [
    ModelSpec("logit", "prob", lambda: make_pipeline(
        StandardScaler(), LogisticRegression(C=0.1, max_iter=1000))),
    ModelSpec("xgb", "prob", lambda: XGBClassifier(
        n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0, random_state=SEED,
        eval_metric="logloss")),
    ModelSpec("ridge", "margin", lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
    ModelSpec("xgb_reg", "margin", lambda: XGBRegressor(
        n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=10, reg_lambda=5.0, random_state=SEED)),
]


def vegas_probs(df: pl.DataFrame) -> pl.DataFrame:
    """Closing-spread implied home win prob (normal approx) and de-vigged moneyline prob."""
    def implied(ml: pl.Expr) -> pl.Expr:
        return pl.when(ml > 0).then(100 / (ml + 100)).otherwise(-ml / (-ml + 100))
    ph, pa = implied(pl.col("home_moneyline")), implied(pl.col("away_moneyline"))
    spread = df["spread_line"].to_numpy().astype(float)
    return df.with_columns(
        p_vegas=pl.Series(norm.cdf(spread / SPREAD_SIGMA)).fill_nan(None),
        p_moneyline=ph / (ph + pa),
    )


def linear_contributions(model, X: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-feature log-odds contributions of a StandardScaler + LogisticRegression pipeline.

    contributions.sum(axis=1) + intercept equals model.decision_function(X), so each
    prediction splits exactly into how much every factor pushed it toward the home team.
    """
    scaler, lr = model[0], model[-1]
    return scaler.transform(X) * lr.coef_[0], float(lr.intercept_[0])


def walk_forward(
    df: pl.DataFrame, features: list[str], start_season: int, models: list[ModelSpec] = MODELS,
    explain: str | None = "logit",
) -> pl.DataFrame:
    """Predictions for every game from start_season on, each from a model fit only on
    games that kicked off before that game's week began.

    For the model named `explain`, also returns contrib_<feature> and contrib_intercept
    columns (log-odds pushes toward the home team) from that same week's fit."""
    df = df.sort("game_date")
    weeks = (
        df.filter(pl.col("season") >= start_season)
        .group_by("season", "week").agg(pl.col("game_date").min().alias("week_start"))
        .sort("week_start")
    )
    complete = pl.all_horizontal(pl.col(f).is_not_null() for f in features)
    out = []
    for season, week, week_start in weeks.iter_rows():
        train = df.filter((pl.col("game_date") < week_start) & pl.col("result").is_not_null()
                          & complete)
        test = df.filter((pl.col("season") == season) & (pl.col("week") == week) & complete)
        if test.height == 0:
            continue
        Xtr, Xte = train.select(features).to_numpy(), test.select(features).to_numpy()
        preds = {}
        for m in models:
            if m.kind == "prob":
                tr = train.filter(pl.col("home_win").is_not_null())
                model = m.factory().fit(tr.select(features).to_numpy(), tr["home_win"].to_numpy())
                preds[f"p_{m.name}"] = model.predict_proba(Xte)[:, 1]
                if m.name == explain:
                    contrib, intercept = linear_contributions(model, Xte)
                    for j, f in enumerate(features):
                        preds[f"contrib_{f}"] = contrib[:, j]
                    preds["contrib_intercept"] = np.full(len(Xte), intercept)
            else:
                model = m.factory().fit(Xtr, train["result"].to_numpy())
                preds[f"margin_{m.name}"] = model.predict(Xte)
        out.append(test.select("game_id").with_columns(
            pl.Series(k, np.asarray(v, dtype=np.float64)) for k, v in preds.items()))
        log.debug("%s wk %s: train %d, test %d", season, week, train.height, test.height)
    return pl.concat(out)


def score(df: pl.DataFrame, prob_cols: list[str], margin_cols: list[str]) -> pl.DataFrame:
    """Log loss / Brier / accuracy per probability column on played, non-tied games with a
    Vegas line; MAE / RMSE per margin column."""
    s = df.filter(pl.col("home_win").is_not_null() & pl.col("p_vegas").is_not_null())
    y = s["home_win"].to_numpy()
    rows = []
    for c in prob_cols:
        p = np.clip(s[c].to_numpy(), 1e-6, 1 - 1e-6)
        rows.append(dict(
            model=c, n=len(y),
            log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            brier=float(np.mean((p - y) ** 2)),
            accuracy=float(np.mean((p > 0.5) == y)),
            mae=None, rmse=None,
        ))
    r = s["result"].to_numpy()
    for c in margin_cols:
        e = s[c].to_numpy() - r
        rows.append(dict(model=c, n=len(r), log_loss=None, brier=None, accuracy=float(
            np.mean((s[c].to_numpy() > 0) == y)), mae=float(np.mean(np.abs(e))),
            rmse=float(np.sqrt(np.mean(e**2)))))
    return pl.DataFrame(rows)


def calibration(df: pl.DataFrame, prob_cols: list[str], bins: int = 10) -> pl.DataFrame:
    s = df.filter(pl.col("home_win").is_not_null() & pl.col("p_vegas").is_not_null())
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for c in prob_cols:
        b = np.clip(np.digitize(s[c].to_numpy(), edges) - 1, 0, bins - 1)
        out.append(s.select(c, "home_win").with_columns(bin=pl.Series(b))
                   .group_by("bin").agg(pl.col(c).mean().alias("predicted"),
                                        pl.col("home_win").mean().alias("observed"),
                                        pl.len().alias("n"))
                   .with_columns(model=pl.lit(c)).sort("bin"))
    return pl.concat(out)
