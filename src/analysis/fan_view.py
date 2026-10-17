"""Plain-language views of predictions and archetypes for the dashboard.

Everything here is a pure polars transform so it can be unit-tested without the app.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import norm

from src.models.game import SPREAD_SIGMA

# Model shown to fans: the final pick (stats model blended with the betting line when one
# is posted, else the stats model alone). STATS_COL is the stats model on its own.
MODEL_COL = "p_final"
STATS_COL = "p_logit"

PLAIN_LABELS = {
    # receiving
    "adot": "How far downfield they're targeted",
    "target_share": "Share of the team's targets",
    "air_yards_share": "Share of the team's downfield passing",
    "yacoe": "Extra yards after the catch",
    "avg_separation": "Separation from defenders",
    "middle_target_rate": "Targets over the middle",
    "rz_target_share": "Red-zone targets",
    "croe": "Catches made vs. expected",
    "epa_per_target": "Value added per target",
    # rushing
    "rush_epa": "Value added per carry",
    "ryoe_per_att": "Yards gained beyond what blocking gave",
    "route_participation": "On the field for pass plays",
    "explosive_run_rate": "Big runs (10+ yards)",
    "goal_line_share": "Carries near the goal line",
    # passing
    "time_to_throw": "Time before throwing",
    "scramble_rate": "Scrambles",
    "cpoe": "Accuracy vs. expected",
    "sack_rate": "Sacks taken",
    "epa_per_dropback": "Value added per dropback",
    "deep_pass_rate": "Deep throws (20+ yards)",
    # team defense
    "pass_epa_allowed": "Damage allowed per pass",
    "rush_epa_allowed": "Damage allowed per run",
    "pressure_rate": "Pressure on the quarterback",
    "blitz_rate": "Blitzing (5+ rushers)",
    "heavy_box_rel": "Stacking the box (8+ defenders)",
    "adot_allowed": "How deep opponents throw",
    "explosive_rate_allowed": "Big plays allowed",
    "takeaway_rate": "Takeaways",
    "third_down_rate_allowed": "Third downs allowed",
    "cpoe_allowed": "Completions allowed vs. expected",
}
QB_LABEL_OVERRIDES = {"adot": "Average throw depth"}


def plain_label(feature: str, position: str) -> str:
    if position == "QB" and feature in QB_LABEL_OVERRIDES:
        return QB_LABEL_OVERRIDES[feature]
    return PLAIN_LABELS.get(feature, feature)


def confidence_label(p: float) -> str:
    """Words for a win probability of the team we're describing."""
    q = max(p, 1 - p)
    if q < 0.55:
        return "Toss-up"
    if q < 0.65:
        return "Slight edge" if p > 0.5 else "Slight underdog"
    if q < 0.75:
        return "Favored" if p > 0.5 else "Underdog"
    return "Strong favorite" if p > 0.5 else "Big underdog"


def spread_text(spread: float | None, team: str) -> str:
    """Vegas spread from the team's point of view (positive = team favored)."""
    if spread is None:
        return "No line yet"
    if spread == 0:
        return "Pick'em"
    side = "favored" if spread > 0 else "underdog"
    return f"{team} {side} by {abs(spread):g}"


def vegas_favorite_text(spread: float | None, home: str, away: str) -> str:
    """Name the Vegas favorite. `spread` is the home team's expected margin."""
    if spread is None:
        return "No line yet"
    if spread == 0:
        return "Pick'em"
    return f"{home if spread > 0 else away} by {abs(spread):g}"


def team_games(preds: pl.DataFrame, team: str) -> pl.DataFrame:
    """Every predicted game for `team`, flipped to that team's point of view.

    Columns: season, week, game_date, is_home, opponent, win_prob, projected_margin,
    vegas_win_prob, vegas_spread, points_for, points_against, margin, status, won,
    model_correct, vegas_correct.
    """
    home = preds.filter(pl.col("home_team") == team).select(
        "game_id", "season", "week", "game_type", "game_date",
        pl.lit(True).alias("is_home"),
        pl.col("away_team").alias("opponent"),
        pl.col(MODEL_COL).alias("win_prob"),
        pl.col(STATS_COL).alias("stats_win_prob"),
        pl.col("p_vegas").alias("vegas_win_prob"),
        pl.col("spread_line").alias("vegas_spread"),
        pl.col("home_score").alias("points_for"),
        pl.col("away_score").alias("points_against"),
        pl.col("result").alias("margin"),
    )
    away = preds.filter(pl.col("away_team") == team).select(
        "game_id", "season", "week", "game_type", "game_date",
        pl.lit(False).alias("is_home"),
        pl.col("home_team").alias("opponent"),
        (1 - pl.col(MODEL_COL)).alias("win_prob"),
        (1 - pl.col(STATS_COL)).alias("stats_win_prob"),
        (1 - pl.col("p_vegas")).alias("vegas_win_prob"),
        (-pl.col("spread_line")).alias("vegas_spread"),
        pl.col("away_score").alias("points_for"),
        pl.col("home_score").alias("points_against"),
        (-pl.col("result")).alias("margin"),
    )
    g = pl.concat([home, away]).sort("game_date")
    # Projected margin is implied by the win probability, so the two never disagree.
    pm = SPREAD_SIGMA * norm.ppf(np.clip(g["win_prob"].to_numpy(), 1e-6, 1 - 1e-6))
    won = pl.when(pl.col("margin") > 0).then(True).when(pl.col("margin") < 0).then(False)
    return g.with_columns(
        projected_margin=pl.Series(pm),
        status=pl.when(pl.col("margin").is_null()).then(pl.lit("upcoming"))
        .otherwise(pl.lit("final")),
        won=won,
    ).with_columns(
        model_correct=(pl.col("win_prob") > 0.5) == pl.col("won"),
        vegas_correct=pl.when(pl.col("vegas_spread") != 0)
        .then((pl.col("vegas_spread") > 0) == pl.col("won")),
    )


def pick_summary(games: pl.DataFrame) -> dict:
    """Counts for 'the model picked the winner in X of Y games'."""
    done = games.filter(pl.col("won").is_not_null())
    vegas = done.filter(pl.col("vegas_correct").is_not_null())
    return {
        "games": done.height,
        "model_right": int(done["model_correct"].sum()) if done.height else 0,
        "vegas_games": vegas.height,
        "vegas_right": int(vegas["vegas_correct"].sum()) if vegas.height else 0,
    }


def league_picks(preds: pl.DataFrame) -> pl.DataFrame:
    """One row per finished, non-tied game from the favored side's point of view."""
    done = preds.filter(pl.col("result").is_not_null() & (pl.col("result") != 0))
    home_won = pl.col("result") > 0
    return done.select(
        "season", "week", "game_date",
        pl.max_horizontal(pl.col(MODEL_COL), 1 - pl.col(MODEL_COL)).alias("model_conf"),
        ((pl.col(MODEL_COL) > 0.5) == home_won).alias("model_correct"),
        ((pl.col(STATS_COL) > 0.5) == home_won).alias("stats_correct"),
        pl.when(pl.col("spread_line") != 0)
        .then((pl.col("spread_line") > 0) == home_won).alias("vegas_correct"),
    )


CONF_BUCKETS = [(0.5, 0.6, "50–60%"), (0.6, 0.7, "60–70%"), (0.7, 0.8, "70–80%"),
                (0.8, 1.01, "80%+")]


def confidence_check(picks: pl.DataFrame) -> pl.DataFrame:
    """For each confidence bucket: how often the model's pick actually won."""
    rows = []
    for lo, hi, label in CONF_BUCKETS:
        b = picks.filter((pl.col("model_conf") >= lo) & (pl.col("model_conf") < hi))
        if b.height:
            rows.append({"bucket": label, "games": b.height,
                         "expected": float(b["model_conf"].mean()),
                         "actual": float(b["model_correct"].mean())})
    return pl.DataFrame(rows)


def describe_archetype(centroid: dict, features: list[str], position: str,
                       n: int = 3) -> list[str]:
    """Top-n defining traits of an archetype centroid, in words."""
    zs = sorted(((f, centroid[f"z_{f}"]) for f in features), key=lambda x: -abs(x[1]))
    out = []
    for f, z in zs[:n]:
        if abs(z) < 0.25:
            break
        strength = "Much more" if abs(z) >= 0.8 else "More"
        if z < 0:
            strength = "Much less" if abs(z) >= 0.8 else "Less"
        out.append(f"{strength} than typical: {plain_label(f, position).lower()}")
    return out or ["Close to average on everything"]


def percentiles(df: pl.DataFrame, features: list[str]) -> pl.DataFrame:
    """Percentile rank (0-100) of each feature within df."""
    return df.with_columns(
        (pl.col(f).rank("average") / pl.col(f).count() * 100).alias(f"pct_{f}")
        for f in features
    )


# ------------------------------------------------------------------ team identity & reasons
# metric -> (plain label, +1 if higher is better / -1 if lower is better)
FACTORS: dict[str, tuple[str, int]] = {
    "off_pass_epa": ("passing offense", 1),
    "off_rush_epa": ("running game", 1),
    "off_success_rate": ("offensive consistency", 1),
    "off_early_down_epa": ("early-down offense", 1),
    "off_turnover_rate": ("ball security", -1),
    "off_pressure_rate": ("pass protection", -1),
    "def_pass_epa": ("pass defense", -1),
    "def_rush_epa": ("run defense", -1),
    "def_success_rate": ("defensive consistency", -1),
    "def_early_down_epa": ("early-down defense", -1),
    "def_turnover_rate": ("takeaways", 1),
    "def_pressure_rate": ("pass rush", 1),
    "def_explosive_rate": ("big-play prevention", -1),
    "def_third_down_rate": ("third-down defense", -1),
    "def_cpoe": ("pass coverage", -1),
    # whole-unit ratings: EPA per play on every snap
    "off_epa_play": ("offense as a whole", 1),
    "def_epa_play": ("defense as a whole", -1),
    # opponent-adjusted rating (point margin, adjusted for who they played)
    "margin_rating": ("overall strength adjusted for schedule", 1),
}
RATING_INPUTS = ["margin_rating", "net_epa_rating", "net_sr_rating"]
OVERALL = {"off": "off_epa_play", "def": "def_epa_play"}
OFF_CORE = ["off_pass_epa", "off_rush_epa", "off_success_rate"]
DEF_CORE = ["def_pass_epa", "def_rush_epa", "def_success_rate"]


def tier(pct: float) -> str:
    """Words for an oriented league percentile (100 = best)."""
    if pct >= 80:
        return "one of the league's best"
    if pct >= 60:
        return "above average"
    if pct >= 40:
        return "about average"
    if pct >= 20:
        return "below average"
    return "one of the league's worst"


def metric_tiers(preds: pl.DataFrame) -> pl.DataFrame:
    """Each team's pre-game rolling metrics as league percentiles for that week.

    Percentiles are oriented so 100 is always best (e.g. lowest pass EPA allowed), and are
    computed among all teams playing in the same season-week. Long format:
    season, week, team, metric, value, pct.
    """
    sides = []
    for side in ("home", "away"):
        sides.append(preds.select(
            "season", "week", pl.col(f"{side}_team").alias("team"),
            *[pl.col(f"{side}_{m}").alias(m) for m in FACTORS]))
    wide = pl.concat(sides).unique(["season", "week", "team"])
    long = wide.unpivot(index=["season", "week", "team"], on=list(FACTORS),
                        variable_name="metric", value_name="value").drop_nulls("value")
    sign = pl.col("metric").replace_strict({m: s for m, (_, s) in FACTORS.items()},
                                           return_dtype=pl.Float64)
    return long.with_columns(oriented=pl.col("value") * sign).with_columns(
        pct=((pl.col("oriented").rank("average").over("season", "week", "metric") - 0.5)
             / pl.len().over("season", "week", "metric") * 100)
    ).drop("oriented")


def team_pcts(tiers: pl.DataFrame, season: int, week: int, team: str) -> dict[str, float]:
    t = tiers.filter((pl.col("season") == season) & (pl.col("week") == week)
                     & (pl.col("team") == team))
    return dict(zip(t["metric"], t["pct"], strict=True))


def strength_word(c: float) -> str:
    c = abs(c)
    return "Big edge" if c >= 0.3 else ("Edge" if c >= 0.12 else "Slight edge")


def game_reasons(game: dict, tiers: pl.DataFrame, names: dict[str, str] | None = None
                 ) -> list[dict]:
    """Why the model leans the way it does for one game (a game_predictions row).

    Each reason: {favors, text, push} where push is the factor's log-odds contribution
    toward `favors`. Sorted strongest first. Metric reasons are kept only when the favored
    team really is better on that metric, so the sentence never contradicts itself.
    """
    names = names or {}
    home, away = game["home_team"], game["away_team"]
    nm = {t: names.get(t, t) for t in (home, away)}
    pct = {t: team_pcts(tiers, game["season"], game["week"], t) for t in (home, away)}

    def line(m: str, fav: str, other: str) -> str:
        return (f"{nm[fav]}' {FACTORS[m][0]} ({tier(pct[fav][m])}) vs. {nm[other]}' "
                f"({tier(pct[other][m])})")

    def consistent(m: str, fav: str, other: str) -> bool:
        return m in pct[fav] and m in pct[other] and pct[fav][m] > pct[other][m]

    out = []
    for unit in ("off", "def"):
        # Every model input for this unit, incl. style ones with no "better" direction.
        pushes = {k[len("contrib_d_"):]: v for k, v in game.items()
                  if k.startswith(f"contrib_d_{unit}_") and v is not None}
        total = sum(pushes.values())
        if not pushes or total == 0:
            continue
        fav, other = (home, away) if total > 0 else (away, home)
        sign = 1 if fav == home else -1
        parts = sorted(((m, sign * v) for m, v in pushes.items()
                        if m in FACTORS and m != OVERALL[unit] and sign * v > 0
                        and consistent(m, fav, other)), key=lambda x: -x[1])
        if consistent(OVERALL[unit], fav, other):
            # One line for the whole unit, naming its biggest single piece.
            detail = f"; biggest gap: {FACTORS[parts[0][0]][0]}" if parts else ""
            out.append({"favors": fav, "push": abs(total), "unit": unit,
                        "text": line(OVERALL[unit], fav, other) + detail})
        else:
            # Units net out the other way overall: list the specific pieces instead.
            for m, c in pushes.items():
                f, o = (home, away) if c > 0 else (away, home)
                if m in FACTORS and consistent(m, f, o):
                    out.append({"favors": f, "push": abs(c), "unit": unit,
                                "text": line(m, f, o)})
    # Opponent-adjusted ratings: one line for all three rating inputs.
    rp = sum(game.get(f"contrib_d_{c}") or 0 for c in RATING_INPUTS)
    if rp:
        fav, other = (home, away) if rp > 0 else (away, home)
        if consistent("margin_rating", fav, other):
            out.append({"favors": fav, "push": abs(rp), "unit": "rating",
                        "text": line("margin_rating", fav, other)})
    hf = (game.get("contrib_intercept") or 0) + (game.get("contrib_neutral") or 0) \
        + (game.get("contrib_div_game") or 0)
    if not game.get("neutral") and abs(hf) > 0:  # no home crowd at a neutral site
        fav = home if hf > 0 else away
        out.append({"favors": fav, "push": abs(hf),
                    "text": f"Home-field advantage for {nm[home]}"})
    rest = game.get("contrib_d_rest")
    hr, ar = game.get("home_rest"), game.get("away_rest")
    if rest and hr is not None and ar is not None and hr != ar:
        fav = home if rest > 0 else away
        more = hr if fav == home else ar
        less = ar if fav == home else hr
        if more > less:
            out.append({"favors": fav, "push": abs(rest),
                        "text": f"{nm[fav]} are better rested ({more:.0f} days vs. {less:.0f})"})
    qb_push = (game.get("contrib_d_qb_change") or 0) + (game.get("contrib_d_qb_delta") or 0)
    changed = [(t, sd) for t, sd in ((home, "home"), (away, "away"))
               if (game.get(f"{sd}_qb_change") or 0) > 0]
    if qb_push and changed:
        # Describe the side whose QB swing is biggest; keep it only if the push points the
        # same way (a downgrade helps the opponent, an upgrade helps the team).
        t, sd = max(changed, key=lambda x: abs(game.get(f"{x[1]}_qb_delta") or 0))
        delta = game.get(f"{sd}_qb_delta") or 0
        helps = t if delta > 0 else (away if t == home else home)
        if (qb_push > 0) == (helps == home):
            word = ("a big downgrade" if delta <= -0.08 else "a downgrade" if delta < -0.02
                    else "an upgrade" if delta > 0.02 else "about even")
            out.append({"favors": helps, "push": abs(qb_push), "text":
                        f"{nm[t]} start {game.get(f'{sd}_expected_qb') or 'a new QB'} instead of "
                        f"{game.get(f'{sd}_recent_qb') or 'their usual starter'} ({word})"})
    return sorted(out, key=lambda r: -r["push"])





def team_identity(pcts: dict[str, float], pass_rate_pct: float | None,
                  key_players: list[dict], defense_type: str | None = None) -> dict:
    """Plain-language team profile from oriented metric percentiles (100 = best),
    neutral-situation pass-rate percentile (0-1), key players with archetypes and the
    defense's archetype."""
    if pass_rate_pct is None:
        lean = "Balanced play-calling"
    elif pass_rate_pct >= 0.67:
        lean = "Pass-first offense"
    elif pass_rate_pct <= 0.33:
        lean = "Run-first offense"
    else:
        lean = "Balanced play-calling"
    def unit(overall: str, core: list[str]) -> float:
        if overall in pcts:
            return float(pcts[overall])
        vals = [pcts[m] for m in core if m in pcts]
        return float(np.mean(vals)) if vals else 50.0

    off, dfn = unit(OVERALL["off"], OFF_CORE), unit(OVERALL["def"], DEF_CORE)
    if off - dfn >= 15:
        side = "carried by its offense" if off >= 50 else "better on offense than defense"
    elif dfn - off >= 15:
        side = "carried by its defense" if dfn >= 50 else "better on defense than offense"
    elif min(off, dfn) >= 60:
        side = "strong on both sides of the ball"
    elif max(off, dfn) <= 40:
        side = "struggling on both sides of the ball"
    else:
        side = "even on both sides of the ball"
    ranked = sorted(((m, p) for m, p in pcts.items()
                     if m not in OVERALL.values() and m != "margin_rating"),
                    key=lambda kv: -kv[1])
    strengths = [f"{FACTORS[m][0]} ({tier(p)})" for m, p in ranked[:2] if p >= 65]
    weak = [f"{FACTORS[m][0]} ({tier(p)})" for m, p in ranked[-1:] if p <= 35]
    players = []
    for pos in ("QB", "RB", "WR", "TE"):
        ps = [p for p in key_players if p["pos"] == pos]
        if ps:
            players.append(pos + " " + ", ".join(
                p["player_name"] + (f" ({p['archetype']})" if p.get("archetype") else "")
                for p in ps))
    def_best = sorted(((m, p) for m, p in pcts.items()
                       if m.startswith("def_") and m != OVERALL["def"]), key=lambda kv: -kv[1])
    defense = {"type": defense_type, "overall": tier(dfn),
               "best_at": FACTORS[def_best[0][0]][0] if def_best else None,
               "worst_at": FACTORS[def_best[-1][0]][0] if len(def_best) > 1 else None}
    return {"label": f"{lean}, {side}", "strengths": strengths, "weakness": weak,
            "players": players, "offense_pct": off, "defense_pct": dfn,
            "offense_overall": tier(off), "defense": defense}
