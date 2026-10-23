"""Paper betting: flat bets on our picks, placed about an hour before kickoff, then settled.

Every run (`python -m src.cli bets`, every 15 minutes in .github/workflows/bets.yml):

1. pulls the current season's schedule from nflverse for the latest lines and scores;
2. for each game kicking off within the next hour that has no bet yet, re-blends the stats
   model with the line as it stands now (the blend's coefficients for that week are stored in
   game_predictions) and places one bet per strategy;
3. settles every open bet whose game has a final score.

The ledger (data/bets.parquet) only ever grows: a bet's odds, line and probability are fixed
when it's placed. Games that kicked off before the ledger existed are backfilled once from
the closing line and marked as such, so the season total starts at week 1.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import polars as pl
from scipy.stats import norm

from src.config import DATA, DB_PATH
from src.models.game import SPREAD_SIGMA

log = logging.getLogger(__name__)

LEDGER = DATA / "bets.parquet"
STAKE = 100.0
LEAD = timedelta(minutes=60)  # bet this long before kickoff...
GRACE = timedelta(minutes=5)  # ...give or take a late scheduled run
VALUE_EDGE = 0.03  # value bets need our win chance 3+ points above the odds' break-even
DEFAULT_SPREAD_ODDS = -110
EASTERN = ZoneInfo("America/New_York")  # nflverse gametimes are Eastern

STRATEGIES = {
    "pick_ml": "Moneyline on our pick",
    "pick_spread": "Spread on our side",
    "value_ml": "Moneyline value bets",
}
STRATEGY_HELP = {
    "pick_ml": "Every game: our pick to win outright, at its moneyline.",
    "pick_spread": "Every game: the side our projected margin says beats the spread.",
    "value_ml": f"Only when our win chance beats the moneyline's break-even by "
                f"{VALUE_EDGE:.0%}+ points (either side, underdogs included).",
}
BLEND_COEFS = ["blend_coef_intercept", "blend_coef_stats_logit", "blend_coef_vegas_logit"]
UTC_TS = pl.Datetime("us", "UTC")
SCHEMA = {
    "bet_id": pl.String, "strategy": pl.String, "season": pl.Int64, "week": pl.Int64,
    "game_id": pl.String, "kickoff": UTC_TS, "placed_at": UTC_TS, "placement": pl.String,
    "team": pl.String, "opponent": pl.String, "is_home": pl.Boolean, "bet_type": pl.String,
    "line": pl.Float64, "odds": pl.Int64, "stake": pl.Float64, "p_win": pl.Float64,
    "breakeven": pl.Float64, "edge": pl.Float64, "pick_source": pl.String,
    "status": pl.String, "team_score": pl.Int64, "opp_score": pl.Int64,
    "profit": pl.Float64, "settled_at": UTC_TS,
}


# ------------------------------------------------------------------ odds math
def breakeven(odds: int) -> float:
    """Win rate needed to break even at American odds (includes the vig)."""
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def win_profit(odds: int, stake: float = STAKE) -> float:
    return stake * (odds / 100 if odds > 0 else 100 / -odds)


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)  # same clip as src.models.game._logit
    return float(np.log(p / (1 - p)))


def kickoff_utc(gameday: str, gametime: str | None) -> datetime:
    t = gametime or "13:00"
    return datetime.fromisoformat(f"{gameday}T{t}").replace(tzinfo=EASTERN).astimezone(UTC)


def home_prob(p_stats: float, spread: float | None, coef: dict | None) -> tuple[float, str]:
    """Final pick (home win chance) with the line as it stands: the week's stats + line blend,
    or the stats model alone when there's no line or no fitted blend."""
    if spread is None or coef is None:
        return p_stats, "stats only"
    p_line = float(norm.cdf(spread / SPREAD_SIGMA))
    z = (coef["blend_coef_intercept"] + coef["blend_coef_stats_logit"] * _logit(p_stats)
         + coef["blend_coef_vegas_logit"] * _logit(p_line))
    return float(1 / (1 + np.exp(-z))), "stats + line"


# ------------------------------------------------------------------ placing
def make_bets(g: dict, p_home: float, source: str, placed_at: datetime | None,
              placement: str) -> list[dict]:
    """One bet per strategy for game g (a schedule row: teams, spread_line, moneylines,
    spread odds), given our final home win chance. Strategies with no usable line skip."""
    spread = g.get("spread_line")
    sides = {
        True: dict(team=g["home_team"], opponent=g["away_team"], p=p_home,
                   ml=g.get("home_moneyline"), spread_odds=g.get("home_spread_odds")),
        False: dict(team=g["away_team"], opponent=g["home_team"], p=1 - p_home,
                    ml=g.get("away_moneyline"), spread_odds=g.get("away_spread_odds")),
    }

    def bet(strategy, home, bet_type, odds, p_win, line=None):
        s = sides[home]
        be = breakeven(odds)
        return dict(
            bet_id=f"{g['game_id']}:{strategy}", strategy=strategy, season=g["season"],
            week=g["week"], game_id=g["game_id"], kickoff=kickoff_utc(g["gameday"],
                                                                       g["gametime"]),
            placed_at=placed_at, placement=placement, team=s["team"], opponent=s["opponent"],
            is_home=home, bet_type=bet_type, line=line, odds=int(odds), stake=STAKE,
            p_win=p_win, breakeven=be, edge=p_win - be, pick_source=source, status="open",
            team_score=None, opp_score=None, profit=None, settled_at=None)

    out = []
    pick_home = p_home >= 0.5
    if sides[pick_home]["ml"] is not None:
        out.append(bet("pick_ml", pick_home, "moneyline", sides[pick_home]["ml"],
                       sides[pick_home]["p"]))
    if spread is not None:
        # Projected margin from the final pick (as on the dashboard) against the spread
        margin = SPREAD_SIGMA * norm.ppf(min(max(p_home, 1e-6), 1 - 1e-6))
        if margin != spread:
            home = margin > spread
            p_cover = float(norm.cdf((margin - spread) / SPREAD_SIGMA))
            out.append(bet("pick_spread", home, "spread",
                           sides[home]["spread_odds"] or DEFAULT_SPREAD_ODDS,
                           p_cover if home else 1 - p_cover,
                           line=-spread if home else spread))
    edges = {h: s["p"] - breakeven(s["ml"]) for h, s in sides.items() if s["ml"] is not None}
    if edges:
        home = max(edges, key=edges.get)
        if edges[home] >= VALUE_EDGE:
            out.append(bet("value_ml", home, "moneyline", sides[home]["ml"], sides[home]["p"]))
    return out


def describe(b: dict) -> str:
    line = "" if b["line"] is None else f" {b['line']:+g}"
    return f"{STRATEGIES[b['strategy']]}: {b['team']}{line} ({b['odds']:+d})"


def settle(ledger: pl.DataFrame, schedule: pl.DataFrame, now: datetime) -> pl.DataFrame:
    """Grade open bets whose games have a final score (ties push a moneyline bet)."""
    scores = schedule.filter(pl.col("result").is_not_null()).select(
        "game_id", "home_score", "away_score")
    open_ = ledger.filter(pl.col("status") == "open").drop("team_score", "opp_score")
    done = open_.join(scores, on="game_id", how="inner")
    if done.height == 0:
        return ledger
    done = done.with_columns(
        team_score=pl.when("is_home").then("home_score").otherwise("away_score"),
        opp_score=pl.when("is_home").then("away_score").otherwise("home_score"),
    ).with_columns(
        cover=(pl.col("team_score") - pl.col("opp_score")).cast(pl.Float64)
        + pl.col("line").fill_null(0.0),
    ).with_columns(
        status=pl.when(pl.col("cover") > 0).then(pl.lit("won"))
        .when(pl.col("cover") < 0).then(pl.lit("lost")).otherwise(pl.lit("push")),
    ).with_columns(
        profit=pl.when(pl.col("status") == "won").then(
            pl.when(pl.col("odds") > 0).then(pl.col("stake") * pl.col("odds") / 100)
            .otherwise(pl.col("stake") * 100 / -pl.col("odds")))
        .when(pl.col("status") == "lost").then(-pl.col("stake")).otherwise(0.0),
        settled_at=pl.lit(now, UTC_TS),
    ).select(list(SCHEMA))
    return pl.concat([ledger.filter(~pl.col("bet_id").is_in(done["bet_id"].implode())),
                      done.cast(SCHEMA)]).sort("kickoff", "bet_id")


def _predictions() -> pl.DataFrame:
    """Latest walk-forward stats model and blend fit per game, from the full database or
    (on the GitHub runner) the published dashboard copy."""
    import duckdb

    from src.export import APP_DB

    path = DB_PATH if DB_PATH.exists() else APP_DB
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute("SELECT * FROM game_predictions").pl()
    finally:
        con.close()


def blend_coefs(preds: pl.DataFrame) -> pl.DataFrame:
    """game_id, p_logit and the blend coefficients to use for it: its own week's fit, or the
    latest fit for games that had no line when the picks were last refreshed."""
    cols = BLEND_COEFS
    fitted = preds.filter(pl.col(cols[0]).is_not_null()).sort("game_date")
    latest = fitted.tail(1).select(cols).row(0, named=True) if fitted.height else {}
    return preds.select("game_id", "p_logit", *[
        pl.col(c).fill_null(latest.get(c)) for c in cols])


def _schedule() -> pl.DataFrame:
    """This season's schedule with the latest lines and scores. nflreadpy is imported here,
    not at the top, because the hosted dashboard imports this module without it."""
    import nflreadpy as nfl

    season = nfl.get_current_season()
    return nfl.load_schedules([season]).filter(pl.col("game_type").is_not_null())


def empty() -> pl.DataFrame:
    return pl.DataFrame(schema=SCHEMA)


def load() -> pl.DataFrame:
    return pl.read_parquet(LEDGER).cast(SCHEMA) if LEDGER.exists() else empty()


def place(schedule: pl.DataFrame, preds: pl.DataFrame, ledger: pl.DataFrame, now: datetime,
          backfill: bool = False) -> pl.DataFrame:
    """New bets: live ones for games kicking off within LEAD (+GRACE) of now, and with
    backfill=True, closing-line bets for finished games that have none."""
    have = set(ledger["game_id"].to_list())
    games = schedule.join(blend_coefs(preds), on="game_id", how="inner").filter(
        pl.col("p_logit").is_not_null() & ~pl.col("game_id").is_in(list(have)))
    rows = []
    for g in games.iter_rows(named=True):
        kick = kickoff_utc(g["gameday"], g["gametime"])
        coef = g if g["blend_coef_intercept"] is not None else None
        if g["result"] is None and now < kick <= now + LEAD + GRACE:
            placed_at, placement = now, "live"
        elif backfill and g["result"] is not None and kick < now:
            placed_at, placement = None, "backfill"
        else:
            continue
        p, source = home_prob(g["p_logit"], g["spread_line"], coef)
        new = make_bets(g, p, source, placed_at, placement)
        rows += new
        if placement == "live":
            log.info("%s: %s", g["game_id"], "; ".join(map(describe, new)) or "no lines yet")
    if not rows:
        return ledger
    return pl.concat([ledger, pl.DataFrame(rows, schema=SCHEMA)]).sort("kickoff", "bet_id")


def run(now: datetime | None = None, backfill: bool | None = None) -> bool:
    """Place due bets and settle finished ones; returns whether the ledger changed. A new
    ledger is backfilled with this season's finished games (backfill=None)."""
    now = now or datetime.now(UTC)
    ledger = load()
    if backfill is None:
        backfill = not LEDGER.exists()
    preds = _predictions()
    if not set(BLEND_COEFS) <= set(preds.columns):
        # Built before the blend coefficients were stored: bets would ignore the line
        log.warning("game_predictions has no blend coefficients; run `make refresh` first")
        return False
    schedule = _schedule()
    before = ledger.height, ledger.filter(pl.col("status") == "open").height
    ledger = place(schedule, preds, ledger, now, backfill=backfill)
    ledger = settle(ledger, schedule, now)
    after = ledger.height, ledger.filter(pl.col("status") == "open").height
    if after == before and LEDGER.exists():
        log.info("no bets due; %d open", after[1])
        return False
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_parquet(LEDGER)
    log.info("ledger: %d bets (%d new), %d open", after[0], after[0] - before[0], after[1])
    return True
