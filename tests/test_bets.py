"""Paper bets: placed only in the hour before kickoff, once per game, and graded correctly."""

from datetime import UTC, date, datetime, timedelta

import polars as pl
import pytest

from src import bets as b

NOW = datetime(2026, 10, 25, 16, 0, tzinfo=UTC)  # noon Eastern


def game(gid, gametime="13:00", gameday="2026-10-25", spread=3.0, hml=-160, aml=135,
         hso=-110, aso=-110, home_score=None, away_score=None):
    result = None if home_score is None else home_score - away_score
    return dict(game_id=gid, season=2026, week=7, game_type="REG", gameday=gameday,
                gametime=gametime, home_team="H" + gid, away_team="A" + gid,
                spread_line=spread, home_moneyline=hml, away_moneyline=aml,
                home_spread_odds=hso, away_spread_odds=aso, home_score=home_score,
                away_score=away_score, result=result)


def preds(ids, p=0.6, coef=(0.0, 0.0, 1.0)):
    """Blend that ignores the stats model and passes the line's probability through."""
    return pl.DataFrame({"game_id": ids, "p_logit": [p] * len(ids),
                         "game_date": [date(2026, 10, 25)] * len(ids),
                         "blend_coef_intercept": [coef[0]] * len(ids),
                         "blend_coef_stats_logit": [coef[1]] * len(ids),
                         "blend_coef_vegas_logit": [coef[2]] * len(ids)})


def test_odds_math():
    assert b.breakeven(-110) == pytest.approx(110 / 210)
    assert b.breakeven(150) == pytest.approx(0.4)
    assert b.win_profit(150) == 150 and b.win_profit(-200) == 50


def test_kickoff_is_eastern_across_dst():
    assert b.kickoff_utc("2026-10-25", "13:00") == datetime(2026, 10, 25, 17, tzinfo=UTC)
    assert b.kickoff_utc("2026-11-08", "13:00") == datetime(2026, 11, 8, 18, tzinfo=UTC)


def test_bets_placed_only_within_the_hour_before_kickoff():
    sched = pl.DataFrame([
        game("soon", "12:40"),  # 40 min out -> bet
        game("later", "16:25"),  # 4+ hours out -> wait
        game("started", "11:30"),  # already kicked off -> missed, never bet late
        game("done", "13:00", gameday="2026-10-19", home_score=24, away_score=17),
    ])
    ledger = b.place(sched, preds(sched["game_id"].to_list()), b.empty(), NOW)
    assert set(ledger["game_id"]) == {"soon"}
    assert (ledger["placement"] == "live").all() and (ledger["placed_at"] == NOW).all()
    # running again (next 15-minute run) adds nothing
    assert b.place(sched, preds(sched["game_id"].to_list()), ledger,
                   NOW + timedelta(minutes=15)).height == ledger.height


def test_backfill_uses_finished_games_only():
    sched = pl.DataFrame([game("done", gameday="2026-10-19", home_score=24, away_score=17),
                          game("later", "16:25")])
    ledger = b.place(sched, preds(["done", "later"]), b.empty(), NOW, backfill=True)
    assert set(ledger["game_id"]) == {"done"}
    assert (ledger["placement"] == "backfill").all() and ledger["placed_at"].is_null().all()


def test_strategies_and_settlement():
    # Home favored by 3; the blend stretches the line (coef 1.3), so our margin is > 3
    g = game("g", spread=3.0, hml=-160, aml=135)
    p, source = b.home_prob(0.5, 3.0, dict(blend_coef_intercept=0.0,
                                           blend_coef_stats_logit=0.0,
                                           blend_coef_vegas_logit=1.3))
    assert source == "stats + line" and p > 0.5
    bets = {x["strategy"]: x for x in b.make_bets(g, p, source, NOW, "live")}
    assert bets["pick_ml"]["team"] == "Hg" and bets["pick_ml"]["odds"] == -160
    assert bets["pick_spread"]["team"] == "Hg" and bets["pick_spread"]["line"] == -3.0
    assert "value_ml" not in bets  # stretched line still short of -160's 61.5% break-even

    ledger = pl.DataFrame(list(bets.values()), schema=b.SCHEMA)
    for (hs, as_), ml, spread in [((24, 21), "won", "push"), ((27, 20), "won", "won"),
                                  ((20, 20), "push", "lost"), ((17, 20), "lost", "lost")]:
        sched = pl.DataFrame([game("g", home_score=hs, away_score=as_)])
        s = b.settle(ledger, sched, NOW).sort("strategy")
        got = dict(zip(s["strategy"], s["status"], strict=True))
        assert got == {"pick_ml": ml, "pick_spread": spread}
        profit = dict(zip(s["strategy"], s["profit"], strict=True))
        assert profit["pick_ml"] == {"won": 62.5, "lost": -100.0, "push": 0.0}[ml]
        assert profit["pick_spread"] == {"won": pytest.approx(100 / 1.1), "lost": -100.0,
                                         "push": 0.0}[spread]


def test_value_bet_takes_the_underdog_when_its_price_is_too_long():
    g = game("g", spread=3.0, hml=-160, aml=200)  # +200 needs 33.3%; we give the dog 40%
    bets = {x["strategy"]: x for x in b.make_bets(g, 0.6, "stats + line", NOW, "live")}
    assert bets["pick_ml"]["team"] == "Hg"
    assert bets["value_ml"]["team"] == "Ag" and bets["value_ml"]["edge"] == pytest.approx(
        0.4 - 1 / 3)


def test_no_line_means_stats_only_and_no_spread_bet():
    g = game("g", spread=None, hml=None, aml=None)
    p, source = b.home_prob(0.62, None, None)
    assert (p, source) == (0.62, "stats only")
    assert b.make_bets(g, p, source, NOW, "live") == []
