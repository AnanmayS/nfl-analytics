"""Expected starting QB and QB ratings: no future data, correct fallbacks."""

from datetime import date

import polars as pl
import pytest

from src.features.quarterbacks import (
    DECAY,
    PRIOR,
    RATING_K,
    expected_starters,
    qb_states,
    rating_asof,
)

QB_GAMES = pl.DataFrame({
    "qb_id": ["star", "star", "star", "backup"],
    "game_id": ["a", "b", "c", "c"],
    "season": [2025, 2025, 2026, 2026],
    "game_date": [date(2025, 9, 7), date(2025, 9, 14), date(2026, 9, 10), date(2026, 9, 10)],
    "epa": [30.0, 10.0, 20.0, -5.0],
    "dropbacks": [40, 40, 40, 10],
})


def lookup(qb, d, season):
    return pl.DataFrame({"qb": [qb], "game_date": [d], "season": [season]})


def test_rating_uses_only_earlier_games_and_decays_old_seasons():
    states = qb_states(QB_GAMES)
    # Before his 2025 week 2 game: only game a counts
    r = rating_asof(lookup("star", date(2025, 9, 14), 2025), states, "qb")[0]
    assert r == pytest.approx((30 + RATING_K * PRIOR) / (40 + RATING_K))
    # Before 2026 game c: all of 2025, decayed one season
    r = rating_asof(lookup("star", date(2026, 9, 10), 2026), states, "qb")[0]
    assert r == pytest.approx((DECAY * 40 + RATING_K * PRIOR) / (DECAY * 80 + RATING_K))


def test_unknown_or_new_qb_is_replacement_level():
    states = qb_states(QB_GAMES)
    assert rating_asof(lookup("rookie", date(2026, 9, 10), 2026), states, "qb")[0] == \
        pytest.approx(PRIOR)
    assert rating_asof(lookup(None, date(2026, 9, 10), 2026), states, "qb")[0] == \
        pytest.approx(PRIOR)


def game(**kw):
    base = dict(game_id="g", team="CIN", season=2026, week=3, game_date=date(2026, 9, 24),
                listed_qb_id=None, listed_qb=None, recent_qb_id="star", recent_qb="Star",
                backup_qb_id="backup", backup_qb="Backup", recent_qb_sit=None)
    return pl.DataFrame([base | kw])


def roster(status, week=3):
    return pl.DataFrame({"season": [2026], "week": [week], "team": ["CIN"],
                         "player_id": ["star"], "status": [status]})


def test_listed_starter_wins():
    out = expected_starters(game(listed_qb_id="backup", listed_qb="Backup"), roster("ACT"),
                            qb_states(QB_GAMES)).row(0, named=True)
    assert out["source"] == "listed" and out["expected_qb"] == "Backup"
    assert out["qb_change"] == 1 and out["qb_delta"] < 0


def test_injured_reserve_falls_back_to_backup():
    # Not on the injury report at all, but on injured reserve in an earlier roster week
    out = expected_starters(game(), roster("RES", week=2), qb_states(QB_GAMES)).row(0, named=True)
    assert out["source"] == "recent starter unavailable" and out["expected_qb"] == "Backup"


def test_injury_report_out_falls_back_to_backup():
    out = expected_starters(game(recent_qb_sit=1.0), roster("ACT"),
                            qb_states(QB_GAMES)).row(0, named=True)
    assert out["expected_qb"] == "Backup"


def test_healthy_recent_starter_means_no_change():
    out = expected_starters(game(recent_qb_sit=0.2), roster("ACT"),
                            qb_states(QB_GAMES)).row(0, named=True)
    assert out["source"] == "recent starter"
    assert out["qb_change"] == 0 and out["qb_delta"] == 0
