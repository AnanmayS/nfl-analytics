"""Opening-line matching and the Tuesday-vs-opening-line evaluation."""

from datetime import date

import polars as pl
import pytest

from src.ingest.odds import match_games
from src.models.early import evaluate

TEAMS = pl.DataFrame({"team_abbr": ["KC", "LV", "OAK", "LA", "LAR"],
                      "team_name": ["Kansas City Chiefs", "Las Vegas Raiders", "Oakland Raiders",
                                    "Los Angeles Rams", "Los Angeles Rams"]})
SCHED = pl.DataFrame({
    "game_id": ["g1", "g2", "g3"],
    "game_date": [date(2019, 9, 8), date(2023, 10, 1), date(2024, 11, 3)],
    "home_team": ["KC", "LA", "LV"], "away_team": ["LV", "KC", "KC"],
})


def lines(rows):
    base = dict(open_spread=0.0, book_close_spread=0.0, p_open_ml=0.5, p_close_ml=0.5,
                open_total=45.0, close_total=45.0)
    return pl.DataFrame([base | r for r in rows])


def test_matches_old_names_date_shift_and_swapped_neutral_games():
    ln = lines([
        # old team name, listed a day late (time zone)
        dict(date=date(2019, 9, 9), home_name="Kansas City Chiefs", away_name="Oakland Raiders",
             open_spread=7.0),
        # Rams: nflverse has two codes; the schedule's (LA) must win
        dict(date=date(2023, 10, 1), home_name="Los Angeles Rams",
             away_name="Kansas City Chiefs", open_spread=-3.0),
        # neutral-site game listed with teams swapped -> line flipped
        dict(date=date(2024, 11, 3), home_name="Kansas City Chiefs",
             away_name="Las Vegas Raiders", open_spread=4.0, p_open_ml=0.7),
    ])
    m = match_games(ln, SCHED, TEAMS).sort("game_id")
    assert m["game_id"].to_list() == ["g1", "g2", "g3"]
    assert m["open_spread"].to_list() == [7.0, -3.0, -4.0]
    assert m.filter(pl.col("game_id") == "g3")["p_open_ml"][0] == pytest.approx(0.3)


def test_evaluate_counts_disagreements_and_line_moves():
    out = pl.DataFrame({
        "season": [2024] * 4,
        "home_win": [1, 0, 1, 0],
        "result": [7.0, -3.0, 2.0, -10.0],
        "p_tuesday": [0.6, 0.4, 0.55, 0.45],   # right, right, right, right
        "p_tue_stats": [0.75, 0.3, 0.6, 0.35],
        "p_open": [0.55, 0.55, 0.45, 0.4],      # right, wrong, wrong, right
        "p_vegas": [0.6, 0.45, 0.5, 0.4],
        "open_spread": [1.0, 1.0, -1.0, -3.0],
        "spread_line": [3.0, -1.0, 0.5, -5.0],  # every line moved toward the model
    })
    summary, by_season = evaluate(out)
    get = dict(zip(summary["metric"], summary["value"], strict=True))
    assert get["Tuesday pick: picks right"] == 1.0
    assert get["Opening line: picks right"] == 0.5
    assert get["Games where only the Tuesday pick was right"] == 2
    assert get["Line moved toward our Tuesday number (disagree by 3+ pts)"] == 1.0
    assert by_season["tuesday_pick"][0] == 1.0
