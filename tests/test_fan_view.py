from datetime import date

import polars as pl
import pytest

from src.analysis.fan_view import (
    confidence_check,
    confidence_label,
    describe_archetype,
    league_picks,
    pick_summary,
    spread_text,
    team_games,
    vegas_favorite_text,
)


@pytest.fixture
def preds():
    return pl.DataFrame({
        "game_id": ["g1", "g2", "g3"],
        "season": [2026] * 3,
        "week": [1, 2, 3],
        "game_type": ["REG"] * 3,
        "game_date": [date(2026, 9, 10), date(2026, 9, 17), date(2026, 9, 24)],
        "home_team": ["KC", "BUF", "KC"],
        "away_team": ["BUF", "KC", "DEN"],
        "p_final": [0.70, 0.60, 0.55],
        "p_logit": [0.66, 0.62, 0.55],
        "p_vegas": [0.65, 0.55, None],
        "spread_line": [3.0, -2.5, None],
        "home_score": [20, 30, None],
        "away_score": [24, 10, None],
        "result": [-4.0, 20.0, None],
    })


def test_team_games_flips_away_games(preds):
    g = team_games(preds, "KC").sort("week")
    assert g["is_home"].to_list() == [True, False, True]
    assert g["opponent"].to_list() == ["BUF", "BUF", "DEN"]
    assert g["win_prob"].to_list() == pytest.approx([0.70, 0.40, 0.55])
    assert g["vegas_spread"].to_list()[:2] == [3.0, 2.5]  # KC favored by 2.5 at BUF
    assert g["points_for"].to_list()[:2] == [20, 10]
    assert g["margin"].to_list()[:2] == [-4.0, -20.0]


def test_projected_margin_agrees_with_win_prob(preds):
    g = team_games(preds, "KC")
    assert ((g["win_prob"] > 0.5) == (g["projected_margin"] > 0)).all()


def test_correctness_and_status(preds):
    g = team_games(preds, "KC").sort("week")
    assert g["status"].to_list() == ["final", "final", "upcoming"]
    assert g["won"].to_list() == [False, False, None]
    assert g["model_correct"].to_list()[:2] == [False, True]  # 70% pick lost; 40% pick right
    assert g["vegas_correct"].to_list()[:2] == [False, False]
    assert pick_summary(g) == {"games": 2, "model_right": 1, "vegas_games": 2, "vegas_right": 0}


def test_league_picks_and_confidence_check(preds):
    picks = league_picks(preds)
    assert picks.height == 2
    assert picks["model_conf"].to_list() == pytest.approx([0.70, 0.60])
    cc = confidence_check(picks)
    assert cc["bucket"].to_list() == ["60–70%", "70–80%"]
    assert cc["actual"].to_list() == [1.0, 0.0]


def test_plain_language_helpers():
    assert confidence_label(0.52) == "Toss-up"
    assert confidence_label(0.80) == "Strong favorite"
    assert confidence_label(0.30) == "Underdog"
    assert spread_text(3.5, "Chiefs") == "Chiefs favored by 3.5"
    assert spread_text(-7, "Chiefs") == "Chiefs underdog by 7"
    assert spread_text(None, "Chiefs") == "No line yet"
    assert vegas_favorite_text(-2.5, "Broncos", "Seahawks") == "Seahawks by 2.5"
    assert vegas_favorite_text(3, "Broncos", "Seahawks") == "Broncos by 3"
    traits = describe_archetype({"z_adot": 1.2, "z_cpoe": -0.4, "z_sack_rate": 0.1},
                                ["adot", "cpoe", "sack_rate"], "QB")
    assert traits == ["Much more than typical: average throw depth",
                      "Less than typical: accuracy vs. expected"]


def _tiers(home_vals: dict, away_vals: dict) -> pl.DataFrame:
    from src.analysis.fan_view import FACTORS, metric_tiers
    others = [{"home_team": f"H{i}", "away_team": f"A{i}",
               **{f"{s}_{m}": 0.0 for s in ("home", "away") for m in FACTORS}}
              for i in range(4)]
    game = {"home_team": "KC", "away_team": "DEN",
            **{f"home_{m}": home_vals.get(m, 0.0) for m in FACTORS},
            **{f"away_{m}": away_vals.get(m, 0.0) for m in FACTORS}}
    df = pl.DataFrame([game, *others]).with_columns(season=pl.lit(2026), week=pl.lit(5))
    return metric_tiers(df)


def test_metric_tiers_orientation():
    t = _tiers({"def_pass_epa": -0.3, "off_pass_epa": 0.3},
               {"def_pass_epa": 0.3, "off_pass_epa": -0.3})
    get = {(r["team"], r["metric"]): r["pct"] for r in t.iter_rows(named=True)}
    # lower EPA allowed is better defense -> KC tops both
    assert get[("KC", "def_pass_epa")] > get[("DEN", "def_pass_epa")]
    assert get[("KC", "off_pass_epa")] > get[("DEN", "off_pass_epa")]


def test_game_reasons_direction_and_consistency():
    from src.analysis.fan_view import game_reasons
    t = _tiers({"off_pass_epa": 0.3, "def_rush_epa": 0.2}, {"off_pass_epa": -0.3,
                                                            "def_rush_epa": -0.2})
    game = {"season": 2026, "week": 5, "home_team": "KC", "away_team": "DEN",
            "contrib_d_off_pass_epa": 0.4,     # favors KC, and KC is better -> kept
            "contrib_d_def_rush_epa": 0.2,     # favors KC, but KC's run D is worse -> dropped
            "contrib_intercept": 0.1, "contrib_neutral": 0.0, "contrib_div_game": 0.0,
            "neutral": 0, "contrib_d_rest": -0.05, "home_rest": 6, "away_rest": 10}
    r = game_reasons(game, t, {"KC": "Chiefs", "DEN": "Broncos"})
    texts = [x["text"] for x in r]
    assert r[0]["favors"] == "KC" and texts[0].startswith("Chiefs' passing offense")
    assert not any("run defense" in x for x in texts)
    assert "Home-field advantage for Chiefs" in texts
    assert any(x["favors"] == "DEN" and "better rested (10 days vs. 6)" in x["text"]
               for x in r)


def test_team_identity_labels():
    from src.analysis.fan_view import team_identity
    pcts = {"off_pass_epa": 30, "off_rush_epa": 35, "off_success_rate": 30,
            "def_pass_epa": 90, "def_rush_epa": 70, "def_success_rate": 75,
            "def_pressure_rate": 20}
    ident = team_identity(pcts, 0.8, [
        {"pos": "QB", "player_name": "A", "archetype": "Conservative Accurate"},
        {"pos": "WR", "player_name": "B", "archetype": None}])
    assert ident["label"] == "Pass-first offense, carried by its defense"
    assert ident["strengths"][0] == "pass defense (one of the league's best)"
    assert ident["weakness"] == ["pass rush (below average)"]
    assert ident["players"] == ["QB A (Conservative Accurate)", "WR B"]


def test_defense_as_a_whole_reason():
    from src.analysis.fan_view import game_reasons
    # DEN's defense is better overall and on the run; KC's pass rush is better
    t = _tiers({"def_epa_play": 0.10, "def_rush_epa": 0.10, "def_pressure_rate": 0.40},
               {"def_epa_play": -0.10, "def_rush_epa": -0.15, "def_pressure_rate": 0.10})
    game = {"season": 2026, "week": 5, "home_team": "KC", "away_team": "DEN",
            "contrib_d_def_rush_epa": -0.30, "contrib_d_def_pressure_rate": 0.05,
            "contrib_d_def_blitz_rate": -0.02, "neutral": 1}
    r = game_reasons(game, t, {"KC": "Chiefs", "DEN": "Broncos"})
    assert len(r) == 1  # one line for the whole defense, not three
    assert r[0]["favors"] == "DEN" and r[0]["push"] == pytest.approx(0.27)
    assert r[0]["text"] == ("Broncos' defense as a whole (one of the league's best) vs. "
                            "Chiefs' (one of the league's worst); biggest gap: run defense")


def test_identity_reports_defense():
    from src.analysis.fan_view import team_identity
    ident = team_identity({"def_epa_play": 85, "off_epa_play": 40, "def_pass_epa": 90,
                           "def_rush_epa": 50}, 0.5, [], "Lockdown Sticky-Coverage Defense")
    assert ident["label"] == "Balanced play-calling, carried by its defense"
    assert ident["defense"] == {"type": "Lockdown Sticky-Coverage Defense",
                                "overall": "one of the league's best", "best_at": "pass defense",
                                "worst_at": "run defense"}


def test_qb_change_reason():
    from src.analysis.fan_view import game_reasons
    t = _tiers({}, {})
    game = {"season": 2026, "week": 5, "home_team": "KC", "away_team": "DEN", "neutral": 1,
            "contrib_d_qb_change": 0.05, "contrib_d_qb_delta": 0.30,
            "home_qb_change": 0.0, "away_qb_change": 1.0, "away_qb_delta": -0.15,
            "away_expected_qb": "Jarrett Stidham", "away_recent_qb": "Bo Nix"}
    r = game_reasons(game, t, {"KC": "Chiefs", "DEN": "Broncos"})
    assert r == [{"favors": "KC", "push": pytest.approx(0.35),
                  "text": "Broncos start Jarrett Stidham instead of Bo Nix (a big downgrade)"}]
    # push pointing the other way contradicts the sentence -> dropped
    game.update(contrib_d_qb_delta=-0.30)
    assert game_reasons(game, t) == []
