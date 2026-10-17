import duckdb
import polars as pl
import pytest

from src.features.receiving import build_receiving


def _play(**kw):
    base = dict(
        season=2023, season_type="REG", game_id="g1", posteam="AAA", receiver_player_id="wr1",
        pass_attempt=1, sack=0, two_point_attempt=0, air_yards=10.0, epa=0.5,
        complete_pass=1, cp=0.6, yards_after_catch=4.0, xyac_mean_yardage=3.0,
        pass_location="left", yardline_100=50,
    )
    return base | kw


@pytest.fixture
def con():
    plays = [
        # wr1: 3 targets in g1 (one in red zone, one to the middle), 1 target in g2
        _play(air_yards=10.0, cp=0.6, complete_pass=1),
        _play(air_yards=20.0, cp=0.4, complete_pass=0, yards_after_catch=None,
              pass_location="middle", epa=-1.0),
        _play(air_yards=0.0, yardline_100=10, cp=0.9, complete_pass=1, yards_after_catch=6.0,
              xyac_mean_yardage=4.0),
        _play(game_id="g2", air_yards=30.0, cp=0.3, complete_pass=0, yards_after_catch=None),
        # teammate wr2 in g1 only: 1 normal target + 1 red-zone target
        _play(receiver_player_id="wr2", air_yards=5.0),
        _play(receiver_player_id="wr2", air_yards=2.0, yardline_100=5),
        # team g3: wr1 did not play -> must not enter his denominators
        _play(game_id="g3", receiver_player_id="wr2", air_yards=40.0),
        # excluded plays: sack, two-point try, postseason
        _play(sack=1, receiver_player_id="wr1", air_yards=99.0),
        _play(two_point_attempt=1, air_yards=99.0, yardline_100=2),
        _play(season_type="POST", game_id="p1", air_yards=99.0),
    ]
    c = duckdb.connect()
    c.register("_pbp", pl.DataFrame(plays).to_arrow())
    c.execute("CREATE TABLE pbp AS SELECT * FROM _pbp")
    c.execute("""CREATE TABLE ps AS SELECT * FROM (VALUES
        (2023, 'REG', 'g1', 'AAA', 'wr1', 'Receiver One'),
        (2023, 'REG', 'g2', 'AAA', 'wr1', 'Receiver One'),
        (2023, 'REG', 'g1', 'AAA', 'wr2', 'Receiver Two'),
        (2023, 'REG', 'g3', 'AAA', 'wr2', 'Receiver Two'))
        t(season, season_type, game_id, team, player_id, player_display_name)""")
    c.execute("""CREATE TABLE ngs AS SELECT * FROM (VALUES
        (2023, 'REG', 0, 'wr1', 3.0, 6.0, 4),
        (2023, 'REG', 1, 'wr2', 2.0, 5.0, 2),
        (2023, 'REG', 2, 'wr2', 4.0, 7.0, 1))
        t(season, season_type, week, player_gsis_id, avg_separation, avg_cushion, targets)""")
    c.execute("""CREATE TABLE ros AS SELECT * FROM (VALUES (2023, 'wr1', 'WR'), (2023, 'wr2', 'WR'))
        t(season, gsis_id, position)""")
    build_receiving(c, out="feat", pbp="pbp", player_stats="ps", ngs="ngs", rosters="ros",
                    min_targets=1)
    return c


def row(con, pid):
    return con.execute("SELECT * FROM feat WHERE player_id = ?", [pid]).pl().row(0, named=True)


def test_excludes_sacks_two_point_and_postseason(con):
    r = row(con, "wr1")
    assert r["targets"] == 4
    assert r["adot"] == pytest.approx((10 + 20 + 0 + 30) / 4)


def test_shares_use_only_games_player_appeared_in(con):
    r = row(con, "wr1")
    # team targets in g1 (5) + g2 (1); g3 excluded because wr1 did not play
    assert r["games"] == 2
    assert r["target_share"] == pytest.approx(4 / 6)
    assert r["air_yards_share"] == pytest.approx(60 / (10 + 20 + 0 + 5 + 2 + 30))
    assert r["rz_target_share"] == pytest.approx(1 / 2)


def test_rate_features(con):
    r = row(con, "wr1")
    assert r["middle_target_rate"] == pytest.approx(1 / 4)
    # CROE = mean(complete - cp) over targets
    assert r["croe"] == pytest.approx(((1 - .6) + (0 - .4) + (1 - .9) + (0 - .3)) / 4)
    # YACOE over receptions only
    assert r["yacoe"] == pytest.approx(((4 - 3) + (6 - 4)) / 2)
    assert r["epa_per_target"] == pytest.approx((0.5 - 1.0 + 0.5 + 0.5) / 4)


def test_ngs_season_row_then_weekly_fallback(con):
    assert row(con, "wr1")["avg_separation"] == pytest.approx(3.0)
    # wr2 has no week-0 row: target-weighted weekly mean
    assert row(con, "wr2")["avg_separation"] == pytest.approx((2 * 2 + 4 * 1) / 3)


def test_min_targets_filter(con):
    build_receiving(con, out="feat3", pbp="pbp", player_stats="ps", ngs="ngs", rosters="ros",
                    min_targets=4)
    ids = con.execute("SELECT player_id FROM feat3").fetchall()
    assert ids == [("wr1",)]
