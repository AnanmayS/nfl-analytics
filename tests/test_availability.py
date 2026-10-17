"""Availability features: roles come only from earlier games, statuses map to sit chances."""

import duckdb
import polars as pl
import pytest

from src.features.availability import build_availability


@pytest.fixture
def con():
    c = duckdb.connect()
    sched = pl.DataFrame({
        "game_id": ["g1", "g2", "g3", "g4", "g5"],
        "season": [2026] * 5, "week": [1, 2, 3, 4, 5],
        "gameday": ["2026-09-10", "2026-09-17", "2026-09-24", "2026-10-01", "2026-10-08"],
        "home_team": ["KC"] * 5, "away_team": ["DEN", "LV", "LAC", "OAK", "NYJ"],
    })
    snaps = []
    for gid, wk in [("g1", 1), ("g2", 2), ("g3", 3), ("g4", 4)]:
        snaps += [
            # starter QB plays every snap until he's hurt after week 2
            dict(game_id=gid, season=2026, week=wk, team="KC", pfr_player_id="qb1",
                 player="Starter QB", position="QB",
                 offense_pct=1.0 if wk < 3 else 0.0, defense_pct=0.0),
            dict(game_id=gid, season=2026, week=wk, team="KC", pfr_player_id="qb2",
                 player="Backup QB", position="QB",
                 offense_pct=0.0 if wk < 3 else 1.0, defense_pct=0.0),
            dict(game_id=gid, season=2026, week=wk, team="KC", pfr_player_id="wr1",
                 player="Top WR", position="WR", offense_pct=0.8, defense_pct=0.0),
            dict(game_id=gid, season=2026, week=wk, team="KC", pfr_player_id="cb1",
                 player="Top CB", position="CB", offense_pct=0.0, defense_pct=1.0),
        ]
    c.register("_s", pl.DataFrame(snaps).with_columns(
        pl.col("offense_pct").cast(pl.Float64), pl.col("defense_pct").cast(pl.Float64)).to_arrow())
    c.execute("CREATE TABLE snaps AS SELECT * FROM _s")
    c.register("_sch", sched.to_arrow())
    c.execute("CREATE TABLE sched AS SELECT * FROM _sch")
    c.execute("""CREATE TABLE players AS SELECT * FROM (VALUES
        ('qb1', 'G-QB1'), ('qb2', 'G-QB2'), ('wr1', 'G-WR1'), ('cb1', 'G-CB1'))
        t(pfr_id, gsis_id)""")
    c.execute("CREATE TABLE rosters AS SELECT * FROM players")
    c.execute("""CREATE TABLE inj AS SELECT * FROM (VALUES
        (2026, 2, 'KC', 'G-WR1', 'Questionable'),   -- week 2: WR questionable
        (2026, 3, 'KC', 'G-QB1', 'Out'),            -- week 3: starter newly out
        (2026, 3, 'KC', 'G-CB1', 'Doubtful'),
        (2026, 5, 'KC', 'G-QB1', 'Out'))            -- week 5: still out, backup has taken over
        t(season, week, team, gsis_id, report_status)""")
    build_availability(c, schedules="sched", snaps="snaps", injuries="inj",
                       players="players", rosters="rosters")
    return c


def row(con, gid, team="KC"):
    return con.execute("SELECT * FROM game_availability WHERE game_id = ? AND team = ?",
                       [gid, team]).pl().row(0, named=True)


def test_week_one_has_no_history_and_no_report(con):
    r = row(con, "g1")
    assert r["qb_out"] == 0 and r["off_missing"] == 0 and not r["report_available"]


def test_questionable_weighted_by_recent_role(con):
    r = row(con, "g2")
    assert r["report_available"]
    assert r["off_missing"] == pytest.approx(0.2 * 0.8)
    assert r["qb_out"] == 0


def test_new_qb_absence_is_flagged(con):
    r = row(con, "g3")
    assert r["starting_qb"] == "Starter QB"
    assert r["qb_out"] == pytest.approx(1.0)
    assert r["def_missing"] == pytest.approx(0.9 * 1.0)
    assert "QB Starter QB (Out)" in r["key_absences"]


def test_long_absence_already_in_recent_form(con):
    # Last 3 games before week 5 are weeks 2-4: QB1 1/3 of snaps, backup 2/3. The backup is
    # now the starter, so QB1 still being Out isn't a new loss for the team.
    r = row(con, "g5")
    assert r["starting_qb"] == "Backup QB"
    assert r["qb_out"] == 0


def test_relocated_team_codes_are_normalized(con):
    assert row(con, "g4", "LV")["qb_out"] == 0
