"""End-to-end leakage check on the real database: features for games on date D must be
identical whether or not games on/after D exist in the input."""

from datetime import date

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.config import DB_PATH
from src.features.team import METRICS, RollingParams, pregame_features, rolling_state, season_priors

pytestmark = pytest.mark.skipif(not DB_PATH.exists(), reason="run `make ingest train` first")


@pytest.fixture(scope="module")
def team_games():
    import duckdb

    con = duckdb.connect(str(DB_PATH), read_only=True)
    try:
        return con.execute("SELECT * FROM team_game_stats").pl()
    except duckdb.CatalogException:
        pytest.skip("team_game_stats not built")
    finally:
        con.close()


@pytest.mark.parametrize("cutoff", [date(2019, 9, 8), date(2021, 11, 14), date(2024, 1, 7)])
def test_features_unchanged_by_truncating_future(team_games, cutoff):
    p = RollingParams()
    on_day = team_games.filter(pl.col("game_date") == cutoff)
    assert on_day.height > 0
    sched = on_day.select("game_id", "team", "season", "game_date")

    def feats(games):
        priors = season_priors(games, METRICS, p)
        return pregame_features(sched, rolling_state(games, priors, METRICS, p), priors,
                                METRICS).sort("game_id", "team")

    full = feats(team_games)
    past = feats(team_games.filter(pl.col("game_date") < cutoff))
    assert_frame_equal(full, past, check_exact=False, rel_tol=1e-12, abs_tol=1e-12)
