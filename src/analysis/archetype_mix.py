"""Part 3: roster archetype mix -> team offensive efficiency.

Tables written to DuckDB:
  player_archetypes        every labeled player-season (all positions)
  player_snaps             game-level offensive snaps keyed by gsis id
  team_archetype_mix       team-season snap share by position group x archetype (descriptive)
  team_season_analysis     team-season EPA/play, talent controls and wide mix shares
  archetype_mix_tests      nested-model comparison (talent -> + archetypes -> + QBxWR)
  qb_wr_pairings           mean talent-adjusted EPA residual by QB x dominant-WR archetype
  pregame_archetype_mix    leak-free per-game mix: snaps from earlier games this season (or
                           last season for a team's first game) x each player's archetype
                           from his most recent *completed earlier* season
"""

from __future__ import annotations

import logging
import re

import numpy as np
import polars as pl
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.db import connect
from src.features.team import TEAM_FIX

log = logging.getLogger(__name__)
GROUPS = ["QB", "RB", "WR", "TE"]
ANALYSIS_SEASONS = (2018, 2025)  # RB archetypes start 2018; 2026 is incomplete


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def _fix(col: str) -> str:
    cases = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in TEAM_FIX.items())
    return f"CASE {col} {cases} ELSE {col} END"


BASE_SQL = f"""
CREATE OR REPLACE TABLE player_archetypes AS
SELECT player_id, season, position, archetype, membership_prob, projected
FROM (
    SELECT player_id, season, position, archetype, membership_prob, projected FROM archetype_qb
    UNION ALL SELECT player_id, season, position, archetype, membership_prob, projected
              FROM archetype_rb
    UNION ALL SELECT player_id, season, position, archetype, membership_prob, projected
              FROM archetype_wr
    UNION ALL SELECT player_id, season, position, archetype, membership_prob, projected
              FROM archetype_te
);

CREATE OR REPLACE TABLE player_snaps AS
WITH xwalk AS (
    SELECT pfr_id, any_value(gsis_id) AS gsis_id
    FROM (SELECT pfr_id, gsis_id FROM raw_players
          UNION SELECT pfr_id, gsis_id FROM raw_rosters)
    WHERE pfr_id IS NOT NULL AND gsis_id IS NOT NULL
    GROUP BY pfr_id
)
SELECT s.game_id, s.season, s.week, s.game_type, {_fix("s.team")} AS team,
       x.gsis_id AS player_id, s.player AS player_name,
       CASE WHEN s.position = 'FB' THEN 'RB' ELSE s.position END AS snap_pos,
       s.offense_snaps
FROM raw_snap_counts s JOIN xwalk x ON x.pfr_id = s.pfr_player_id
WHERE s.position IN ('QB', 'RB', 'FB', 'WR', 'TE') AND s.offense_snaps > 0;

CREATE OR REPLACE TABLE team_archetype_mix AS
WITH snaps AS (
    SELECT ps.season, ps.team, ps.player_id,
           coalesce(a.position, ps.snap_pos) AS pos,
           coalesce(a.archetype, 'Unlabeled') AS archetype,
           sum(ps.offense_snaps) AS snaps
    FROM player_snaps ps
    LEFT JOIN player_archetypes a ON a.player_id = ps.player_id AND a.season = ps.season
    WHERE ps.game_type = 'REG'
    GROUP BY ALL
)
SELECT season, team, pos, archetype, sum(snaps) AS snaps,
       sum(snaps) / sum(sum(snaps)) OVER (PARTITION BY season, team, pos) AS snap_share
FROM snaps GROUP BY season, team, pos, archetype;
"""

PREGAME_SQL = f"""
CREATE OR REPLACE TABLE pregame_archetype_mix_long AS
WITH tg AS (
    SELECT game_id, season, week, {_fix("home_team")} AS team FROM raw_schedules
    UNION ALL
    SELECT game_id, season, week, {_fix("away_team")} AS team FROM raw_schedules
),
cur AS (  -- snaps from this team's earlier games this season
    SELECT tg.game_id, tg.team, tg.season, s.player_id, any_value(s.snap_pos) AS snap_pos,
           sum(s.offense_snaps) AS snaps
    FROM tg JOIN player_snaps s
      ON s.team = tg.team AND s.season = tg.season AND s.week < tg.week
    GROUP BY ALL
),
prev AS (  -- first game of a season: last season's snaps for the franchise
    SELECT tg.game_id, tg.team, tg.season, s.player_id, any_value(s.snap_pos) AS snap_pos,
           sum(s.offense_snaps) AS snaps
    FROM tg JOIN player_snaps s ON s.team = tg.team AND s.season = tg.season - 1
    WHERE NOT EXISTS (SELECT 1 FROM cur c WHERE c.game_id = tg.game_id AND c.team = tg.team)
    GROUP BY ALL
),
snaps AS (SELECT * FROM cur UNION ALL SELECT * FROM prev),
labeled AS (  -- archetype from the player's most recent season strictly before this one
    SELECT s.*, a.position AS arch_pos, a.archetype
    FROM snaps s
    ASOF LEFT JOIN (SELECT * FROM player_archetypes WHERE NOT projected) a
      ON a.player_id = s.player_id AND s.season > a.season
)
SELECT game_id, team, pos, archetype,
       sum(snaps) / sum(sum(snaps)) OVER (PARTITION BY game_id, team, pos) AS share
FROM (SELECT game_id, team, snaps, coalesce(arch_pos, snap_pos) AS pos,
             coalesce(archetype, 'Unlabeled') AS archetype FROM labeled)
GROUP BY game_id, team, pos, archetype;
"""

TEAM_SEASON_SQL = """
WITH plays AS (
    SELECT season, posteam AS team, epa
    FROM raw_pbp
    WHERE season_type = 'REG' AND play_type IN ('pass', 'run') AND epa IS NOT NULL
      AND coalesce(two_point_attempt, 0) = 0
),
off AS (SELECT season, team, avg(epa) AS off_epa_play, count(*) AS plays
        FROM plays GROUP BY ALL),
qb AS (SELECT season, team, sum(dropbacks * epa_per_dropback) / sum(dropbacks) AS qb_eff
       FROM feat_passing GROUP BY ALL),
rec AS (SELECT season, team, sum(targets * epa_per_target) / sum(targets) AS rec_eff
        FROM feat_receiving GROUP BY ALL),
rush AS (SELECT season, team, sum(carries * rush_epa) / sum(carries) AS rush_eff
         FROM feat_rushing GROUP BY ALL),
prior AS (SELECT season + 1 AS season, team, off_epa_play AS prior_off_epa FROM off),
qb_primary AS (
    SELECT m.season, m.team, arg_max(m.archetype, m.snaps) AS qb_archetype
    FROM team_archetype_mix m WHERE m.pos = 'QB' AND m.archetype <> 'Unlabeled'
    GROUP BY ALL
),
wr_dominant AS (
    SELECT season, team, arg_max(archetype, snap_share) AS wr_archetype
    FROM team_archetype_mix WHERE pos = 'WR' AND archetype <> 'Unlabeled'
    GROUP BY ALL
)
SELECT o.season, o.team, o.off_epa_play, o.plays, qb.qb_eff, rec.rec_eff, rush.rush_eff,
       prior.prior_off_epa, qp.qb_archetype, wd.wr_archetype
FROM off o
LEFT JOIN qb USING (season, team)
LEFT JOIN rec USING (season, team)
LEFT JOIN rush USING (season, team)
LEFT JOIN prior USING (season, team)
LEFT JOIN qb_primary qp USING (season, team)
LEFT JOIN wr_dominant wd USING (season, team)
"""


def wide_mix(long: pl.DataFrame, keys: list[str], value: str) -> pl.DataFrame:
    """Long (keys, pos, archetype, value) -> wide mix_<pos>_<archetype> columns, 0-filled."""
    w = long.with_columns(
        col=pl.format("mix_{}_{}", pl.col("pos").str.to_lowercase(),
                      pl.col("archetype").map_elements(slug, return_dtype=pl.String))
    ).filter(pl.col("pos").is_in(GROUPS)).pivot(on="col", index=keys, values=value,
                                                aggregate_function="sum")
    cols = sorted(c for c in w.columns if c.startswith("mix_"))
    return w.select(*keys, *[pl.col(c).fill_null(0.0) for c in cols])


def ols_fit(X: np.ndarray, y: np.ndarray) -> tuple[float, int]:
    """Residual sum of squares and parameter count (with intercept)."""
    X1 = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    rank = np.linalg.matrix_rank(X1)
    return float(np.sum((y - X1 @ beta) ** 2)), int(rank)


def nested_f(Xr: np.ndarray, Xf: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    rss_r, k_r = ols_fit(Xr, y)
    rss_f, k_f = ols_fit(Xf, y)
    df1, df2 = k_f - k_r, len(y) - k_f
    F = ((rss_r - rss_f) / df1) / (rss_f / df2)
    return float(F), float(stats.f.sf(F, df1, df2)), df1


def cv_r2(X: np.ndarray, y: np.ndarray) -> float:
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0))
    cv = RepeatedKFold(n_splits=10, n_repeats=10, random_state=42)
    return float(cross_val_score(model, X, y, cv=cv, scoring="r2").mean())


def run_tests(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Talent-only -> + archetype main effects -> + QB x WR interactions, under two talent
    definitions: same-season individual efficiency (mechanically close to team EPA, so a
    strict test) and prior-season team EPA (no mechanical overlap)."""
    lo, hi = ANALYSIS_SEASONS
    df = df.filter(pl.col("season").is_between(lo, hi) & pl.col("qb_archetype").is_not_null())
    mix_cols = [c for c in df.columns if c.startswith("mix_") and not c.endswith("_unlabeled")
                and not c.startswith("mix_qb_")]
    wr_cols = [c for c in mix_cols if c.startswith("mix_wr_")]
    qb_levels = sorted(df["qb_archetype"].unique())
    qb_dummies = [f"qb_is_{slug(q)}" for q in qb_levels[1:]]  # first level is the baseline
    df = df.with_columns(
        (pl.col("qb_archetype") == q).cast(pl.Float64).alias(f"qb_is_{slug(q)}")
        for q in qb_levels[1:]
    )
    inter = []
    for q in qb_dummies:
        for w in wr_cols:
            df = df.with_columns((pl.col(q) * pl.col(w)).alias(f"{q}__x__{w}"))
            inter.append(f"{q}__x__{w}")

    rows = []
    for label, talent in [("same-season efficiency", ["qb_eff", "rec_eff", "rush_eff"]),
                          ("prior-season team EPA", ["prior_off_epa"])]:
        d = df.drop_nulls(talent)
        yy = d["off_epa_play"].to_numpy()
        X0 = d.select(talent).to_numpy()
        X1 = d.select(talent + qb_dummies + mix_cols).to_numpy()
        X2 = d.select(talent + qb_dummies + mix_cols + inter).to_numpy()
        for name, Xr, Xf in [("+ archetype mix", X0, X1), ("+ QB x WR pairings", X1, X2)]:
            F, p, df1 = nested_f(Xr, Xf, yy)
            rows.append(dict(talent_controls=label, step=name, n=len(yy), added_params=df1,
                             F=F, p_value=p, cv_r2_before=cv_r2(Xr, yy),
                             cv_r2_after=cv_r2(Xf, yy)))
    tests = pl.DataFrame(rows).with_columns(
        cv_r2_gain=pl.col("cv_r2_after") - pl.col("cv_r2_before")
    )

    # Pairing table: residual from the strict talent model, by QB x dominant WR archetype.
    d = df.drop_nulls(["qb_eff", "rec_eff", "rush_eff", "wr_archetype"])
    X0 = np.column_stack([np.ones(d.height), d.select("qb_eff", "rec_eff", "rush_eff").to_numpy()])
    beta, *_ = np.linalg.lstsq(X0, d["off_epa_play"].to_numpy(), rcond=None)
    resid = d["off_epa_play"].to_numpy() - X0 @ beta
    pairs = (d.with_columns(resid=pl.Series(resid))
             .group_by("qb_archetype", "wr_archetype")
             .agg(pl.len().alias("n"), pl.col("resid").mean().alias("mean_resid"),
                  (pl.col("resid").std() / pl.len().sqrt()).alias("se"),
                  pl.col("off_epa_play").mean().alias("mean_off_epa"))
             .sort("mean_resid", descending=True))
    return tests, pairs


def _write(con, name: str, df: pl.DataFrame) -> None:
    con.register("_t", df.to_arrow())
    con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM _t")
    con.unregister("_t")


def build() -> tuple[pl.DataFrame, pl.DataFrame]:
    with connect() as con:
        con.execute(BASE_SQL)
        con.execute(PREGAME_SQL)
        mix = con.execute("SELECT * FROM team_archetype_mix").pl()
        ts = con.execute(TEAM_SEASON_SQL).pl().join(
            wide_mix(mix, ["season", "team"], "snap_share"), on=["season", "team"], how="left")
        _write(con, "team_season_analysis", ts)
        tests, pairs = run_tests(ts)
        _write(con, "archetype_mix_tests", tests)
        _write(con, "qb_wr_pairings", pairs)
        pre = wide_mix(con.execute("SELECT * FROM pregame_archetype_mix_long").pl(),
                       ["game_id", "team"], "share")
        _write(con, "pregame_archetype_mix", pre)
    log.info("archetype mix tests:\n%s", tests)
    return tests, pairs
