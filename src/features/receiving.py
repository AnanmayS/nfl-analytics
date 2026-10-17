"""WR/TE player-season receiving features, built in DuckDB from raw_* tables.

Feature definitions (regular season only, two-point tries and sacks excluded):
  adot                 mean air_yards per target
  target_share         player targets / team targets, over games the player appeared in
  air_yards_share      player air yards / team air yards, same denominator games
  yacoe                mean(yards_after_catch - xyac_mean_yardage) over receptions
  avg_separation       NGS season average separation at catch/incompletion (yards)
  avg_cushion          NGS season average cushion at snap (yards)
  middle_target_rate   share of targets thrown to pass_location='middle' (slot proxy;
                       true alignment data is not public)
  rz_target_share      player targets inside the 20 / team red-zone targets in his games
  croe                 mean(complete_pass - cp) over targets, i.e. catch rate over expected
  epa_per_target       mean EPA over targets
"""

from __future__ import annotations

import duckdb

# All features are clustered on. CROE and EPA/target carry talent/QB-quality signal, so
# Part 3 controls for individual efficiency before crediting archetype mix.
RECEIVING_FEATURES = [
    "adot",
    "target_share",
    "air_yards_share",
    "yacoe",
    "avg_separation",
    "middle_target_rate",
    "rz_target_share",
    "croe",
    "epa_per_target",
]

RECEIVING_SQL = """
CREATE OR REPLACE TABLE {out} AS
WITH tgt AS (
    SELECT season, game_id, posteam AS team, receiver_player_id AS player_id,
           air_yards, epa, complete_pass, cp, yards_after_catch, xyac_mean_yardage,
           pass_location, yardline_100
    FROM {pbp}
    WHERE season_type = 'REG'
      AND pass_attempt = 1
      AND coalesce(sack, 0) = 0
      AND coalesce(two_point_attempt, 0) = 0
      AND receiver_player_id IS NOT NULL
),
team_game AS (
    SELECT season, game_id, team,
           count(*)                                   AS tm_targets,
           sum(coalesce(air_yards, 0))                AS tm_air_yards,
           count(*) FILTER (WHERE yardline_100 <= 20) AS tm_rz_targets
    FROM tgt
    GROUP BY ALL
),
player_game AS (  -- every game a player appeared in (any stat row or any target)
    SELECT DISTINCT season, game_id, team, player_id
    FROM {player_stats}
    WHERE season_type = 'REG'
    UNION
    SELECT DISTINCT season, game_id, team, player_id FROM tgt
),
denoms AS (
    SELECT pg.season, pg.player_id,
           count(DISTINCT pg.game_id) AS games,
           sum(tg.tm_targets)         AS tm_targets,
           sum(tg.tm_air_yards)       AS tm_air_yards,
           sum(tg.tm_rz_targets)      AS tm_rz_targets
    FROM player_game pg
    JOIN team_game tg USING (season, game_id, team)
    GROUP BY ALL
),
player_team AS (
    SELECT season, player_id, team, count(*) AS n
    FROM tgt GROUP BY ALL
),
primary_team AS (
    SELECT season, player_id, arg_max(team, n) AS team
    FROM player_team GROUP BY ALL
),
agg AS (
    SELECT season, player_id,
           count(*)                                              AS targets,
           sum(complete_pass)                                    AS receptions,
           avg(air_yards)                                        AS adot,
           sum(coalesce(air_yards, 0))                           AS air_yards,
           avg(yards_after_catch - xyac_mean_yardage)
               FILTER (WHERE complete_pass = 1)                  AS yacoe,
           avg(CASE WHEN pass_location = 'middle' THEN 1.0 ELSE 0.0 END)
               FILTER (WHERE pass_location IS NOT NULL)          AS middle_target_rate,
           count(*) FILTER (WHERE yardline_100 <= 20)            AS rz_targets,
           avg(complete_pass - cp) FILTER (WHERE cp IS NOT NULL) AS croe,
           avg(epa)                                              AS epa_per_target
    FROM tgt
    GROUP BY ALL
),
ngs_season AS (
    SELECT season, player_gsis_id AS player_id, avg_separation, avg_cushion
    FROM {ngs}
    WHERE season_type = 'REG' AND week = 0
),
ngs_weekly AS (  -- fallback for players missing a season row: target-weighted weekly mean
    SELECT season, player_gsis_id AS player_id,
           sum(avg_separation * targets) / nullif(sum(targets), 0) AS avg_separation,
           sum(avg_cushion * targets)    / nullif(sum(targets), 0) AS avg_cushion
    FROM {ngs}
    WHERE season_type = 'REG' AND week > 0
    GROUP BY ALL
),
pos AS (
    SELECT season, gsis_id AS player_id, mode(position) AS position
    FROM {rosters}
    GROUP BY ALL
),
names AS (
    SELECT player_id, arg_max(player_display_name, season) AS player_name
    FROM {player_stats}
    GROUP BY ALL
)
SELECT a.player_id, n.player_name, a.season, pt.team, p.position, d.games,
       a.targets, a.receptions, a.air_yards, a.rz_targets,
       a.adot,
       a.targets    / d.tm_targets                AS target_share,
       a.air_yards  / nullif(d.tm_air_yards, 0)   AS air_yards_share,
       a.yacoe,
       coalesce(ns.avg_separation, nw.avg_separation) AS avg_separation,
       coalesce(ns.avg_cushion, nw.avg_cushion)       AS avg_cushion,
       a.middle_target_rate,
       a.rz_targets / nullif(d.tm_rz_targets, 0)  AS rz_target_share,
       a.croe,
       a.epa_per_target
FROM agg a
JOIN denoms d       USING (season, player_id)
JOIN primary_team pt USING (season, player_id)
LEFT JOIN pos p     USING (season, player_id)
LEFT JOIN names n   USING (player_id)
LEFT JOIN ngs_season ns USING (season, player_id)
LEFT JOIN ngs_weekly nw USING (season, player_id)
WHERE a.targets >= {min_targets}
ORDER BY season, targets DESC
"""


def build_receiving(
    con: duckdb.DuckDBPyConnection,
    out: str = "feat_receiving",
    pbp: str = "raw_pbp",
    player_stats: str = "raw_player_stats",
    ngs: str = "raw_ngs_receiving",
    rosters: str = "raw_rosters",
    min_targets: int = 20,
) -> None:
    con.execute(
        RECEIVING_SQL.format(
            out=out, pbp=pbp, player_stats=player_stats, ngs=ngs, rosters=rosters,
            min_targets=min_targets,
        )
    )
