"""QB player-season features, built in DuckDB from raw_* tables.

A dropback (qb_dropback = 1) is credited to the passer on passes and sacks and to the
rusher on scrambles. Regular season only, two-point tries excluded.
  adot             mean air yards per pass attempt
  time_to_throw    NGS average time to throw (season row, else attempt-weighted weekly mean)
  scramble_rate    scrambles / dropbacks
  cpoe             mean completion % over expected (nflfastR cp model), in percentage points
  sack_rate        sacks / dropbacks
  epa_per_dropback mean qb_epa per dropback
  deep_pass_rate   share of attempts with air yards >= 20
"""

from __future__ import annotations

import duckdb

PASSING_FEATURES = [
    "adot",
    "time_to_throw",
    "scramble_rate",
    "cpoe",
    "sack_rate",
    "epa_per_dropback",
    "deep_pass_rate",
]

PASSING_SQL = """
CREATE OR REPLACE TABLE {out} AS
WITH db AS (
    SELECT season, game_id, posteam AS team,
           coalesce(passer_player_id, rusher_player_id) AS player_id,
           pass_attempt, sack, qb_scramble, air_yards, cpoe, qb_epa
    FROM {pbp}
    WHERE season_type = 'REG' AND qb_dropback = 1
      AND coalesce(two_point_attempt, 0) = 0
      AND coalesce(passer_player_id, rusher_player_id) IS NOT NULL
),
agg AS (
    SELECT season, player_id,
           count(*)                                                       AS dropbacks,
           count(*) FILTER (WHERE pass_attempt = 1 AND coalesce(sack, 0) = 0) AS attempts,
           count(DISTINCT game_id)                                        AS games,
           mode(team)                                                     AS team,
           avg(air_yards) FILTER (WHERE coalesce(sack, 0) = 0)            AS adot,
           avg(coalesce(qb_scramble, 0))                                  AS scramble_rate,
           avg(cpoe)                                                      AS cpoe,
           avg(coalesce(sack, 0))                                         AS sack_rate,
           avg(qb_epa)                                                    AS epa_per_dropback,
           avg(CASE WHEN air_yards >= 20 THEN 1.0 ELSE 0.0 END)
               FILTER (WHERE air_yards IS NOT NULL AND coalesce(sack, 0) = 0) AS deep_pass_rate
    FROM db GROUP BY ALL
),
ngs_season AS (
    SELECT season, player_gsis_id AS player_id, avg_time_to_throw AS ttt
    FROM {ngs} WHERE season_type = 'REG' AND week = 0
),
ngs_weekly AS (
    SELECT season, player_gsis_id AS player_id,
           sum(avg_time_to_throw * attempts) / nullif(sum(attempts), 0) AS ttt
    FROM {ngs} WHERE season_type = 'REG' AND week > 0
    GROUP BY ALL
),
pos AS (
    SELECT season, gsis_id AS player_id, mode(position) AS position
    FROM {rosters} GROUP BY ALL
),
names AS (
    SELECT player_id, arg_max(player_display_name, season) AS player_name
    FROM {player_stats} GROUP BY ALL
)
SELECT a.player_id, n.player_name, a.season, a.team, p.position, a.games,
       a.dropbacks, a.attempts,
       a.adot,
       coalesce(ns.ttt, nw.ttt) AS time_to_throw,
       a.scramble_rate, a.cpoe, a.sack_rate, a.epa_per_dropback, a.deep_pass_rate
FROM agg a
LEFT JOIN ngs_season ns USING (season, player_id)
LEFT JOIN ngs_weekly nw USING (season, player_id)
LEFT JOIN pos p   USING (season, player_id)
LEFT JOIN names n USING (player_id)
WHERE a.dropbacks >= {min_dropbacks}
ORDER BY season, dropbacks DESC
"""


def build_passing(
    con: duckdb.DuckDBPyConnection,
    out: str = "feat_passing",
    pbp: str = "raw_pbp",
    player_stats: str = "raw_player_stats",
    ngs: str = "raw_ngs_passing",
    rosters: str = "raw_rosters",
    min_dropbacks: int = 50,
) -> None:
    con.execute(
        PASSING_SQL.format(
            out=out, pbp=pbp, player_stats=player_stats, ngs=ngs, rosters=rosters,
            min_dropbacks=min_dropbacks,
        )
    )
