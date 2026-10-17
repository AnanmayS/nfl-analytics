"""Team-season profile inputs for the dashboard's plain-language team identity.

team_profile      season, team, neutral-situation pass rate and its league percentile
team_key_players  each team-season's most-used QB, RB, WRs and TE with their archetype
                  (same season if labeled, else the player's most recent earlier one)
"""

from __future__ import annotations

import duckdb

from src.features.team import TEAM_FIX

KEY_SLOTS = {"QB": 1, "RB": 1, "WR": 2, "TE": 1}

PROFILE_SQL = """
CREATE OR REPLACE TABLE team_profile AS
WITH p AS (  -- early downs, game still competitive, outside the two-minute drill
    SELECT season, posteam AS team, pass
    FROM raw_pbp
    WHERE play_type IN ('pass', 'run') AND down IN (1, 2) AND wp BETWEEN 0.2 AND 0.8
      AND half_seconds_remaining > 120 AND posteam IS NOT NULL
)
SELECT season, team, avg(pass) AS pass_rate, count(*) AS plays,
       percent_rank() OVER (PARTITION BY season ORDER BY avg(pass)) AS pass_rate_pct
FROM p GROUP BY season, team
"""

KEY_PLAYERS_SQL = """
CREATE OR REPLACE TABLE team_key_players AS
WITH snaps AS (
    SELECT season, team, player_id, any_value(player_name) AS player_name,
           any_value(snap_pos) AS pos, sum(offense_snaps) AS snaps
    FROM player_snaps
    GROUP BY season, team, player_id
),
ranked AS (
    SELECT *, row_number() OVER (PARTITION BY season, team, pos ORDER BY snaps DESC) AS rk
    FROM snaps WHERE pos IN ('QB', 'RB', 'WR', 'TE')
)
SELECT r.season, r.team, r.pos, r.rk, r.player_id, r.player_name, r.snaps,
       a.archetype, a.season AS archetype_season
FROM ranked r
ASOF LEFT JOIN player_archetypes a
  ON a.player_id = r.player_id AND r.season >= a.season
WHERE {slots}
"""


def build(con: duckdb.DuckDBPyConnection) -> None:
    """Requires raw_pbp, player_snaps and player_archetypes (built by archetype_mix)."""
    con.execute(PROFILE_SQL)
    cases = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in TEAM_FIX.items())
    con.execute(f"UPDATE team_profile SET team = CASE team {cases} ELSE team END")
    slots = " OR ".join(f"(r.pos = '{p}' AND r.rk <= {n})" for p, n in KEY_SLOTS.items())
    con.execute(KEY_PLAYERS_SQL.format(slots=slots))
