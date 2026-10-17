"""Team-season defensive profile for defense archetypes, built from team_game_stats.

Values are per-game averages over the regular season (one row per team-season):
  pass_epa_allowed / rush_epa_allowed   EPA per opponent pass / run play
  pressure_rate                         sacks or QB hits per opponent dropback
  blitz_rate                            share of dropbacks with 5+ pass rushers
  heavy_box_rel                         share of early-down snaps with 8+ in the box, minus
                                        that season's league average (the source changed
                                        in 2024, so only the relative value is comparable)
  adot_allowed                          opponents' average throw depth
  explosive_rate_allowed                runs of 10+ / passes of 20+ yards allowed per play
  takeaway_rate                         interceptions + lost fumbles forced per play
  third_down_rate_allowed               opponents' third-down conversion rate
  cpoe_allowed                          opponents' completion % over expected
Requires team_game_stats (rebuilt by `make features`).
"""

from __future__ import annotations

import duckdb

DEFENSE_FEATURES = [
    "pass_epa_allowed",
    "rush_epa_allowed",
    "pressure_rate",
    "blitz_rate",
    "heavy_box_rel",
    "adot_allowed",
    "explosive_rate_allowed",
    "takeaway_rate",
    "third_down_rate_allowed",
    "cpoe_allowed",
]

DEFENSE_SQL = """
CREATE OR REPLACE TABLE {out} AS
WITH g AS (
    SELECT t.* FROM {team_games} t
    JOIN {schedules} s ON s.game_id = t.game_id AND s.game_type = 'REG'
),
agg AS (
    SELECT season, team, count(*) AS games,
           avg(def_pass_epa) AS pass_epa_allowed,
           avg(def_rush_epa) AS rush_epa_allowed,
           avg(def_pressure_rate) AS pressure_rate,
           avg(def_blitz_rate) AS blitz_rate,
           avg(def_heavy_box_rate) AS heavy_box_rate,
           avg(def_adot) AS adot_allowed,
           avg(def_explosive_rate) AS explosive_rate_allowed,
           avg(def_turnover_rate) AS takeaway_rate,
           avg(def_third_down_rate) AS third_down_rate_allowed,
           avg(def_cpoe) AS cpoe_allowed
    FROM g GROUP BY season, team
)
SELECT a.team AS player_id, coalesce(t.team_name, a.team) AS player_name, a.season, a.team,
       'DEF' AS position, a.games,
       a.pass_epa_allowed, a.rush_epa_allowed, a.pressure_rate, a.blitz_rate,
       a.heavy_box_rate - avg(a.heavy_box_rate) OVER (PARTITION BY a.season) AS heavy_box_rel,
       a.adot_allowed, a.explosive_rate_allowed, a.takeaway_rate,
       a.third_down_rate_allowed, a.cpoe_allowed
FROM agg a LEFT JOIN {teams} t ON t.team_abbr = a.team
ORDER BY season, team
"""


def build_defense(con: duckdb.DuckDBPyConnection, out: str = "feat_defense",
                  team_games: str = "team_game_stats", schedules: str = "raw_schedules",
                  teams: str = "raw_teams") -> None:
    con.execute(DEFENSE_SQL.format(out=out, team_games=team_games, schedules=schedules,
                                   teams=teams))
