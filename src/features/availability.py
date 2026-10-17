"""Who's missing: pre-kickoff availability features from the official injury report.

For every scheduled team-game (played or upcoming):
  1. Each player's recent role = average offensive / defensive snap share over the team's
     last 3 games played strictly before this game (crossing into last season for week 1).
     A player already out for those games has a role near 0, so a long-term absence that
     the team's recent form already reflects isn't counted again.
  2. That week's injury report status is turned into a chance the player sits:
     Out 1.0, Doubtful 0.9, Questionable 0.2.
  3. Features:
       qb_out       chance the starting QB (most recent snaps, share >= 50%) sits
       off_missing  sum of (chance sits x offensive snap share) for non-QBs
                    -- "full-time offensive starters missing"
       def_missing  same for defense
       report_available  whether that team has an injury report for the week yet
     plus key_absences, a short readable list for the dashboard.

The injury report is published before kickoff (final report two days before most games),
so these are legitimate pre-game inputs.
"""

from __future__ import annotations

import duckdb

from src.features.team import TEAM_FIX

STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.9, "Questionable": 0.2}
RECENT_GAMES = 3
STARTER_SHARE = 0.5


def _fix(col: str) -> str:
    cases = " ".join(f"WHEN '{k}' THEN '{v}'" for k, v in TEAM_FIX.items())
    return f"CASE {col} {cases} ELSE {col} END"


AVAILABILITY_SQL = f"""
CREATE OR REPLACE TABLE player_game_snaps AS
WITH xwalk AS (
    SELECT pfr_id, any_value(gsis_id) AS gsis_id
    FROM (SELECT pfr_id, gsis_id FROM {{players}} UNION SELECT pfr_id, gsis_id FROM {{rosters}})
    WHERE pfr_id IS NOT NULL AND gsis_id IS NOT NULL
    GROUP BY pfr_id
)
SELECT s.game_id, s.season, s.week, {_fix("s.team")} AS team, x.gsis_id AS player_id,
       s.player AS player_name, s.position,
       coalesce(s.offense_pct, 0) AS offense_pct, coalesce(s.defense_pct, 0) AS defense_pct
FROM {{snaps}} s JOIN xwalk x ON x.pfr_id = s.pfr_player_id;

CREATE OR REPLACE TABLE game_availability AS
WITH tg AS (
    SELECT game_id, season, week, CAST(gameday AS DATE) AS game_date,
           {_fix("home_team")} AS team FROM {{schedules}}
    UNION ALL
    SELECT game_id, season, week, CAST(gameday AS DATE) AS game_date,
           {_fix("away_team")} AS team FROM {{schedules}}
),
played AS (
    SELECT DISTINCT tg.game_id, tg.team, tg.game_date
    FROM tg JOIN player_game_snaps s ON s.game_id = tg.game_id AND s.team = tg.team
),
recent AS (  -- the team's last few games strictly before this one (max ~10 months back)
    SELECT t.game_id, t.team, p.game_id AS prior_game,
           row_number() OVER (PARTITION BY t.game_id, t.team ORDER BY p.game_date DESC) AS rn
    FROM tg t JOIN played p
      ON p.team = t.team AND p.game_date < t.game_date
     AND p.game_date >= t.game_date - INTERVAL 300 DAY
),
n_recent AS (
    SELECT game_id, team, count(*) AS n FROM recent WHERE rn <= {RECENT_GAMES}
    GROUP BY ALL
),
roles AS (
    SELECT r.game_id, r.team, s.player_id,
           any_value(s.player_name) AS player_name, any_value(s.position) AS position,
           sum(s.offense_pct) / any_value(n.n) AS off_share,
           sum(s.defense_pct) / any_value(n.n) AS def_share
    FROM recent r
    JOIN n_recent n ON n.game_id = r.game_id AND n.team = r.team
    JOIN player_game_snaps s ON s.game_id = r.prior_game AND s.team = r.team
    WHERE r.rn <= {RECENT_GAMES}
    GROUP BY r.game_id, r.team, s.player_id
),
inj AS (
    SELECT season, week, {_fix("team")} AS team, gsis_id AS player_id,
           max(CASE report_status {" ".join(f"WHEN '{k}' THEN {v}::DOUBLE"
                                             for k, v in STATUS_WEIGHT.items())}
                   ELSE 0::DOUBLE END) AS sit_chance,
           arg_max(report_status, CASE report_status WHEN 'Out' THEN 3 WHEN 'Doubtful' THEN 2
                                   WHEN 'Questionable' THEN 1 ELSE 0 END) AS status
    FROM {{injuries}}
    GROUP BY ALL
),
reported AS (SELECT DISTINCT season, week, team FROM inj),
starter AS (
    SELECT game_id, team, arg_max(player_id, off_share) AS qb_id,
           arg_max(player_name, off_share) AS qb_name, max(off_share) AS qb_share
    FROM roles WHERE position = 'QB' GROUP BY ALL
),
backup AS (  -- next QB by recent snaps: the likely replacement if the starter can't go
    SELECT r.game_id, r.team, arg_max(r.player_id, r.off_share) AS qb_id,
           arg_max(r.player_name, r.off_share) AS qb_name
    FROM roles r JOIN starter st ON st.game_id = r.game_id AND st.team = r.team
    WHERE r.position = 'QB' AND r.player_id <> st.qb_id GROUP BY ALL
),
hit AS (  -- reported players with a recent role
    SELECT t.game_id, t.team, r.player_id, r.player_name, r.position, i.status,
           i.sit_chance, r.off_share, r.def_share,
           i.sit_chance * greatest(r.off_share, r.def_share) AS impact
    FROM tg t
    JOIN inj i ON i.season = t.season AND i.week = t.week AND i.team = t.team
    JOIN roles r ON r.game_id = t.game_id AND r.team = t.team AND r.player_id = i.player_id
    WHERE i.sit_chance > 0
)
SELECT t.game_id, t.team,
       (rp.team IS NOT NULL) AS report_available,
       coalesce(max(CASE WHEN h.player_id = st.qb_id AND st.qb_share >= {STARTER_SHARE}
                         THEN h.sit_chance END), 0) AS qb_out,
       any_value(st.qb_name) AS starting_qb, any_value(st.qb_id) AS starting_qb_id,
       any_value(bk.qb_id) AS backup_qb_id, any_value(bk.qb_name) AS backup_qb,
       coalesce(sum(CASE WHEN h.position <> 'QB' THEN h.sit_chance * h.off_share END), 0)
           AS off_missing,
       coalesce(sum(h.sit_chance * h.def_share), 0) AS def_missing,
       string_agg(CASE WHEN h.impact >= 0.3
                       THEN h.position || ' ' || h.player_name || ' (' || h.status || ')' END,
                  '; ' ORDER BY h.impact DESC) AS key_absences
FROM tg t
LEFT JOIN reported rp ON rp.season = t.season AND rp.week = t.week AND rp.team = t.team
LEFT JOIN starter st ON st.game_id = t.game_id AND st.team = t.team
LEFT JOIN backup bk ON bk.game_id = t.game_id AND bk.team = t.team
LEFT JOIN hit h ON h.game_id = t.game_id AND h.team = t.team
GROUP BY t.game_id, t.team, rp.team
"""


def build_availability(con: duckdb.DuckDBPyConnection, schedules: str = "raw_schedules",
                       snaps: str = "raw_snap_counts", injuries: str = "raw_injuries",
                       players: str = "raw_players", rosters: str = "raw_rosters") -> None:
    con.execute(AVAILABILITY_SQL.format(schedules=schedules, snaps=snaps, injuries=injuries,
                                        players=players, rosters=rosters))
