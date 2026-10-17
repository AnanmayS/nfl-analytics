"""RB player-season features, built in DuckDB from raw_* tables.

Feature definitions (regular season only, two-point tries excluded):
  rush_epa             mean EPA per designed rush (QB scrambles are not rush attempts here)
  ryoe_per_att         NGS rush yards over expected per attempt (season row, else
                       attempt-weighted weekly mean)
  target_share         player targets / team targets, over games the player appeared in
  route_participation  share of team dropbacks with the player on the field (participation
                       data; a proxy since per-player routes are not public). Null for
                       seasons without participation data.
  explosive_run_rate   share of rushes gaining 10+ yards
  goal_line_share      player carries inside the 5 / team carries inside the 5 in his games
"""

from __future__ import annotations

import duckdb

RUSHING_FEATURES = [
    "rush_epa",
    "ryoe_per_att",
    "target_share",
    "route_participation",
    "explosive_run_rate",
    "goal_line_share",
]

RUSHING_SQL = """
CREATE OR REPLACE TABLE {out} AS
WITH plays AS (
    SELECT season, game_id, play_id, posteam AS team, rusher_player_id, receiver_player_id,
           rush_attempt, pass_attempt, qb_dropback, sack, epa, yards_gained, yardline_100
    FROM {pbp}
    WHERE season_type = 'REG' AND coalesce(two_point_attempt, 0) = 0
),
rush AS (
    SELECT * FROM plays
    WHERE rush_attempt = 1 AND rusher_player_id IS NOT NULL AND coalesce(qb_dropback, 0) = 0
),
tgt AS (
    SELECT * FROM plays
    WHERE pass_attempt = 1 AND coalesce(sack, 0) = 0 AND receiver_player_id IS NOT NULL
),
team_game AS (
    SELECT p.season, p.game_id, p.team,
           count(*) FILTER (WHERE pass_attempt = 1 AND coalesce(sack, 0) = 0
                            AND receiver_player_id IS NOT NULL)                  AS tm_targets,
           count(*) FILTER (WHERE rush_attempt = 1 AND coalesce(qb_dropback, 0) = 0
                            AND yardline_100 <= 5)                               AS tm_gl_carries,
           count(*) FILTER (WHERE qb_dropback = 1)                               AS tm_dropbacks
    FROM plays p
    GROUP BY ALL
),
player_game AS (
    SELECT DISTINCT season, game_id, team, player_id
    FROM {player_stats} WHERE season_type = 'REG'
    UNION SELECT DISTINCT season, game_id, team, rusher_player_id FROM rush
    UNION SELECT DISTINCT season, game_id, team, receiver_player_id FROM tgt
),
denoms AS (
    SELECT pg.season, pg.player_id,
           count(DISTINCT pg.game_id) AS games,
           sum(tg.tm_targets)         AS tm_targets,
           sum(tg.tm_gl_carries)      AS tm_gl_carries
    FROM player_game pg JOIN team_game tg USING (season, game_id, team)
    GROUP BY ALL
),
on_field AS (  -- dropbacks with the player on the field
    SELECT p.season, trim(u.player_id) AS player_id, count(*) AS dropbacks_on_field
    FROM plays p
    JOIN {participation} pa ON pa.nflverse_game_id = p.game_id AND pa.play_id = p.play_id
    CROSS JOIN unnest(string_split(pa.offense_players, ';')) AS u(player_id)
    WHERE p.qb_dropback = 1
    GROUP BY ALL
),
part_seasons AS (  -- seasons where participation exists at all
    SELECT DISTINCT p.season FROM plays p
    JOIN {participation} pa ON pa.nflverse_game_id = p.game_id AND pa.play_id = p.play_id
),
dropback_denoms AS (  -- team dropbacks in the player's games, only where participation exists
    SELECT pg.season, pg.player_id, sum(tg.tm_dropbacks) AS tm_dropbacks
    FROM player_game pg JOIN team_game tg USING (season, game_id, team)
    WHERE pg.season IN (SELECT season FROM part_seasons)
    GROUP BY ALL
),
rush_agg AS (
    SELECT season, rusher_player_id AS player_id,
           count(*)                                     AS carries,
           avg(epa)                                     AS rush_epa,
           avg(CASE WHEN yards_gained >= 10 THEN 1.0 ELSE 0.0 END) AS explosive_run_rate,
           count(*) FILTER (WHERE yardline_100 <= 5)    AS gl_carries
    FROM rush GROUP BY ALL
),
tgt_agg AS (
    SELECT season, receiver_player_id AS player_id, count(*) AS targets
    FROM tgt GROUP BY ALL
),
rec_agg AS (
    SELECT season, player_id, sum(receptions)::BIGINT AS receptions
    FROM {player_stats} WHERE season_type = 'REG' GROUP BY ALL
),
primary_team AS (
    SELECT season, rusher_player_id AS player_id, mode(team) AS team FROM rush GROUP BY ALL
),
ngs_season AS (
    SELECT season, player_gsis_id AS player_id, rush_yards_over_expected_per_att AS ryoe
    FROM {ngs} WHERE season_type = 'REG' AND week = 0
),
ngs_weekly AS (
    SELECT season, player_gsis_id AS player_id,
           sum(rush_yards_over_expected_per_att * rush_attempts)
               / nullif(sum(rush_attempts), 0) AS ryoe
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
SELECT r.player_id, n.player_name, r.season, pt.team, p.position, d.games,
       r.carries, coalesce(t.targets, 0) AS targets, coalesce(rc.receptions, 0) AS receptions,
       r.carries + coalesce(rc.receptions, 0) AS touches,
       r.rush_epa,
       coalesce(ns.ryoe, nw.ryoe)                       AS ryoe_per_att,
       coalesce(t.targets, 0) / d.tm_targets            AS target_share,
       least(1.0, coalesce(o.dropbacks_on_field, 0) / nullif(dd.tm_dropbacks, 0))
                                                        AS route_participation,
       r.explosive_run_rate,
       r.gl_carries / nullif(d.tm_gl_carries, 0)        AS goal_line_share
FROM rush_agg r
JOIN denoms d        USING (season, player_id)
JOIN primary_team pt USING (season, player_id)
LEFT JOIN tgt_agg t  USING (season, player_id)
LEFT JOIN rec_agg rc USING (season, player_id)
LEFT JOIN on_field o USING (season, player_id)
LEFT JOIN dropback_denoms dd USING (season, player_id)
LEFT JOIN ngs_season ns USING (season, player_id)
LEFT JOIN ngs_weekly nw USING (season, player_id)
LEFT JOIN pos p      USING (season, player_id)
LEFT JOIN names n    USING (player_id)
WHERE r.carries + coalesce(rc.receptions, 0) >= {min_touches}
ORDER BY season, touches DESC
"""


def build_rushing(
    con: duckdb.DuckDBPyConnection,
    out: str = "feat_rushing",
    pbp: str = "raw_pbp",
    player_stats: str = "raw_player_stats",
    ngs: str = "raw_ngs_rushing",
    rosters: str = "raw_rosters",
    participation: str = "raw_participation",
    min_touches: int = 30,
) -> None:
    con.execute(
        RUSHING_SQL.format(
            out=out, pbp=pbp, player_stats=player_stats, ngs=ngs, rosters=rosters,
            participation=participation, min_touches=min_touches,
        )
    )
