"""Build all player-season feature tables."""

from __future__ import annotations

import logging

from src.db import connect
from src.features.defense import build_defense
from src.features.passing import build_passing
from src.features.receiving import build_receiving
from src.features.rushing import build_rushing
from src.features.team import build_team_game_stats

log = logging.getLogger(__name__)

BUILDERS = {
    "receiving": build_receiving,  # WR + TE
    "rushing": build_rushing,  # RB
    "passing": build_passing,  # QB
    "defense": build_defense,  # whole-team defenses
}


def build(groups: list[str] | None = None) -> None:
    with connect() as con:
        for name, fn in BUILDERS.items():
            if groups and name not in groups:
                continue
            if name == "defense":
                build_team_game_stats(con)  # defense profile is aggregated from these
            fn(con)
            n = con.execute(f"SELECT count(*) FROM feat_{name}").fetchone()[0]
            log.info("feat_%s: %d player-seasons", name, n)
