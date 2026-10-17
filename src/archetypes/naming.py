"""Turn cluster centroid z-scores into readable archetype names.

Each feature has a descriptor for a high and/or low centroid. A cluster's name is built
from its most extreme features (by |z|), so names stay tied to what the data says.
"""

from __future__ import annotations

import numpy as np

# feature -> (descriptor when centroid z is high, descriptor when low, word position).
# Terms are chosen by |z| but printed in word-position order, e.g. "Deep Contested",
# "Underneath Alpha". None means that direction isn't a defining trait.
Descriptors = dict[str, tuple[str | None, str | None, int]]

RECEIVING: Descriptors = {
    "adot": ("Deep", "Underneath", 0),
    "middle_target_rate": ("Middle-Field", "Perimeter", 1),
    "rz_target_share": ("Red-Zone", None, 1),
    "yacoe": ("YAC", None, 2),
    "avg_separation": ("Separator", "Contested", 2),
    "target_share": ("Alpha", "Complement", 3),
    "air_yards_share": (None, None, 3),  # redundant with target_share x aDOT
    "croe": ("Sure-Handed", "Low-Catch", 2),
    "epa_per_target": ("Efficient", "Inefficient", 2),
}

RUSHING: Descriptors = {
    "route_participation": ("Every-Down", None, 0),
    "explosive_run_rate": ("Explosive", None, 1),
    "goal_line_share": ("Goal-Line Back", None, 3),
    "ryoe_per_att": ("Elusive", None, 1),
    "rush_epa": ("Efficient", "Inefficient", 2),
    "target_share": ("Pass-Catcher", "Two-Down Back", 3),
}

DEFENSE: Descriptors = {
    "blitz_rate": ("Blitz-Heavy", "Four-Man-Rush", 0),
    "heavy_box_rel": ("Box-Stacking", "Light-Box", 0),
    "adot_allowed": ("Tight-Underneath", "Keep-It-In-Front", 0),
    "pressure_rate": ("Pass-Rushing", "Low-Pressure", 1),
    "takeaway_rate": ("Ball-Hawking", None, 1),
    "rush_epa_allowed": ("Run-Vulnerable", "Run-Stuffing", 1),
    "pass_epa_allowed": ("Pass-Vulnerable", "Lockdown", 1),
    "explosive_rate_allowed": ("Big-Play-Prone", "No-Big-Plays", 1),
    "third_down_rate_allowed": ("Third-Down-Leaky", "Third-Down-Stout", 1),
    "cpoe_allowed": ("Soft-Coverage", "Sticky-Coverage", 1),
}

PASSING: Descriptors = {
    "adot": ("Aggressive", "Conservative", 0),
    "deep_pass_rate": ("Vertical", "Short-Area", 1),
    "time_to_throw": ("Hold-the-Ball", "Quick-Release", 1),
    "cpoe": ("Accurate", "Erratic", 2),
    "sack_rate": ("Sack-Prone", "Sack-Avoiding", 2),
    "epa_per_dropback": ("Efficient", "Inefficient", 2),
    "scramble_rate": ("Scrambler", "Pocket Passer", 3),
}

MIN_Z = 0.35


def describe(
    centroid: np.ndarray, features: list[str], descriptors: Descriptors = RECEIVING
) -> list[tuple[str, int]]:
    """(descriptor, word position) for one centroid, most extreme feature first."""
    terms = []
    for i in np.argsort(-np.abs(centroid)):
        if abs(centroid[i]) < MIN_Z:
            break
        hi, lo, order = descriptors.get(features[i], (None, None, 9))
        term = hi if centroid[i] > 0 else lo
        if term and term not in [t for t, _ in terms]:
            terms.append((term, order))
    return terms if terms else [("Balanced", 0)]


def compose(terms: list[tuple[str, int]], n: int) -> str:
    return " ".join(t for t, _ in sorted(terms[:n], key=lambda x: x[1]))


def name_clusters(
    Z: np.ndarray, labels: np.ndarray, features: list[str], descriptors: Descriptors = RECEIVING,
    suffix: str = "",
) -> dict[int, str]:
    """Map cluster id -> unique name. Noise (-1) is left out."""
    ids = sorted(int(c) for c in set(labels) if c >= 0)
    terms = {c: describe(Z[labels == c].mean(0), features, descriptors) for c in ids}
    n_terms = dict.fromkeys(ids, 2)
    while True:  # lengthen only names that collide
        names = {c: compose(terms[c], n_terms[c]) for c in ids}
        grow = [c for c in ids if list(names.values()).count(names[c]) > 1
                and n_terms[c] < len(terms[c])]
        if not grow:
            break
        for c in grow:
            n_terms[c] += 1
    seen: dict[str, int] = {}
    for c in ids:  # last resort: suffix remaining duplicates
        seen[names[c]] = seen.get(names[c], 0) + 1
        if seen[names[c]] > 1:
            names[c] = f"{names[c]} {seen[names[c]]}"
    return {c: f"{n} {suffix}" if suffix else n for c, n in names.items()}
