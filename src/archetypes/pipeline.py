"""Archetype discovery: standardize -> UMAP -> HDBSCAN vs GMM -> label from centroid z-scores.

Models are fit on completed seasons only. The in-progress season is projected onto the
fitted scaler/UMAP/GMM, with its volume threshold prorated by weeks completed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import polars as pl
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.archetypes.naming import (
    DEFENSE,
    PASSING,
    RECEIVING,
    RUSHING,
    Descriptors,
    name_clusters,
)
from src.config import OUT
from src.db import connect
from src.features.defense import DEFENSE_FEATURES
from src.features.passing import PASSING_FEATURES
from src.features.receiving import RECEIVING_FEATURES
from src.features.rushing import RUSHING_FEATURES

log = logging.getLogger(__name__)
SEED = 42
STABILITY_TOL = 0.02


@dataclass
class PositionSpec:
    position: str
    table: str
    volume_col: str
    min_volume: int
    features: list[str]  # clustered on
    descriptors: Descriptors = field(default_factory=lambda: RECEIVING)
    profile: list[str] = field(default_factory=list)  # z-scored and reported, not clustered
    gmm_k: range = field(default_factory=lambda: range(3, 10))
    min_k: int = 4  # fewest archetypes worth naming
    first_season: int = 2016  # earliest season with every feature populated
    name_suffix: str = ""  # appended to every archetype name, e.g. "Defense"
    min_floor: int = 10  # smallest volume accepted for an in-progress season
    hdbscan_sizes: tuple[int, ...] = (15, 25, 40)


SPECS: dict[str, PositionSpec] = {
    "WR": PositionSpec("WR", "feat_receiving", "targets", 50, RECEIVING_FEATURES),
    "TE": PositionSpec("TE", "feat_receiving", "targets", 40, RECEIVING_FEATURES),
    # NGS rush yards over expected starts in 2018
    "RB": PositionSpec("RB", "feat_rushing", "touches", 100, RUSHING_FEATURES, RUSHING,
                       first_season=2018),
    "QB": PositionSpec("QB", "feat_passing", "dropbacks", 200, PASSING_FEATURES, PASSING,
                       hdbscan_sizes=(8, 12, 20)),
    # Whole-team defenses (one row per team-season), volume = games played
    "DEF": PositionSpec("DEF", "feat_defense", "games", 12, DEFENSE_FEATURES, DEFENSE,
                        gmm_k=range(3, 9), hdbscan_sizes=(8, 12, 20), name_suffix="Defense",
                        min_floor=3),
}


def season_progress(con) -> tuple[int, float]:
    """(current season, fraction of regular-season weeks completed)."""
    season, weeks = con.execute(
        """SELECT max(season), max(week) FILTER (WHERE season = (SELECT max(season) FROM
               raw_schedules) AND game_type = 'REG' AND result IS NOT NULL)
           FROM raw_schedules"""
    ).fetchone()
    return season, min(1.0, (weeks or 0) / 18)


def load_players(con, spec: PositionSpec) -> tuple[pl.DataFrame, int]:
    current, progress = season_progress(con)
    current_min = max(spec.min_floor, round(spec.min_volume * progress))
    df = con.execute(
        f"""SELECT * FROM {spec.table}
            WHERE position = ? AND season >= ?
              AND ({spec.volume_col} >= ? AND season < ? OR {spec.volume_col} >= ? AND season = ?)
            ORDER BY season, player_id""",
        [spec.position, spec.first_season, spec.min_volume, current, current_min, current],
    ).pl()
    # A full season in the current year means nothing is in progress.
    if progress >= 1.0:
        current += 1
    return df, current


def standardize(fit: pl.DataFrame, apply: pl.DataFrame, features: list[str]):
    """Median-impute then z-score using statistics from the fit set only."""
    med = fit.select(features).median()
    def prep(d):
        return d.select(pl.col(f).fill_null(med[f][0]) for f in features).to_numpy()
    scaler = StandardScaler().fit(prep(fit))
    return scaler.transform(prep(fit)), scaler.transform(prep(apply)) if apply.height else None


def gmm_stability(E: np.ndarray, labels: np.ndarray, k: int, reps: int = 10) -> float:
    """Mean ARI between full-data labels and labels from GMMs refit on 80% subsamples."""
    rng = np.random.default_rng(SEED)
    scores = []
    for r in range(reps):
        idx = rng.choice(len(E), int(0.8 * len(E)), replace=False)
        g = GaussianMixture(k, n_init=3, random_state=r).fit(E[idx])
        scores.append(adjusted_rand_score(labels, g.predict(E)))
    return float(np.mean(scores))


def select_model(Z: np.ndarray, E: np.ndarray, spec: PositionSpec):
    """Fit candidate HDBSCAN and GMM models on the clustering embedding E.

    Silhouette is reported both in embedding space and in the original z-space; the
    z-space number is the honest one since UMAP inflates cluster separation. BIC on a
    UMAP embedding keeps falling with k (UMAP manufactures local density), so GMM k is
    chosen by bootstrap stability among k >= min_k; BIC is reported for reference.
    """
    import hdbscan  # imported here: slow to load, and the dashboard only needs SPECS

    rows, fitted = [], {}
    for k in spec.gmm_k:
        g = GaussianMixture(k, covariance_type="full", n_init=5, random_state=SEED).fit(E)
        lab = g.predict(E)
        rows.append(dict(method="gmm", param=k, n_clusters=k, noise_frac=0.0, bic=g.bic(E),
                         stability=gmm_stability(E, lab, k),
                         sil_embed=silhouette_score(E, lab), sil_z=silhouette_score(Z, lab)))
        fitted[("gmm", k)] = g
    for size in spec.hdbscan_sizes:
        h = hdbscan.HDBSCAN(min_cluster_size=size, min_samples=5, prediction_data=True).fit(E)
        lab = h.labels_
        mask = lab >= 0
        n = len(set(lab[mask]))
        ok = n >= 2
        rows.append(dict(method="hdbscan", param=size, n_clusters=n, noise_frac=1 - mask.mean(),
                         bic=None, stability=None,
                         sil_embed=silhouette_score(E[mask], lab[mask]) if ok else None,
                         sil_z=silhouette_score(Z[mask], lab[mask]) if ok else None))
        fitted[("hdbscan", size)] = h
    sel = pl.DataFrame(rows)

    gmm = sel.filter((pl.col("method") == "gmm") & (pl.col("n_clusters") >= spec.min_k))
    # Stability estimates are noisy; among near-ties prefer the simpler model.
    best_gmm = (
        gmm.filter(pl.col("stability") >= gmm["stability"].max() - STABILITY_TOL)
        .sort("n_clusters")
        .row(0, named=True)
    )
    hdb = sel.filter((pl.col("method") == "hdbscan") & (pl.col("n_clusters") >= spec.min_k)
                     & (pl.col("noise_frac") <= 0.25)).sort("sil_z", descending=True)
    choice = best_gmm
    # HDBSCAN wins only if it separates the original feature space clearly better.
    if hdb.height and hdb["sil_z"][0] > best_gmm["sil_z"] + 0.05:
        choice = hdb.row(0, named=True)
    sel = sel.with_columns(
        chosen=(pl.col("method") == choice["method"]) & (pl.col("param") == choice["param"])
    )
    return sel, choice["method"], fitted[(choice["method"], choice["param"])]


def predict(model, method: str, E: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if method == "gmm":
        p = model.predict_proba(E)
        return p.argmax(1), p.max(1)
    import hdbscan

    lab, prob = hdbscan.approximate_predict(model, E)
    return lab, prob


def run_position(con, spec: PositionSpec) -> pl.DataFrame:
    import umap

    df, current = load_players(con, spec)
    fit_df = df.filter(pl.col("season") < current)
    proj_df = df.filter(pl.col("season") >= current)
    log.info("%s: fit on %d player-seasons, projecting %d", spec.position, fit_df.height,
             proj_df.height)

    cols = spec.features + spec.profile
    Zall, Zpall = standardize(fit_df, proj_df, cols)
    nf = len(spec.features)
    Z, Zp = Zall[:, :nf], (Zpall[:, :nf] if Zpall is not None else None)
    viz = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=SEED).fit(Z)
    emb = umap.UMAP(n_components=5, n_neighbors=30, min_dist=0.0, random_state=SEED).fit(Z)
    E = emb.embedding_

    sel, method, model = select_model(Z, E, spec)
    log.info("%s model selection:\n%s", spec.position, sel)
    labels, prob = predict(model, method, E)

    names = name_clusters(Z, labels, spec.features, spec.descriptors, spec.name_suffix)

    def frame(base: pl.DataFrame, Zm, XY, lab, pr, projected: bool):
        return base.select(["player_id", "player_name", "season", "team", "position",
                            spec.volume_col, *cols]).with_columns(
            pl.Series("cluster", lab).cast(pl.Int32),
            pl.Series("membership_prob", pr),
            pl.Series("umap_x", XY[:, 0]),
            pl.Series("umap_y", XY[:, 1]),
            pl.lit(projected).alias("projected"),
            *[pl.Series(f"z_{f}", Zm[:, i]) for i, f in enumerate(cols)],
        )

    out = frame(fit_df, Zall, viz.embedding_, labels, prob, False)
    if Zp is not None:
        lp, pp = predict(model, method, emb.transform(Zp))
        out = pl.concat([out, frame(proj_df, Zpall, viz.transform(Zp), lp, pp, True)])
    out = out.with_columns(
        archetype=pl.col("cluster").replace_strict(
            names, default="Unassigned", return_dtype=pl.String
        )
    )

    centroids = (
        out.filter(~pl.col("projected") & (pl.col("cluster") >= 0))
        .group_by("cluster", "archetype")
        .agg(pl.len().alias("n"), *[pl.col(f"z_{f}").mean() for f in cols],
             *[pl.col(f).mean() for f in cols])
        .sort("n", descending=True)
    )
    pos = spec.position.lower()
    con.register("_out", out.to_arrow())
    con.execute(f"CREATE OR REPLACE TABLE archetype_{pos} AS SELECT * FROM _out")
    con.register("_cent", centroids.to_arrow())
    con.execute(f"CREATE OR REPLACE TABLE archetype_centroids_{pos} AS SELECT * FROM _cent")
    con.register("_sel", sel.with_columns(pl.lit(spec.position).alias("position")).to_arrow())
    con.execute(f"CREATE OR REPLACE TABLE archetype_selection_{pos} AS SELECT * FROM _sel")
    OUT.mkdir(parents=True, exist_ok=True)
    out.write_parquet(OUT / f"archetype_{pos}.parquet")
    return out


def run(positions: list[str] | None = None) -> None:
    with connect() as con:
        for pos, spec in SPECS.items():
            if positions and pos not in positions:
                continue
            run_position(con, spec)
