"""Standalone HTML report for one position's archetypes: UMAP scatter + centroid heatmap."""

from __future__ import annotations

import plotly.express as px
import plotly.graph_objects as go
import polars as pl

from src.archetypes.pipeline import SPECS
from src.config import OUT
from src.db import connect

LABELS = {
    "adot": "aDOT", "target_share": "Target share", "air_yards_share": "Air yards share",
    "yacoe": "YAC over exp.", "avg_separation": "Separation", "middle_target_rate":
    "Middle-field tgt rate", "rz_target_share": "RZ target share", "croe": "Catch rate over exp.",
    "epa_per_target": "EPA/target", "rush_epa": "Rush EPA/att", "ryoe_per_att": "RYOE/att",
    "route_participation": "Route participation", "explosive_run_rate": "Explosive run rate",
    "goal_line_share": "Goal-line share", "time_to_throw": "Time to throw",
    "scramble_rate": "Scramble rate", "cpoe": "CPOE", "sack_rate": "Sack rate",
    "epa_per_dropback": "EPA/dropback", "deep_pass_rate": "Deep pass rate",
    "pass_epa_allowed": "Pass EPA allowed", "rush_epa_allowed": "Rush EPA allowed",
    "pressure_rate": "Pressure rate", "blitz_rate": "Blitz rate",
    "heavy_box_rel": "8+ box rate (vs. league)", "adot_allowed": "aDOT allowed",
    "explosive_rate_allowed": "Explosive plays allowed", "takeaway_rate": "Takeaway rate",
    "third_down_rate_allowed": "3rd-down conv. allowed", "cpoe_allowed": "CPOE allowed",
}


def build(position: str = "WR") -> str:
    pos = position.lower()
    spec = SPECS[position]
    with connect(read_only=True) as con:
        df = con.execute(f"SELECT * FROM archetype_{pos}").pl()
        cent = con.execute(f"SELECT * FROM archetype_centroids_{pos} ORDER BY n DESC").pl()
        sel = con.execute(f"SELECT * FROM archetype_selection_{pos}").pl()

    order = cent["archetype"].to_list()
    hover = df.with_columns(
        label=pl.format(
            "{} '{}", "player_name", (pl.col("season") % 100).cast(pl.String).str.zfill(2)
        ),
        kind=pl.when("projected").then(pl.lit("2026 (projected)")).otherwise(pl.lit("2016–25")),
    )
    scatter = px.scatter(
        hover, x="umap_x", y="umap_y", color="archetype", symbol="kind",
        category_orders={"archetype": order}, hover_name="label",
        hover_data={"team": True, spec.volume_col: True,
                    **{f: ":.3f" for f in spec.features + spec.profile},
                    "membership_prob": ":.2f", "umap_x": False, "umap_y": False, "kind": False},
        opacity=0.8, height=650, title=f"{position} archetypes — UMAP projection (n={df.height})",
    )
    scatter.update_traces(marker=dict(size=7, line=dict(width=0.3, color="white")))
    scatter.update_layout(xaxis_title="", yaxis_title="", template="plotly_white",
                          legend_title_text="")

    zcols = [c for c in cent.columns if c.startswith("z_")]
    heat = go.Figure(go.Heatmap(
        z=cent.select(zcols).to_numpy(), x=[LABELS.get(c[2:], c[2:]) for c in zcols],
        y=[f"{a} (n={n})" for a, n in zip(cent["archetype"], cent["n"], strict=True)],
        colorscale="RdBu_r", zmid=0, zmin=-1.5, zmax=1.5,
        text=cent.select(zcols).to_numpy().round(2), texttemplate="%{text}",
        colorbar=dict(title="z"),
    ))
    heat.update_layout(title="Cluster centroids (z-scores)", height=420, template="plotly_white",
                       yaxis=dict(autorange="reversed"))

    table = sel.select("method", "param", "n_clusters",
                       pl.col("noise_frac").round(3), pl.col("bic").round(0),
                       pl.col("stability").round(3), pl.col("sil_embed").round(3),
                       pl.col("sil_z").round(3), "chosen")
    tfig = go.Figure(go.Table(
        header=dict(values=table.columns, fill_color="#eee"),
        cells=dict(values=[table[c].to_list() for c in table.columns]),
    ))
    tfig.update_layout(title="Model selection (GMM k by bootstrap stability, k≥4; "
                       "silhouette in z-space is the honest separation measure)", height=420)

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"archetypes_{pos}.html"
    with open(path, "w") as f:
        f.write(f"<html><head><meta charset='utf-8'><title>{position} archetypes</title></head>"
                "<body style='font-family:sans-serif;max-width:1200px;margin:auto'>")
        for i, fig in enumerate([scatter, heat, tfig]):
            f.write(fig.to_html(full_html=False, include_plotlyjs=(i == 0)))
        f.write("</body></html>")
    return str(path)
