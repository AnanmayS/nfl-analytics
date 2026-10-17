"""Command-line entry point: python -m src.cli <command>."""

import logging
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


@app.command()
def ingest(
    datasets: Annotated[list[str] | None, typer.Option("--dataset", "-d")] = None,
    force: bool = False,
):
    """Pull nflverse data to data/raw and load raw_* tables into DuckDB."""
    from src.ingest import nflverse

    nflverse.run(datasets, force=force)


@app.command()
def features(
    positions: Annotated[list[str] | None, typer.Option("--position", "-p")] = None,
):
    """Build player-season feature tables (feat_<pos>) in DuckDB."""
    from src.features import player

    player.build(positions)


@app.command()
def archetypes(
    positions: Annotated[list[str] | None, typer.Option("--position", "-p")] = None,
):
    """Cluster player-seasons into archetypes; writes archetype_<pos> tables."""
    from src.archetypes import pipeline

    pipeline.run(positions)
    from src.analysis import archetype_report

    for pos in positions or list(pipeline.SPECS):
        typer.echo(f"report: {archetype_report.build(pos)}")


@app.command()
def train():
    """Build pre-kickoff game features and run the walk-forward game models."""
    from src.models import train as t

    t.run()


if __name__ == "__main__":
    app()
