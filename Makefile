PY := uv run python -m src.cli

.PHONY: install ingest features archetypes train app test lint all

install:
	uv sync

ingest:
	$(PY) ingest

features:
	$(PY) features

archetypes:
	$(PY) archetypes

train:
	$(PY) train

app:
	uv run streamlit run app/streamlit_app.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests app

all: ingest features archetypes train
