PY := uv run python -m src.cli

.PHONY: install ingest features archetypes train evaluate refresh app test lint all

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

evaluate:  # train + comparison models (slower; refreshes the lift tables)
	$(PY) train --evaluate

refresh:  # weekly: latest stats, injury reports and lines, then new picks
	$(PY) refresh

app:
	uv run streamlit run app/streamlit_app.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests app

all: ingest features archetypes evaluate
