PY := uv run python -m src.cli

.PHONY: install ingest features archetypes train evaluate refresh bets export schedule unschedule app test lint all

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

bets:  # paper bets: place on games starting within the hour, settle finished ones
	$(PY) bets

export:  # data/app.duckdb: the dashboard's tables only, for the hosted app
	$(PY) export-app

schedule:  # run `refresh` automatically on Tuesdays and Saturdays at 8 AM (macOS launchd)
	scripts/schedule_refresh.sh install

unschedule:
	scripts/schedule_refresh.sh uninstall

app:
	uv run streamlit run app/streamlit_app.py

test:
	uv run pytest -q

lint:
	uv run ruff check src tests app

all: ingest features archetypes evaluate
