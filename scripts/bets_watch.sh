#!/usr/bin/env bash
# Keeps paper bets on time for .github/workflows/bets.yml. GitHub delays and drops frequent
# scheduled runs (a 15-minute schedule ran about every 6 hours), so instead one run stays up
# for most of the 6-hour job limit: it runs the bet step, sleeps until the step says it's next
# needed (an hour before the next kickoff, or a score check while a game is on), and repeats.
# At the end it reports whether the next run is due within HANDOFF_HOURS; the workflow then
# starts a fresh run to carry on. Expects data/bets.parquet (if any) already downloaded.
set -uo pipefail

WATCH_MINUTES=${WATCH_MINUTES:-335}  # stay under the 360-minute job limit
HANDOFF_HOURS=${HANDOFF_HOURS:-24}  # longer gaps (midweek) are left to the backstop schedule
deadline=$(( $(date +%s) + WATCH_MINUTES * 60 ))
out=${GITHUB_OUTPUT:-/dev/null}

publish() {  # upload the ledger if it changed: the backup first, so one copy is always whole
  [ -f data/bets.parquet ] || return 0
  sha256sum -c --status data/bets.sha 2>/dev/null && return 0
  cp data/bets.parquet data/bets-backup.parquet
  gh release upload app-data data/bets-backup.parquet --clobber &&
    gh release upload app-data data/bets.parquet --clobber &&
    sha256sum data/bets.parquet > data/bets.sha
}

while :; do
  rm -f data/next_run
  # Fresh predictions every round, so Saturday's refresh is used for Sunday's bets
  if gh release download app-data -p app.duckdb -D data --clobber &&
     uv run python -m src.cli bets --next-run data/next_run; then
    publish || echo "::warning::couldn't publish the ledger; will retry next round"
  else
    echo "::warning::bet step failed; retrying in 10 minutes"
  fi
  now=$(date +%s)
  if [ ! -f data/next_run ]; then
    next=$(( now + 600 ))
  else
    next=$(cat data/next_run)
    if [ -z "$next" ]; then
      echo "Nothing left to bet or settle this season."
      echo "handoff=false" >> "$out"
      exit 0
    fi
  fi
  [ "$next" -ge "$deadline" ] && break
  echo "Sleeping until $(date -d "@$next" 2>/dev/null || date -r "$next")"
  sleep $(( next > now ? next - now : 0 ))
done

if [ $(( next - $(date +%s) )) -lt $(( HANDOFF_HOURS * 3600 )) ]; then
  echo "Next run due $(date -d "@$next" 2>/dev/null || date -r "$next"); handing off."
  echo "handoff=true" >> "$out"
else
  echo "Next run isn't due for over $HANDOFF_HOURS hours; leaving it to the schedule."
  echo "handoff=false" >> "$out"
fi
