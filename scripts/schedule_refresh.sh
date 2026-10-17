#!/bin/bash
# Schedule `make refresh` with macOS launchd: Tuesdays and Saturdays at 8:00 AM.
#   scripts/schedule_refresh.sh install | uninstall | status | run-now
# If the Mac is asleep at 8:00, launchd runs the job when it wakes. If it's shut down,
# that run is skipped. Output goes to data/refresh_scheduled.log.
set -euo pipefail

LABEL="com.nflanalytics.refresh"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
UV="$(command -v uv || echo "$HOME/.local/bin/uv")"
LOG="$ROOT/data/refresh_scheduled.log"

case "${1:-status}" in
  install)
    mkdir -p "$(dirname "$PLIST")" "$ROOT/data"
    cat > "$PLIST" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key>
  <array>
    <string>$UV</string><string>run</string><string>python</string>
    <string>-m</string><string>src.cli</string><string>refresh</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>EnvironmentVariables</key>
  <dict><key>PATH</key><string>$(dirname "$UV"):/usr/local/bin:/usr/bin:/bin</string></dict>
  <key>StartCalendarInterval</key>
  <array>
    <dict><key>Weekday</key><integer>2</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
    <dict><key>Weekday</key><integer>6</integer><key>Hour</key><integer>8</integer><key>Minute</key><integer>0</integer></dict>
  </array>
  <key>StandardOutPath</key><string>$LOG</string>
  <key>StandardErrorPath</key><string>$LOG</string>
</dict>
</plist>
PLIST
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    echo "Scheduled: Tuesdays and Saturdays at 8:00 AM (log: $LOG)"
    ;;
  uninstall)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
    echo "Schedule removed"
    ;;
  run-now)
    launchctl kickstart "gui/$(id -u)/$LABEL"
    echo "Started; follow progress in $LOG"
    ;;
  status)
    if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then
      echo "Installed: Tuesdays and Saturdays at 8:00 AM"
      launchctl print "gui/$(id -u)/$LABEL" | grep -E "state =|last exit code" || true
    else
      echo "Not installed (run: make schedule)"
    fi
    ;;
  *) echo "usage: $0 install|uninstall|status|run-now" >&2; exit 2 ;;
esac
