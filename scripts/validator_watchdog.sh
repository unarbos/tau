#!/usr/bin/env bash
# Validator wedge watchdog. Restarts pm2 'validator' if dashboard_data.json
# (the natural heartbeat written by the publisher) has not been touched in
# WEDGE_THRESHOLD seconds.
#
# Default: 3600s (60 min). Validator publishes dashboard at the top of every
# poll iteration AND at every 15s heartbeat during a parallel duel, so under
# normal conditions the file refreshes every few seconds. The threshold is
# generous to absorb genuinely-long chain RPC stalls (substrate websocket
# can hang for several minutes during chain congestion) without trapping
# the validator in a restart loop -- a too-aggressive watchdog actively
# prevents recovery because the validator can't finish startup before being
# killed again.
set -euo pipefail
HEARTBEAT="${HEARTBEAT:-/home/const/subnet66/tau/workspace/validate/netuid-66/dashboard_data.json}"
WEDGE_THRESHOLD="${WEDGE_THRESHOLD:-3600}"
PM2_PROC="${PM2_PROC:-validator}"
LOG="${LOG:-/home/const/subnet66/tau/logs/watchdog.log}"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

while true; do
  if [[ -f "$HEARTBEAT" ]]; then
    mtime=$(stat -c '%Y' "$HEARTBEAT")
    now=$(date +%s)
    age=$(( now - mtime ))
    if (( age > WEDGE_THRESHOLD )); then
      echo "$(ts) WEDGE detected: $HEARTBEAT age=${age}s > ${WEDGE_THRESHOLD}s; restarting $PM2_PROC" >> "$LOG"
      pm2 restart "$PM2_PROC" --update-env >> "$LOG" 2>&1 || echo "$(ts) pm2 restart failed" >> "$LOG"
      sleep 300  # cooldown: validator startup (chain init + commitment scan)
                 # alone can take 60-90s; need a generous warm-up window.
    fi
  fi
  sleep 60
done
