#!/usr/bin/env bash
# Persist the live PM2 process list so resurrect honors manual stop/start state.
# Does not start or restart any process.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${SYNC_PM2_LOG:-$ROOT/logs/sync_pm2_dump.log}"
VALIDATOR_LOG="${PM2_HOME:-$HOME/.pm2}/logs/validator-error.log"
MAX_VALIDATOR_LOG_BYTES="${MAX_VALIDATOR_LOG_BYTES:-157286400}" # 150 MiB

mkdir -p "$(dirname "$LOG")"

ts() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }

log() {
  echo "$(ts) $*" >>"$LOG"
}

maybe_flush_validator_log() {
  [[ -f "$VALIDATOR_LOG" ]] || return 0
  local size
  size="$(stat -c '%s' "$VALIDATOR_LOG")"
  if (( size > MAX_VALIDATOR_LOG_BYTES )); then
    log "flushing validator logs (${size} bytes > ${MAX_VALIDATOR_LOG_BYTES})"
    pm2 flush validator >>"$LOG" 2>&1 || true
  fi
}

main() {
  if ! command -v pm2 >/dev/null 2>&1; then
    log "pm2 not found; skipping"
    return 0
  fi

  maybe_flush_validator_log
  pm2 save >>"$LOG" 2>&1
  log "pm2 save completed"
}

main "$@"
