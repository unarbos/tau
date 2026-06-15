#!/usr/bin/env bash
# Install PM2 dump sync (pm2 save only — never auto-starts stopped processes).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEPLOY="$ROOT/deploy"

if [[ "$(id -un)" != "const" ]]; then
  echo "Run as user const (or adjust paths)." >&2
  exit 1
fi

chmod +x "$ROOT/scripts/sync_pm2_dump.sh"

echo "Removing legacy auto-restart hook (if present)..."
sudo rm -f /etc/systemd/system/pm2-const.service.d/override.conf
sudo rmdir /etc/systemd/system/pm2-const.service.d 2>/dev/null || true

echo "Removing legacy ensure-critical timer (if present)..."
sudo systemctl disable --now pm2-ensure-critical.timer 2>/dev/null || true
sudo rm -f /etc/systemd/system/pm2-ensure-critical.timer
sudo rm -f /etc/systemd/system/pm2-ensure-critical.service

echo "Installing pm2-sync-dump timer..."
sudo cp "$DEPLOY/pm2-sync-dump.service" /etc/systemd/system/
sudo cp "$DEPLOY/pm2-sync-dump.timer" /etc/systemd/system/

sudo systemctl daemon-reload
sudo systemctl enable --now pm2-sync-dump.timer

echo "Running sync once now..."
"$ROOT/scripts/sync_pm2_dump.sh"

echo "Done."
systemctl status pm2-sync-dump.timer --no-pager | head -5
