#!/bin/bash
set -e
cd /home/const/subnet66/tau
source /home/const/subnet66/.venv/bin/activate
export GITHUB_TOKEN=$(doppler secrets get GITHUB_TOKEN_UNARBOS -p arbos -c dev --plain 2>/dev/null)
export OPENROUTER_API_KEY=$(doppler secrets get OPENROUTER_API_KEY -p arbos -c dev --plain 2>/dev/null)
export CURSOR_API_KEY=$(doppler secrets get CURSOR_API_KEY -p arbos -c dev --plain 2>/dev/null)

rm -f /home/const/subnet66/tau/workspace/validate/netuid-66/state.json
rm -rf /home/const/subnet66/tau/workspace/tasks/validate-*
rm -f /home/const/subnet66/tau/workspace/validate/netuid-66/duels/*.json

exec tau validate \
  --wallet-name sn66_owner \
  --wallet-hotkey default \
  --mock-set-weights \
  --max-duels 1 \
  --min-rounds 2 \
  --max-rounds 3 \
  --concurrency 2
