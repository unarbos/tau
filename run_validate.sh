#!/bin/bash
set -e
cd /home/const/subnet66/tau

rm -f /home/const/subnet66/tau/workspace/validate/netuid-66/state.json
rm -rf /home/const/subnet66/tau/workspace/tasks/validate-*
rm -f /home/const/subnet66/tau/workspace/validate/netuid-66/duels/*.json

exec doppler run -p arbos -c dev -- \
  /home/const/subnet66/.venv/bin/python -m cli validate \
  --wallet-name sn66_owner \
  --wallet-hotkey default \
  --mock-set-weights \
  --max-duels 1 \
  --min-rounds 2 \
  --max-rounds 3 \
  --concurrency 2
