#!/bin/bash
exec doppler run -p arbos -c dev -- \
  /home/const/subnet66/.venv/bin/python -m cli validate \
  --workspace-root /home/const/subnet66/tau \
  --wallet-name sn66_owner \
  --wallet-hotkey default \
  --max-concurrency 1 \
  --round-concurrency 100 \
  --task-pool-target 150 \
  --duel-rounds 100 \
  --win-margin 8 \
  --min-commitment-block 0 \
  --pool-filler-concurrency 24
