#!/bin/bash
exec doppler run -p arbos -c dev -- \
  /home/const/subnet66/.venv/bin/python -m cli validate \
  --workspace-root /home/const/subnet66/tau \
  --wallet-name sn66_owner \
  --wallet-hotkey default \
  --solver-model minimax/minimax-m2.7 \
  --solver-provider-sort throughput \
  --solver-provider-only minimax/highspeed \
  --max-concurrency 1 \
  --round-concurrency 25 \
  --candidate-timeout-streak-limit 5 \
  --poll-interval-seconds 600 \
  --task-pool-target 50 \
  --task-pool-static \
  --task-pool-fill-from-saved \
  --task-pool-refresh-count 0 \
  --task-pool-refresh-interval-seconds 0 \
  --duel-rounds 50 \
  --win-margin 3 \
  --min-commitment-block 7951985 \
  --hotkey-spent-since-block 8104340 \
  --pool-filler-concurrency 2 \
  --watch-github-prs \
  --github-pr-only \
  --github-pr-cleanup \
  --github-pr-cleanup-stale-after-hours 6 \
  --github-pr-repo unarbos/ninja \
  --github-pr-base main
