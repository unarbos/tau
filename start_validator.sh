#!/bin/bash
exec doppler run -p arbos -c dev -- \
  env DIFF_JUDGE_MODEL_CONCURRENCY=15 \
  DIFF_JUDGE_SANITIZER_MODEL=openai/gpt-5.4-nano \
  /home/const/subnet66/.venv/bin/python -m cli validate \
  --workspace-root /home/const/subnet66/tau \
  --wallet-name sn66_owner \
  --wallet-hotkey default \
  --solver-model minimax/minimax-m2.7 \
  --solver-provider-sort throughput \
  --solver-provider-only minimax/highspeed \
  --solver-provider-disable-fallbacks \
  --max-concurrency 1 \
  --round-concurrency 15 \
  --candidate-timeout-streak-limit 5 \
  --poll-interval-seconds 600 \
  --task-pool-target 0 \
  --task-pool-refresh-count 0 \
  --task-pool-refresh-interval-seconds 3600 \
  --duel-rounds 50 \
  --win-margin 3 \
  --min-commitment-block 7951985 \
  --hotkey-spent-since-block 8104340 \
  --pool-filler-concurrency 24 \
  --watch-github-prs \
  --github-pr-only \
  --github-pr-cleanup \
  --github-pr-repo unarbos/ninja \
  --github-pr-base main
