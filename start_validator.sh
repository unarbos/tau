#!/bin/bash
exec doppler run -p arbos -c dev -- \
  /home/const/subnet66/.venv/bin/python -m cli validate \
  --workspace-root /home/const/subnet66/tau \
  --wallet-name sn66_owner \
  --wallet-hotkey default \
  --solver-model minimax/minimax-m2.7 \
  --solver-provider-only minimax/fp8 \
  --solver-provider-disable-fallbacks \
  --max-concurrency 1 \
  --round-concurrency 25 \
  --candidate-timeout-streak-limit 10 \
  --poll-interval-seconds 600 \
  --task-pool-target 50 \
  --task-pool-static \
  --record-rollouts \
  --rollout-root /home/const/subnet66/tau/workspace/rollouts \
  --duel-rounds 50 \
  --win-margin 3 \
  --min-commitment-block 7951985 \
  --hotkey-spent-since-block 8104340 \
  --watch-private-submissions \
  --private-submission-only \
  --publish-repo unarbos/ninja \
  --publish-base main
