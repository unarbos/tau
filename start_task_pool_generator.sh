#!/bin/bash
exec doppler run -p arbos -c dev -- \
  env DIFF_JUDGE_MODEL_CONCURRENCY=15 \
  DIFF_JUDGE_SANITIZER_MODEL=openai/gpt-5.4-nano \
  /home/const/subnet66/.venv/bin/python -m cli generate-pool \
  --workspace-root /home/const/subnet66/tau \
  --solver-model minimax/minimax-m2.7 \
  --solver-provider-sort throughput \
  --solver-provider-only minimax/highspeed \
  --solver-provider-disable-fallbacks \
  --task-pool-target 200 \
  --pool-filler-concurrency 24 \
  --no-task-pool-fill-from-saved \
  --task-pool-refresh-count 6 \
  --task-pool-refresh-interval-seconds 3600
