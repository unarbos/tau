#!/bin/bash
exec doppler run -p arbos -c dev -- bash -lc '
set -euo pipefail
TAU_POLAR_HF_DATASET="Wejh/ninja-rollouts-polar"
VALIDATE_TASK_ARCHIVE_HF_DATASET="$TAU_POLAR_HF_DATASET"
TAU_ROLLOUT_HF_DATASET="$TAU_POLAR_HF_DATASET"
: "${HF_TOKEN:?Set HF_TOKEN for Hugging Face task archive uploads}"
exec /home/const/subnet66/.venv/bin/python -m cli pool-manager \
  --workspace-root /home/const/subnet66/tau \
  --solver-model minimax/minimax-m2.7 \
  --solver-provider-only minimax/fp8 \
  --solver-provider-disable-fallbacks \
  --poll-interval-seconds 10 \
  --task-pool-target 50 \
  --task-pool-static \
  --task-archive-enabled \
  --task-archive-hf-dataset "$VALIDATE_TASK_ARCHIVE_HF_DATASET" \
  --task-archive-per-hour 10 \
  --record-rollouts \
  --rollout-root /home/const/subnet66/tau/workspace/rollouts \
  --push-rollouts-to-hf \
  --rollout-hf-dataset "$TAU_ROLLOUT_HF_DATASET" \
  --pool-filler-concurrency 10
'
