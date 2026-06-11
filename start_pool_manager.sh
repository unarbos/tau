#!/bin/bash
exec doppler run -p arbos -c dev -- bash -lc '
set -euo pipefail
TAU_POLAR_HF_DATASET="Wejh/ninja-rollouts-polar"
VALIDATE_TASK_ARCHIVE_HF_DATASET="$TAU_POLAR_HF_DATASET"
TAU_ROLLOUT_HF_DATASET="$TAU_POLAR_HF_DATASET"
: "${HF_TOKEN:?Set HF_TOKEN for Hugging Face task archive uploads}"
: "${OPENROUTER_UPSTREAM_BASE_URL:?Set OPENROUTER_UPSTREAM_BASE_URL to the det endpoint base URL}"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY for the det endpoint}"
export OPENROUTER_UPSTREAM_BASE_URL OPENROUTER_API_KEY
exec /home/const/subnet66/.venv/bin/python -m cli pool-manager \
  --workspace-root /home/const/subnet66/tau \
  --solver-model deepseek-ai/DeepSeek-V4-Flash \
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
  --pool-filler-concurrency 32 \
  --pool-fill-during-duel \
  --docker-solver-start-concurrency 32
'
