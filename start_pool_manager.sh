#!/bin/bash
exec doppler run -p arbos -c dev -- bash -lc '
set -euo pipefail
: "${VALIDATE_TASK_ARCHIVE_HF_DATASET:?Set VALIDATE_TASK_ARCHIVE_HF_DATASET to owner/name}"
: "${HF_TOKEN:?Set HF_TOKEN for Hugging Face task archive uploads}"
exec /home/const/subnet66/.venv/bin/python -m cli pool-manager \
  --workspace-root /home/const/subnet66/tau \
  --solver-model minimax/minimax-m2.7 \
  --solver-provider-only minimax/fp8 \
  --solver-provider-disable-fallbacks \
  --poll-interval-seconds 10 \
  --task-pool-target 50 \
  --task-pool-static \
  --task-pool-fill-from-saved \
  --task-archive-enabled \
  --task-archive-hf-dataset "$VALIDATE_TASK_ARCHIVE_HF_DATASET" \
  --task-archive-per-hour 10 \
  --pool-filler-concurrency 25
'
