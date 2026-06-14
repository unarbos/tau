#!/bin/bash
exec doppler run -p arbos -c dev -- bash -lc '
set -euo pipefail
TAU_POLAR_HF_DATASET="Wejh/ninja-rollouts-polar"
TAU_ROLLOUT_HF_DATASET="$TAU_POLAR_HF_DATASET"
: "${HF_TOKEN:?Set HF_TOKEN for Hugging Face task archive uploads}"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in Doppler}"
export OPENROUTER_UPSTREAM_BASE_URL=https://openrouter.ai/api/v1
export OPENROUTER_PROVIDER_ONLY=google-ai-studio
export OPENROUTER_PROVIDER_ALLOW_FALLBACKS=false
export SOLVER_SHELL_TOOLS=true
export SOLVER_TEMPERATURE=0
export SOLVER_EMPTY_RESPONSE_RETRIES=5
export GENERATOR_MODEL=google/gemini-3.1-flash-lite
export EVAL_MODEL=google/gemini-3.1-flash-lite
# Cap concurrent GitHub-sourced task generation (commit sampling) independently
# of solve concurrency, to avoid GitHub secondary rate-limit pauses when solve
# concurrency is scaled up. Tunable; remove to restore unbounded generation.
export TAU_POOL_GENERATION_CONCURRENCY=6
exec /home/const/subnet66/.venv/bin/python -m cli pool-manager \
  --workspace-root /home/const/subnet66/tau \
  --solver-model google/gemini-3.1-flash-lite \
  --solver-provider-only google-vertex/global \
  --solver-provider-disable-fallbacks \
  --poll-interval-seconds 10 \
  --task-pool-target 50 \
  --task-pool-static \
  --record-rollouts \
  --rollout-root /home/const/subnet66/tau/workspace/rollouts \
  --push-rollouts-to-hf \
  --rollout-hf-dataset "$TAU_ROLLOUT_HF_DATASET" \
  --pool-filler-concurrency 32 \
  --docker-solver-start-concurrency 32
'
