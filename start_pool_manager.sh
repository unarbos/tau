#!/bin/bash
exec doppler run -p sn66 -c prd -- bash -lc '
set -euo pipefail
umask 002
TAU_POLAR_HF_DATASET="Wejh/ninja-rollouts-polar"
TAU_ROLLOUT_HF_DATASET="$TAU_POLAR_HF_DATASET"
: "${HF_TOKEN:?Set HF_TOKEN for Hugging Face task archive uploads}"
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in Doppler}"
: "${SOLVER_UPSTREAM_API_KEY:?Set SOLVER_UPSTREAM_API_KEY (self-hosted Qwen endpoint key) in Doppler}"
# Self-hosted Qwen3-32B for solver + generator + eval (routed via SELF_HOSTED_MODEL).
# Keep endpoint URLs in local env/Doppler. SOLVER_UPSTREAM_BASE_URLS may contain a
# comma-separated GPU list; each solve pins to one upstream for all tool turns so
# prefix cache stays useful.
: "${SOLVER_UPSTREAM_BASE_URL:?Set SOLVER_UPSTREAM_BASE_URL in local env/Doppler}"
export SOLVER_UPSTREAM_BASE_URL SOLVER_UPSTREAM_BASE_URLS
export SELF_HOSTED_MODEL=Qwen/Qwen3-32B
export GENERATOR_MODEL=Qwen/Qwen3-32B
export EVAL_MODEL=Qwen/Qwen3-32B
# OpenRouter base for any non-self-hosted calls (kept consistent with the validator).
export OPENROUTER_UPSTREAM_BASE_URL=https://openrouter.ai/api/v1
export SOLVER_SHELL_TOOLS=true
export SOLVER_TEMPERATURE=0
export SOLVER_EMPTY_RESPONSE_RETRIES=5
# Cap concurrent GitHub-sourced task generation (commit sampling) independently
# of solve concurrency, to avoid GitHub secondary rate-limit pauses when solve
# concurrency is scaled up. Tunable; remove to restore unbounded generation.
export TAU_POOL_GENERATION_CONCURRENCY=6
exec /home/const/subnet66/.venv/bin/python -m cli pool-manager \
  --workspace-root /home/const/subnet66/tau \
  --solver-model Qwen/Qwen3-32B \
  --poll-interval-seconds 10 \
  --task-pool-target 50 \
  --task-pool-static \
  --record-rollouts \
  --rollout-root /home/const/subnet66/tau/workspace/rollouts \
  --push-rollouts-to-hf \
  --rollout-hf-dataset "$TAU_ROLLOUT_HF_DATASET" \
  --pool-filler-concurrency 25 \
  --docker-solver-start-concurrency 25
'
