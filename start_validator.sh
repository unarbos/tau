#!/bin/bash
exec doppler run -p arbos -c dev -- bash -lc '
set -euo pipefail
: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in Doppler}"
export OPENROUTER_UPSTREAM_BASE_URL=https://openrouter.ai/api/v1
export OPENROUTER_PROVIDER_ONLY=google-ai-studio
export OPENROUTER_PROVIDER_ALLOW_FALLBACKS=false
export SOLVER_SHELL_TOOLS=true
export SOLVER_TEMPERATURE=0
export SOLVER_EMPTY_RESPONSE_RETRIES=5
export SOLVER_RATE_LIMIT_RETRIES=6
export GENERATOR_MODEL=google/gemini-3.1-flash-lite
export EVAL_MODEL=google/gemini-3.1-flash-lite
export PRIVATE_SUBMISSION_JUDGE_MODEL=google/gemini-3.1-flash-lite
exec /home/const/subnet66/.venv/bin/python -m cli validate \
  --workspace-root /home/const/subnet66/tau \
  --wallet-name sn66_owner \
  --wallet-hotkey default \
  --solver-model google/gemini-3.1-flash-lite \
  --solver-provider-only google-vertex/global \
  --solver-provider-disable-fallbacks \
  --max-concurrency 1 \
  --round-concurrency 50 \
  --docker-solver-start-concurrency 50 \
  --candidate-timeout-streak-limit 10 \
  --poll-interval-seconds 600 \
  --task-pool-target 50 \
  --task-pool-static \
  --record-rollouts \
  --rollout-root /home/const/subnet66/tau/workspace/rollouts \
  --duel-rounds 50 \
  --win-margin 6 \
  --min-commitment-block 7951985 \
  --hotkey-spent-since-block 8104340 \
  --watch-private-submissions \
  --private-submission-only \
  --publish-repo unarbos/ninja \
  --publish-base main
'
