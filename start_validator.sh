#!/bin/bash
exec doppler run -p arbos -c dev -- \
  env \
    OPENROUTER_UPSTREAM_BASE_URL="${OPENROUTER_UPSTREAM_BASE_URL:?Set OPENROUTER_UPSTREAM_BASE_URL to the det endpoint base URL}" \
    OPENROUTER_API_KEY="${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY for the det endpoint}" \
    PRIVATE_SUBMISSION_JUDGE_MODEL=deepseek-ai/DeepSeek-V4-Flash \
  /home/const/subnet66/.venv/bin/python -m cli validate \
  --workspace-root /home/const/subnet66/tau \
  --wallet-name sn66_owner \
  --wallet-hotkey default \
  --solver-model deepseek-ai/DeepSeek-V4-Flash \
  --max-concurrency 1 \
  --round-concurrency 50 \
  --docker-solver-start-concurrency 32 \
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
