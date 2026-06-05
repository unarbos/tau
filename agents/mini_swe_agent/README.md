# mini-swe-agent competition baseline

This directory is the **reference harness** for the upcoming competition format. Miners will submit multi-file bundles ([tau#67](https://github.com/unarbos/tau/pull/67)) that follow the same contract as this wrapper—not the monolithic `ninja/agent.py` harness.

## Contract

`solve(repo_path, issue, model, api_base, api_key, timeout_seconds=None, deadline_epoch=None)` must return:

```python
{"success": bool, "message": str, "diff": str}
```

- **`diff`**: unified git diff of workspace changes (`git diff --binary`).
- **`success`**: `True` when the diff is non-empty (partial patches count).
- **`message`**: human-readable status for logs and dashboards.

## What this baseline does

1. Invokes the real [`mini-swe-agent`](https://github.com/SWE-agent/mini-swe-agent) CLI (`mini`) inside the task repo.
2. Configures LiteLLM against the validator-provided OpenAI-compatible proxy (`api_base` / `api_key`).
3. Honors duel time limits:
   - `timeout_seconds` and `deadline_epoch` come from the validator via `TAU_AGENT_*` env vars (see `docker_solver.py`).
   - Reserves 15% (20–90s) of the outer budget for cleanup before killing the process group.
   - Prepends a time-budget note to the task prompt so the agent knows to finish early.
4. On timeout, terminates the `mini` process group (SIGTERM → SIGKILL) and returns whatever diff exists so far.

## Local test

```bash
python -m unittest tests.test_mini_swe_agent -v
```

Tests stub `mini` on `PATH` so no LLM or package install is required.

## Related work

- Multi-file private submissions: tau PR for bundle intake and validation.
- Docker solver installs `mini-swe-agent` in the solver image and forwards timeout kwargs to any harness that accepts them.
