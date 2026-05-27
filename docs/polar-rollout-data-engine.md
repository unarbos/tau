# Polar-Style Rollout Data Engine for tau/ninja

## Sources

- tau inspected at branch `polar-rollout-data-engine-design`.
- ninja inspected from local sibling repo `/home/const/subnet66/ninja`, branch `main`.
- Polar reference: arXiv `2605.24220`, "Polar: Agentic RL on Any Harness at Scale".

Polar's important lesson is not "save more files." It is: keep the harness black-box, proxy the model calls, reconstruct token-faithful trajectories, and expose asynchronous rollouts that trainers can consume. tau/ninja are already close to the black-box-harness setup; they are not yet close to token-faithful training data.

## Current-State Map

### ninja

`ninja` is correctly small and miner-facing.

- `agent.py` is the miner-editable single-file harness.
- `solve(repo_path, issue, model, api_base, api_key, ...)` returns `patch`, `logs`, `steps`, `cost`, and `success`.
- It calls the validator-managed OpenAI-compatible endpoint through `chat_completion`.
- It executes shell commands through `run_command`.
- It returns miner-authored logs, which are useful for debugging but untrusted as training/audit truth.
- It does not own task generation, Docker isolation, scoring, R2/HF publishing, model routing, or promotion logic.

This boundary should stay. Do not put rollout storage, HF exporters, replay, or trainer APIs in ninja.

### tau

tau already owns the validator side of the stack:

- Task generation and task workspaces live in `src/pipeline.py`, `src/task_generation.py`, and `src/workspace.py`.
- Docker execution lives in `src/docker_solver.py`.
- Managed inference lives in `src/openrouter_proxy.py`.
- Solver result normalization lives in `src/solver_runner.py`.
- King/challenger validation and scoring live in `src/validate.py`.
- Pairwise/diff judging lives mostly in `src/validate.py` and `src/eval.py`.
- Public artifact publishing and sanitization live in `src/r2.py`.
- HF task archival lives in `src/task_pool_manager.py`.

tau currently records strong eval artifacts:

- task metadata, prompt text, commit metadata, and reference patch in the task workspace
- solution repo, final solution diff, solve metadata, raw output, usage summary, and optional `rollout.jsonl`
- compare/eval JSON
- duel JSON with round winners, score fields, LLM judge winner/rationale/error/model, exit reasons, and timing
- public R2 artifacts with private task/reference/baseline/rollout/raw-output fields stripped
- HF archived task rows containing task metadata, commit metadata, pool metadata, king metadata, and artifact records

tau does not currently record a trusted full trajectory:

- `OpenRouterProxy` stores usage records, model ids, token counts, latency, cost, and errors, but not full request messages or full response content.
- `docker_solver.py` captures the whole harness stdout/stderr and parses a final JSON result, but command events are only whatever the miner logs claim.
- There is no canonical `rollout_id`.
- There is no source-of-truth trajectory file linking proxy LLM calls, Docker commands, observations, patch snapshots, judge labels, and duel outcomes.
- Public R2 explicitly deletes legacy public `training.jsonl`, which is the correct security stance.

## Polar Comparison

| Capability | tau/ninja today | Polar-style target | Gap |
| --- | --- | --- | --- |
| Black-box harness | Yes. tau calls miner `agent.py` through a stable solve contract. | Keep harness black-box. | Already good. |
| Managed LLM proxy | Yes. `OpenRouterProxy` enforces model/provider/sampling/budgets. | Proxy all model calls and capture token-faithful request/response. | Need full body capture and event ids. |
| Token-faithful prompts/responses | No. Usage summary only. | Store exact request messages, response content/tool fields, usage, provider route, and timestamps. | Biggest missing piece. |
| Turn boundaries | Partial. Agent logs imply steps, but untrusted. | Derive turn ids from proxy request order and harness event stream. | Need tau-side event stream. |
| Tool/command events | Partial/untrusted. `agent.py` logs commands; Docker captures stdout/stderr wholesale. | Runner-level command, cwd, exit code, stdout, stderr, duration, timeout, blocked status. | Need trusted runner instrumentation. |
| Observations | Partial/untrusted/truncated. | Store stdout/stderr observations with truncation metadata and content hashes. | Need source-of-truth capture. |
| Edit events | Only final diff plus optional miner patch. | Store patch snapshots or per-command diff deltas after commands that mutate repo. | Need runner diff snapshots. |
| Final patch | Yes. `solution.diff` / `SolveResult.solution_diff`. | Keep as rollout terminal artifact. | Already good. |
| Judge reward | Yes. LLM judge scores/rationale/winner in round result. | Attach scalar rewards and rationale/pairwise labels to rollout records. | Need rollout linkage and privacy tiering. |
| Pairwise winner/loser preference | Yes at duel round level. | Export DPO/pairwise rows linking winning and losing trajectories. | Need both trajectory ids in round metadata. |
| Async rollout execution | Yes for validation rounds via `ThreadPoolExecutor`. | Expose rollout-as-a-service queue/API for trainers/offline generation. | Later phase; not needed for minimal patch. |
| HF export | Task archive only. Public training export removed. | Private trajectory dataset plus optional public sanitized metadata. | Need new private HF exporter. |

## Concrete Rollout Schema

Use JSONL: one rollout row per agent attempt. Store large bodies inline for private/internal datasets; for local storage also allow gzip chunks and content-addressed blobs.

```json
{
  "schema_version": 1,
  "rollout_id": "rol_...",
  "duel_id": 123,
  "task_id": "validate-000123",
  "task_name": "validate-000123",
  "repo": "owner/repo",
  "commit_sha": "...",
  "issue": "...",
  "reference_patch": "...",
  "agent_hash": "sha256(agent.py)",
  "agent_source": {"kind": "github|private|local", "commit_sha": "..."},
  "hotkey": "...",
  "uid": 42,
  "role": "king|challenger|baseline|offline",
  "started_at": "...",
  "finished_at": "...",
  "runner": {
    "backend": "docker-file",
    "image": "swe-eval/file-solver:...",
    "timeout_seconds": 300,
    "container_network": "none"
  },
  "trajectory": [
    {
      "event_id": "evt_...",
      "type": "llm_call",
      "turn_index": 0,
      "started_at": "...",
      "finished_at": "...",
      "method": "POST",
      "path": "/v1/chat/completions",
      "request": {"model": "...", "messages": [], "max_tokens": 8192},
      "response": {"choices": [], "usage": {}},
      "response_text": "...",
      "usage": {"prompt_tokens": 123, "completion_tokens": 456, "total_tokens": 579},
      "model_requested": "...",
      "model_effective": "...",
      "provider": "...",
      "cost": 0.03,
      "latency_ms": 1234,
      "source": "tau_proxy"
    },
    {
      "event_id": "evt_...",
      "type": "command",
      "turn_index": 0,
      "started_at": "...",
      "finished_at": "...",
      "cmd": "pytest tests/test_x.py -q",
      "cwd": "/work/repo",
      "exit_code": 1,
      "stdout": "...",
      "stderr": "...",
      "duration_ms": 1200,
      "timed_out": false,
      "source": "tau_runner"
    },
    {
      "event_id": "evt_...",
      "type": "edit",
      "turn_index": 0,
      "path": "src/foo.py",
      "diff": "...",
      "repo_diff_sha256": "...",
      "source": "tau_runner"
    }
  ],
  "final_patch": "...",
  "miner_logs": "...",
  "steps": 12,
  "cost": 0.03,
  "success": true,
  "exit_reason": "completed",
  "judge": {
    "score": 8.1,
    "llm_score": 0.82,
    "winner": "challenger",
    "model": "anthropic/claude-sonnet-4.6",
    "rationale": "...",
    "error": null
  },
  "pairwise": {
    "duel_id": 123,
    "opponent_rollout_id": "rol_...",
    "won_vs_opponent": true,
    "winner_role": "challenger",
    "loser_role": "king"
  },
  "visibility": "private",
  "redactions": []
}
```

Also export derived training formats:

- GRPO row: prompt/messages plus trajectory, final answer/patch, scalar reward, group id.
- DPO row: task prompt plus chosen trajectory/patch and rejected trajectory/patch.
- SFT row: successful high-margin trajectories only, preferably filtered to public tasks or private internal use.

## What Should Stay in ninja

Keep only miner-facing agent behavior:

- `solve` contract
- prompt strategy
- command selection
- command parsing
- local safety checks
- patch extraction
- miner convenience logs

Do not add:

- rollout schemas
- replay/storage/export code
- HF credentials
- judge labels
- hidden task metadata
- trusted telemetry claims

ninja can optionally emit a miner-side `harness_events` list in its return JSON, but tau must treat it as untrusted convenience data. It should never be the canonical training/audit source.

## What Should Move or Be Added in tau

Add a new package:

```text
src/tau/rollouts/
  __init__.py
  schema.py
  ids.py
  store.py
  proxy_logger.py
  runner_events.py
  replay.py
  export_hf.py
  export_grpo.py
  export_dpo.py
  redaction.py
```

Functional split:

- `schema.py`: typed dict/dataclass definitions and pure validation helpers.
- `ids.py`: deterministic `rollout_id`, `event_id`, `duel_id/task_id/role` helpers.
- `store.py`: append-only local JSONL/gzip writer and manifest writer.
- `proxy_logger.py`: converts `OpenRouterProxy` requests/responses into `llm_call` events.
- `runner_events.py`: trusted command/edit event structures from Docker runner instrumentation.
- `redaction.py`: pure redaction policy for secrets, tokens, hidden task fields, and public/private tiers.
- `export_hf.py`: private HF JSONL upload similar to task archive, but for rollouts.
- `export_grpo.py`: derived RL rows.
- `export_dpo.py`: derived preference rows from king/challenger rounds.
- `replay.py`: reconstructs a rollout transcript and can rehydrate task workspace plus final patch for regression.

## Minimal Patch Plan

1. Add config and CLI flags.

- `src/config.py`
  - `record_rollouts: bool`
  - `rollout_root: Path | None`
  - `push_rollouts_to_hf: bool`
  - `rollout_hf_dataset: str | None`
  - `rollout_hf_token_env: str`
  - `rollout_export_format: str`
  - `rollout_visibility_default: str`
- `src/cli.py`
  - `tau validate --record-rollouts --rollout-root ... --push-rollouts-to-hf --rollout-hf-dataset ... --export-format jsonl`
  - same flags for `pool-manager` only if offline/pool rollouts are exported there.

2. Add the rollout package with pure schema/id/redaction/store helpers.

- Keep schema construction mostly pure: `build_rollout_record(inputs...) -> dict`.
- Keep mutation in `store.append_rollout(record, root)` and HF upload functions.

3. Extend `OpenRouterProxy` with an optional event sink.

- Add constructor fields like `event_sink: Callable[[dict], None] | None` and `capture_bodies: bool`.
- In `_handle_request`, capture sanitized prepared request payload and response payload/body.
- Keep existing `SolveUsageSummary` behavior unchanged.
- Add `event_id`, `turn_index`, timestamps, latency, model_requested/effective, usage, cost.

This is the source of truth for LLM calls.

4. Instrument Docker runner at tau level.

The cleanest minimal implementation is to add a tau-owned wrapper script in `_harness_runner_script()` that monkey-patches the loaded agent module:

- wrap `chat_completion` if present only as a convenience, but do not trust it over proxy events
- wrap `run_command` if present to emit command events with stdout/stderr/exit code
- after each command, run `git diff --binary` and emit an `edit` event when the diff hash changes
- write events to a mounted file such as `/work/tau_events.jsonl`
- collect that file in `docker_solver.py` after `_run_solver_command`

This is not perfect for arbitrary future agents, but it covers the current ninja contract without bloating ninja. Longer term, move command execution behind a tau-owned tool API so every command is necessarily runner-mediated.

5. Link rollout ids into solve/round results.

- `src/solver_runner.py`: add `rollout_id`, `rollout_path`, maybe `trusted_events_path` to `SolveResult`.
- `src/docker_solver.py`: create one rollout context per solve, pass event sink to proxy, collect runner events, write a local rollout record.
- `src/pipeline.py`: write rollout metadata into `solve.json`.
- `src/validate.py`: add `king_rollout_id` and `challenger_rollout_id` to `ValidationRoundResult`, plus pairwise linkage after `_judge_round_diffs`.

6. Add private HF exporter.

- Mirror `task_pool_manager.append_hf_dataset_jsonl`, but use separate rollout dataset config.
- Do not use public R2.
- Add hourly sharding: `rollouts/YYYY-MM-DD-HH/{duel_id}.jsonl.gz`.
- Add derived exporters:
  - `grpo/YYYY-MM-DD-HH.jsonl`
  - `dpo/YYYY-MM-DD-HH.jsonl`

7. Tests.

- `tests/test_rollout_schema.py`: schema validation and deterministic ids.
- `tests/test_openrouter_proxy_rollout_logging.py`: proxy captures request/response bodies when enabled and redacts auth.
- `tests/test_docker_solver_rollout_events.py`: fake/minimal agent emits command/edit events.
- `tests/test_rollout_public_redaction.py`: public export excludes hidden tasks, reference patches, judge rationale if configured private.
- `tests/test_rollout_dpo_export.py`: pairwise row chooses challenger/king correctly from round winner.

## Security Concerns

Be strict here. Full trajectories are dangerous if published blindly.

- Hidden promotion tasks: publishing `issue`, `reference_patch`, command observations, or final patches lets miners hill-climb against the validator distribution.
- Judge rationales: public rationales reveal scorer preferences and failure modes. They are useful for training but are also a reward-hacking guide.
- Token-faithful prompts: prompts can contain hidden task text, repo paths, inferred reference behavior, proxy routing, and accidental secrets from command output.
- Command stdout/stderr: tests often print env vars, paths, tokens, dependency URLs, private filenames, or hidden task clues.
- Miner logs are adversarial: include them only as `miner_logs_untrusted`; never use them as the trusted event stream.
- Agent code may attempt telemetry poisoning: runner/proxy events need `source=tau_proxy` or `source=tau_runner`; miner-provided events need `source=miner_untrusted`.
- Replay can become exfiltration: replay tools must default to private local paths and never replay hidden tasks into public CI.
- Pairwise data can leak exact promotion tasks and reward boundaries. Keep public datasets delayed, sampled, or synthetic unless the subnet explicitly accepts hill-climbing risk.

## Public vs Private Data

Public HF/R2 should contain only:

- duel ids and timestamps
- public participant metadata already exposed on dashboard
- sanitized final patches for king/challenger if the current R2 policy accepts that
- sanitized solve summaries: success, exit_reason, token/cost totals, request counts
- aggregate scores/winner labels
- maybe delayed public task ids after tasks are retired and no longer influence promotion

Private/internal HF should contain:

- full `issue`
- `reference_patch`
- token-faithful LLM request/response bodies
- command stdout/stderr observations
- edit deltas
- final patches
- judge scores/rationales
- pairwise chosen/rejected trajectory ids
- replay metadata

Do not publish full rollout trajectories for active hidden validation tasks. If public training data is a goal, create a separate public/offline task stream where leakage is harmless, or publish with a long delay after those tasks are permanently retired.

## Blunt Recommendation

Do the first implementation in tau only.

The minimal high-value patch is not a new agent abstraction. It is:

1. turn `OpenRouterProxy` into a body-capturing rollout event source,
2. collect runner-level command/edit events from Docker,
3. write private local rollout JSONL with stable ids,
4. attach rollout ids to `solve.json` and duel round results,
5. export private GRPO/DPO JSONL from those records.

That converts tau from "task -> final patch -> score" into "prompt -> response -> command -> observation -> edit -> final patch -> reward" without giving miners a larger trusted surface to game.
