# tau

`tau` is a small CLI for running a staged SWE workflow:

1. `generate` mines a commit and creates a task.
2. `solve` runs a solver against that task.
3. `compare` scores two saved solutions by changed-line similarity.
4. `eval` compares multiple solutions with an LLM judge.
5. `delete` removes saved task artifacts.
6. `private-submit` validates and stores a signed private miner submission.
7. `serve-submissions-api` accepts private miner submissions over HTTP.
8. `validate` runs the live king-of-the-hill validator loop.
9. `restore-r2-kings` republishes the validator dashboard's recent king window.
10. `benchmarks` plans or runs external benchmark harnesses for SWE-rebench,
    DeepSWE, and Terminal-Bench.
11. `swebench-king-benchmark` watches crowned kings and runs SWE-bench Verified
    against the pi baseline.

## External Benchmarks

Use `tau benchmarks` to keep a repeatable run plan for the three external
agent evals we care about after a challenger becomes king:

```bash
tau benchmarks --agent king --model openai/gpt-5.2 --n-tasks 25 --sample-seed 66
```

By default this writes `workspace/benchmarks/benchmark-plan.json` without
executing external tools. Add `--run` once the harness CLIs are installed and
their agent selectors are wired:

```bash
tau benchmarks --agent king --model openai/gpt-5.2 --n-tasks 25 --sample-seed 66 --run
```

The configured suite is:

- SWE-rebench, via the `nebius/SWE-rebench-leaderboard` dataset
- DeepSWE, via the `datacurve-ai/deep-swe` task suite
- Terminal-Bench, via `terminal-bench-core==head`

For a sub-five-minute external smoke, use Runloop's public SWE-bench Verified
and Terminal-Bench jobs with both agents planned side by side:

```bash
tau benchmarks \
  --provider runloop \
  --preset smoke \
  --agent king \
  --baseline pi \
  --scenario <swe-or-terminal-scenario-id>
```

This writes a plan for `swe-bench-verified` and `terminal-bench-2` using
Runloop's supported multi-agent job form. Pass explicit scenario ids for
sub-five-minute smoke runs; otherwise Runloop runs the full benchmark. Add
`--run` after `rli` is installed and `RUNLOOP_API_KEY` is configured.

## Crown SWE-Bench Benchmark

Run the crowned-king SWE-bench daemon under PM2 with:

```bash
pm2 start ./start_swebench_king_benchmark.sh --name swebench-king-benchmark
```

The daemon watches `workspace/validate/netuid-66/state.json`. When
`current_king.commit_sha` changes, it runs the fixed
`data/swebench_verified_sample_50_seed66.json` sample for both the crowned king
and `https://github.com/earendil-works/pi`, using `minimax/minimax-m2.7` and
provider `minimax/fp8`. It writes predictions, solve results, proxy usage,
official SWE-bench scoring output, and `comparison.json` under:

```text
workspace/validate/netuid-66/benchmarks/swebench-verified/<king_commit>/
```

The daemon also merges a compact summary into local dashboard JSON and R2 under
`benchmarks.swebench_verified`.

## Miner Harness

The canonical miner-editable harness is a single file in the public
[`unarbos/ninja`](https://github.com/unarbos/ninja) repository.
`tau` owns task generation, Docker execution, validation, scoring, and managed
inference; `ninja` is only the base agent for miners to edit.

### What belongs in `ninja`

- `agent.py` (plus comments and docs for miners)
- no task generators, validator code, pm2 configs, wallets, task pool tooling, or
  R2 helpers

For local tests you can run either the published ninja repo or a local clone:

```bash
source .venv/bin/activate
tau solve --task my-task --solution ninja-main --agent unarbos/ninja
tau solve --task my-task --solution local-ninja --agent ../ninja
```

`agent.py` must define:

```python
def solve(repo_path: str, issue: str, model: str, api_base: str, api_key: str) -> dict:
    ...
```

and should return `patch`, `logs`, `steps`, `cost`, and `success`.
`model`, `api_base`, and `api_key` are always provided by the validator and must
be treated as read-only invocation parameters.

### Multi-file agents

A submission may also be a set of Python files with `agent.py` as the
entrypoint (for example `agent.py` plus a support package). All files must be
relative `*.py` paths without traversal, are subject to the same scope-guard
checks as `agent.py`, and may import each other (the harness puts the agent
directory on `sys.path`). Submission routes:

- API: pass the extra modules in a `files` form field containing a JSON object
  of `{relative_path: content}` alongside the usual `agent` field.
- CLI: `tau private-submit --agent <directory>` collects every `*.py` under the
  directory.
- GitHub commitments: commit a `tau_agent_files.json` manifest (a JSON array of
  relative paths including `agent.py`) at the pinned commit; repos without the
  manifest keep the legacy single-file `agent.py` extraction.

Single-file submissions keep their historical sha256-of-`agent.py` hash;
multi-file submissions hash the whole file set, and the same hash is used in
the commitment and signature payloads below. The public base harness repo
([unarbos/ninja](https://github.com/unarbos/ninja)) is a working multi-file
example.

### Private miner submission rules

In production, miners do not submit code through public GitHub PRs. They submit
their `agent.py` privately to the validator API, and the validator stores a
private bundle under `private-submissions/<submission-id>/`. The validator tracks
the private bundle id and file hash internally:

```text
private-submission:<submission-id>:<sha256-of-agent.py>
```

The private submission route blocks submissions that do:

- change the `solve(...)` contract
- hardcode or import external model/provider credentials
- override provider routing (`api_base`, `api_key`, or `model`)
- set sampling/decoding params (`temperature`, `top_p`, `top_k`, `seed`,
  penalties, `logprobs`, etc.)
- add direct network/provider calls intended to bypass the validator-managed proxy
- fail Python compile or pyflakes smoke checks
- fail the OpenRouter private submission judge

The miner must sign this payload with the submitting hotkey:

```text
tau-private-submission-v1:<hotkey>:<submission-id>:<sha256-of-agent.py>
```

The validator verifies that signature before queueing the private bundle, so a
different miner cannot copy someone else's private code.

Submissions can also include an optional agent username proof. Sign this
message with the coldkey that owns the submitting hotkey:

```text
tau-agent-submission-username:<username>
```

Then include `agent_username`, `coldkey`, and `coldkey_signature` in the private
submission API request, or pass `--agent-username`, `--coldkey`, and
`--coldkey-signature` to `tau private-submit`. The validator stores the username
only when the coldkey currently owns the submitting hotkey and the signature
verifies; invalid or incomplete username proofs are ignored without blocking the
submission.

You can still test a local agent from any GitHub repo for research, e.g.:

```bash
source .venv/bin/activate
tau solve --task my-task --solution shared --agent owner/repo
```

or:

```bash
source .venv/bin/activate
tau solve --task my-task --solution shared --agent https://github.com/owner/repo
```

Production miner submissions should use the private submission API, not GitHub
PRs or raw `owner/repo@sha` commitments.

## Prerequisites

- Python 3.11+
- `uv`
- Docker
- A GitHub token for task generation
- An OpenRouter API key for Docker file solves and evaluation
- A Cursor API key for Cursor solves

## Setup

From the `tau/` directory:

```bash
source .venv/bin/activate
uv pip install -e .
```

Create a `.env` file in `tau/` if you do not already have one:

```bash
GITHUB_TOKEN=your_github_token
OPENROUTER_API_KEY=your_openrouter_api_key
CURSOR_API_KEY=your_cursor_api_key
```

`tau` loads `.env` automatically from the project root.

Optional environment defaults for centralized solver routing:

```bash
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
SOLVER_MAX_REQUESTS=40
SOLVER_MAX_TOTAL_TOKENS=200000
SOLVER_MAX_PROMPT_TOKENS=160000
SOLVER_MAX_COMPLETION_TOKENS=40000
SOLVER_MAX_TOKENS_PER_REQUEST=4096
SOLVER_MAX_COST=1.00
```

CLI flags still override these values for one-off runs.

## Validator Private Submission Mode

The live validator scores private miner edits from local bundle storage. Miners
send `agent.py`, hotkey, submission id, and hotkey signature over a private
operator-controlled channel. The operator stores the bundle with:

```bash
tau private-submit \
  --hotkey <miner-hotkey> \
  --agent /path/to/submitted-agent.py \
  --base-agent /path/to/current-public-agent.py \
  --signature <hotkey-signature> \
  --private-submission-root /secure/private-submissions \
  --network finney
```

The command prints JSON with the private submission id/hash, the signature
payload, `ci_checks`, and the raw `llm_judge` result. If the
operator already knows the current registration block, `--registration-block`
can be supplied instead of doing the chain lookup.

To serve the miner-facing private submission API behind `ninja66.ai`, run:

```bash
tau serve-submissions-api \
  --host 127.0.0.1 \
  --port 8066 \
  --base-agent /path/to/current-public-agent.py \
  --private-submission-root /secure/private-submissions \
  --max-request-bytes 5000000 \
  --max-agent-bytes 5000000 \
  --rate-limit-max-requests 6 \
  --rate-limit-max-failures 3 \
  --network finney
```

The HTTP API accepts `POST /api/submissions` as multipart form data with
`agent`, `hotkey`, `submission_id` (optional), `signature`, and optional
`agent_username`/`coldkey`/`coldkey_signature`. It returns the
same acceptance JSON as `private-submit`, including `ci_checks` and `llm_judge`
on failures before returning a non-2xx status. Accepted submissions refresh the
public accepted-submissions payload at `sn66/api/submissions`, which is exposed
as `https://ninja66.ai/api/submissions` by the same R2/domain mapping used for
the dashboard. The validator queues accepted API submissions directly from the
private ledger after rechecking that the hotkey is still registered.

Run this API behind nginx, Cloudflare, or an equivalent edge proxy. The Python
server rejects oversized submissions, limits concurrent expensive checks, and
rate-limits each client IP, but network-layer floods should be absorbed before
they reach the validator host.

`private-submit` accepts and stores at most one valid bundle for a hotkey's
current registration block. It records accepted submissions in
`_accepted_submissions.json` under the private submission root; a second valid
bundle from the same hotkey is rejected until the hotkey re-registers and the
registration block advances. The validator also re-checks registration status
before queueing an accepted API submission.

The validator only queues the private submission when all of these match:

- the submission comes from a registered subnet hotkey
- the hotkey has not already used an accepted submission in its current registration
- the private submission gate has accepted no other bundle from this hotkey in its current registration
- the private bundle exists under the configured private submission root
- `agent.py` hashes to the committed SHA256
- the bundle hotkey matches the submitting hotkey
- the hotkey signature verifies for the submitted payload
- local checks are green: `Agent Smoke`, `Submission Scope Guard`, and `OpenRouter Submission Judge`

A miner can resubmit from the same hotkey only after it is freshly registered
again. Accepted API submissions are treated as spent for the hotkey's current
registration period; submissions from an older registration are ignored after
the hotkey re-registers.

### Validator-side guardrails

- Private bundles are checked against validator-side API gates:
  - `Agent Smoke`
  - `Submission Scope Guard`
  - `OpenRouter Submission Judge`
- `Agent Smoke` compiles `agent.py` and runs pyflakes.
- `Submission Scope Guard` rejects edits that break the
  solve contract or attempt forbidden provider/sampling control.
- `OpenRouter Submission Judge` reviews the diff with the private submission
  gatekeeping prompt through OpenRouter using `z-ai/glm-5.2`
  at temperature 0 with a 16000-token output cap and required overall/component
  score floors. Override with `--judge-model` or
  `PRIVATE_SUBMISSION_JUDGE_MODEL`; Anthropic models additionally get medium
  reasoning effort. Reorder-only or gate-order-only submissions must show
  concrete evidence that they improve broad task behavior, not just that they
  change control flow.

The validator keeps two independent 50-task pools: a primary pool for the
first challenger-vs-king duel, and a retest pool used only when the challenger
wins the primary duel. Promotion requires the challenger to also win the retest,
which checks the improvement on a separate task set before changing the king.
Parallel duels run the full gathered task set instead of stopping early once an
outcome is mathematically decided. By default both pools are static fixed-size
sets: once each pool reaches 50 tasks, the validator reuses that same ordered
set until the king changes or an operator explicitly enables pool refresh.

The production validator continuously drains queued candidates in queue order
and refreshes accepted API submissions every 10 minutes, adding newly eligible
private submissions to the queue. Each duel can run up to 25 round workers with challenger agent
timeouts capped at 600 seconds. If a challenger hits 5 consecutive round
timeouts, the validator stops submitting new rounds for that challenger and
moves on after its already-running rounds finish.

When a private challenger becomes king, the validator publishes the winning
`agent.py` directly to the configured public base repo, records the king as the resulting base
repo commit while keeping the miner hotkey metadata, flushes the old task
pool, and assigns all validator weight to the winning hotkey on the next
allowed weight-set epoch.

The background pool filler pre-solves tasks before challengers arrive. It caps
Cursor and king pool solves at 300 seconds, skips timed-out or empty Cursor
baselines, and the duel gatherer preserves the cached task order so every
challenger sees the same sequence.
With the default settings, once the primary and retest pools are full they stay
static at 50 tasks each. Scheduled recycling is disabled unless
`--task-pool-refresh-count` and `--task-pool-refresh-interval-seconds` are set
to non-zero values.

`start_validator.sh` routes both judges through OpenRouter with
`TAU_DIFF_JUDGE_MODEL=z-ai/glm-5.2` and
`PRIVATE_SUBMISSION_JUDGE_MODEL=z-ai/glm-5.2`. Solver, generator, and eval
traffic uses the self-hosted `Qwen/Qwen3-32B` endpoint. The script runs
`cli validate` with notable flags such as:

```bash
python -m cli validate \
  --solver-model Qwen/Qwen3-32B \
  --round-concurrency 25 \
  --docker-solver-start-concurrency 25 \
  --candidate-timeout-streak-limit 10 \
  --poll-interval-seconds 600 \
  --task-pool-target 50 \
  --task-pool-static \
  --duel-rounds 50 \
  --win-margin 6 \
  --hotkey-spent-since-block 8104340 \
  --watch-private-submissions \
  --private-submission-only \
  --publish-repo unarbos/ninja \
  --publish-base main
```

`--private-submission-only` means normal `unarbos/ninja@sha` submissions are
ignored by the live validator. This keeps miner submissions private until a
challenger becomes king.

## Validator Duel Scoring

Each validation task still starts from a mined GitHub commit: `task/original` is the repo before the commit, `task/reference` is the repo after it, and `task/reference.patch` is used to filter out tiny tasks.

For duels, the score comes solely from the LLM diff judge. The pool filler still creates a Cursor baseline solution at `solutions/baseline` so the validator can keep compatibility telemetry, copy checks, and timeout calibration data, but Cursor-baseline similarity no longer contributes to the winner.

Round score is based only on the LLM diff judgment. The diff judge uses
`z-ai/glm-5.2` through OpenRouter at temperature 0 with a
16000-token output cap, then scores the king and challenger patches against
the task/reference context. Candidate patches are role-blinded and treated as
untrusted input. Up to four attempts run within a 300-second total timeout.
Change the model with `TAU_DIFF_JUDGE_MODEL`; Anthropic models use adaptive
reasoning and prompt-cache breakpoints, but production currently runs GLM 5.2
with no configured fallback models.

Cursor is telemetry only for round scoring. The challenger does not need to beat Cursor directly; it only needs more decisive round wins than the current king plus the configured margin. `start_validator.sh` currently uses `--win-margin 6`.

The validator still compares `king` to `challenger` separately for copy detection, but that pairwise similarity does not affect the round score.

## Managed Inference Policy

Docker file agents receive a validator-managed OpenAI-compatible endpoint through `solve(..., model, api_base, api_key)`. The upstream provider key is never passed into miner code.

The proxy forwards to OpenRouter and enforces:

- the validator-selected model, currently self-hosted `Qwen/Qwen3-32B` for solver inference unless overridden by validator config
- `temperature=0.0`
- `top_p=0.01` (override via `TAU_TOP_P`)
- removal of miner-controlled sampling fields such as `top_k`, `seed`, penalties, `logit_bias`, and `logprobs`
- request, token, and cost budgets

Miner agents should use only the supplied `api_base` and `api_key`. Attempts to choose another provider, model, sampling policy, or credential path are rejected by `ninja` CI and overwritten or stripped by the validator proxy.

## Basic Usage

Show top-level help:

```bash
source .venv/bin/activate
tau --help
```

All commands write their artifacts under:

```text
workspace/tasks/
```

You can override that with `--workspace-root /path/to/root`.

## Generate A Task

```bash
source .venv/bin/activate
tau generate --task my-task
```

Useful options:

- `--generator-model <model>`
- `--seed <int>`
- `--max-mining-attempts <int>`
- `--agent-timeout <seconds>`
- `--debug`

## Solve A Task

`solve` supports multiple backends. The `--agent` value can be:

- `cursor` to run the Cursor CLI in Docker
- `claude` to run the local Claude CLI on the host
- `claw` to run the local Claw CLI on the host
- a local `agent.py` file for the Docker file solver
- a local repo root containing `agent.py` for the Docker file solver
- a GitHub repo URL or shorthand like `owner/repo` for the Docker file solver

Example using Cursor:

```bash
source .venv/bin/activate
tau solve --task my-task --solution cursor-run --agent cursor
```

Example using Claude:

```bash
source .venv/bin/activate
tau solve --task my-task --solution claude-run --agent claude
```

Example using Claw:

```bash
source .venv/bin/activate
tau solve --task my-task --solution claw-run --agent claw
```

Example using the public `ninja` harness:

```bash
source .venv/bin/activate
tau solve --task my-task --solution baseline --agent unarbos/ninja
```

Example using a local checkout of `ninja`:

```bash
source .venv/bin/activate
tau solve --task my-task --solution baseline --agent ../ninja
```

Useful options:

- `--solver-model <model>`
- `--baseline-model <model>`
- `--solver-max-requests <int>`
- `--solver-max-total-tokens <int>`
- `--solver-max-prompt-tokens <int>`
- `--solver-max-completion-tokens <int>`
- `--solver-max-tokens-per-request <int>`
- `--solver-max-cost <float>`
- `--solver-provider-sort price|throughput|latency`
- `--solver-provider-only <provider[,provider...]>`
- `--solver-provider-disable-fallbacks`
- `--solver-provider-min-throughput-p50 <float>`
- `--solver-provider-min-throughput-p90 <float>`
- `--docker-solver-memory 2g`
- `--docker-solver-cpus 2`
- `--docker-solver-no-cache`
- `--agent-timeout <seconds>`
- `--debug`

## Compare Solutions

Compare two saved solutions using changed-lines-only similarity:

```bash
source .venv/bin/activate
tau compare --task my-task --solutions cursor-run baseline
```

Comma-separated values also work:

```bash
source .venv/bin/activate
tau compare --task my-task --solutions cursor-run,baseline
```

## Evaluate Solutions

Compare two or more solutions for the same task:

```bash
source .venv/bin/activate
tau eval --task my-task --solutions baseline candidate-a candidate-b
```

Comma-separated values also work:

```bash
source .venv/bin/activate
tau eval --task my-task --solutions baseline,candidate-a,candidate-b
```

Useful options:

- `--eval-model <model>`
- `--seed <int>`
- `--agent-timeout <seconds>`
- `--debug`

## Delete Saved Artifacts

Delete one task:

```bash
source .venv/bin/activate
tau delete --task my-task
```

Delete all saved tasks:

```bash
source .venv/bin/activate
tau delete task --all
```

## End-To-End Example

```bash
source .venv/bin/activate
tau generate --task demo-task
tau solve --task demo-task --solution run-1 --agent cursor
tau solve --task demo-task --solution run-2 --agent unarbos/ninja
tau compare --task demo-task --solutions run-1 run-2
tau eval --task demo-task --solutions run-1 run-2
```

## Single-File Agent In Docker

When you pass a local file, local repo directory, or GitHub repo to `--agent`, tau builds a small Python Docker image, imports `agent.py`, and calls its `solve(...)` function.

### What happens

1. A Docker image (`swe-eval/file-solver:<hash>`) is built from `python:3.11-slim`.
2. A container starts with resource limits (memory, CPU, pids, tmpfs).
3. The task repo is copied into the container at `/work/repo`.
4. The submitted `agent.py` is copied into the container and imported.
5. The validator calls `solve(repo_path="/work/repo", issue=..., model=..., api_base=..., api_key=...)` with the managed model id, local proxy URL, and per-run proxy token.
6. The diff is collected from the container and applied back to the host repo.
7. The container is torn down.

The submitted agent does not receive the upstream OpenRouter key. On Linux the solver container runs with Docker network disabled and reaches the validator proxy through a local socket bridge, so LLM calls flow through one managed endpoint.

## Cursor Agent In Docker

When you pass `--agent cursor`, tau builds a Docker image, runs the Cursor CLI inside it, and collects the resulting diff.

### What happens

1. A Docker image (`swe-eval/cursor-solver:<hash>`) is built from `python:3.11-slim` with the Cursor CLI installed via `curl https://cursor.com/install | bash`.
2. A container starts with resource limits (memory, CPU, pids, tmpfs).
3. The task repo is copied into the container at `/work/repo` and the prompt is written to `/work/task.txt`.
4. The Cursor `agent` CLI runs inside the container with `CURSOR_API_KEY` injected:

```bash
agent -p --force --trust --sandbox disabled --output-format stream-json \
    --workspace /work/repo "$PROMPT"
```

5. The diff is collected from the container and applied back to the host repo.
6. The container is torn down.

### Usage

```bash
source .venv/bin/activate
tau solve --task my-task --solution cursor-run --agent cursor
```

`CURSOR_API_KEY` must be set in your environment or in `tau/.env`.

### Docker options

| Flag | Purpose |
|------|---------|
| `--solver-model <model>` | Override the model used by Cursor |
| `--agent-timeout <seconds>` | Time limit for the solve |
| `--docker-solver-memory 2g` | Container memory limit |
| `--docker-solver-cpus 2` | Container CPU limit |
| `--docker-solver-no-cache` | Force rebuild the Docker image |
| `--debug` | Enable debug logging |

## Notes

- `generate` needs `GITHUB_TOKEN` or `GH_TOKEN`.
- `tau solve --agent cursor` needs `CURSOR_API_KEY` and Docker.
- `tau solve --agent claude` needs the `claude` CLI installed on the host.
- `tau solve --agent claw` needs the `claw` CLI installed on the host.
- Docker file solves and `eval` need `OPENROUTER_API_KEY`.
- `compare` reads saved solution artifacts and does not call a model.
- Docker-backed solves use Docker, so Docker must be installed and running.
- Generated task, solution, and evaluation paths are printed by the CLI after each command finishes.
