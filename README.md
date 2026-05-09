# tau

`tau` is a small CLI for running a staged SWE workflow:

1. `generate` mines a commit and creates a task.
2. `solve` runs a solver against that task.
3. `compare` scores two saved solutions by changed-line similarity.
4. `eval` compares multiple solutions with an LLM judge.
5. `delete` removes saved task artifacts.

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

### Miner PR rules (blocked by CI)

In production, miners submit through PRs to `unarbos/ninja`.
The PR workflow blocks, and/or fails, PRs that do:

- modify files outside `agent.py` in `ninja`
- change the `solve(...)` contract
- touch validation/CI files (for example workflow files) or add non-miner infra
  files
- hardcode or import external model/provider credentials
- override provider routing (`api_base`, `api_key`, or `model`)
- set sampling/decoding params (`temperature`, `top_p`, `top_k`, `seed`,
  penalties, `logprobs`, etc.)
- include PR titles that do not start with the exact miner hotkey (with no `hkey:` prefix)
- add direct network/provider calls intended to bypass the validator-managed proxy

Only PRs with only allowed edits to `agent.py` and compliant metadata are judged in
`ninja` CI.

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

Production miner submissions should use PR commitments to `ninja`, not raw
`owner/repo@sha` commitments.

## Prerequisites

- Python 3.11+
- `uv`
- Docker
- A GitHub token for task generation
- An OpenRouter API key for Docker file solves and evaluation

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

## Validator GitHub PR Mode

The live validator can score miner edits from the public `unarbos/ninja` harness repo. In this mode, miners do not commit arbitrary GitHub repos directly. They open a PR against `unarbos/ninja`, then commit the PR head on-chain.

Miner commitment format:

```text
github-pr:unarbos/ninja#<pr-number>@<head-sha>
```

Miners can also protect a submission before the PR is public by committing the
final local Git head first:

```text
github-pr-head:unarbos/ninja@<head-sha>
```

With this format, the validator waits until it can find an open PR whose title
starts with the committing hotkey, whose base is `unarbos/ninja:main`, and whose
current head matches the precommitted SHA.

The PR title must start with the exact committing miner hotkey:

```text
<miner-hotkey> improve harness loop
```

The validator only queues the PR when all of these match:

- the commitment comes from a registered subnet hotkey
- the hotkey has not committed since the later of the configured hotkey-spent cutoff or its current registration block
- the watched repo is `unarbos/ninja` and the base branch is `main`
- the PR is open and not draft
- the PR title starts with the committing hotkey
- the committed SHA matches the current PR head SHA
- required GitHub checks are green: `PR Scope Guard` and `OpenRouter PR Judge`
- the PR head commit is publicly fetchable

A miner can resubmit from the same hotkey only after it is freshly registered
again. By default, any prior on-chain commitment at or after block `8,104,340`
spends the current registration period; older commitments, including commitments
before the hotkey's current registration block, do not.

Miner-side preflight for the pre-PR flow:

```bash
python3 scripts/precommit_ninja_pr.py \
  --repo ../ninja \
  --hotkey <miner-hotkey> \
  --judge
```

The script refuses dirty worktrees by default, prints the exact
`github-pr-head:...` commitment for `HEAD`, runs local static CI-style checks,
and with `--judge` calls the same OpenRouter judge prompt from the trusted base
branch. Add `--commit-on-chain` after the preflight passes to submit the
commitment before pushing/opening the PR.

### Validator-side guardrails

- PRs are checked against required CI checks:
  - `PR Scope Guard`
  - `OpenRouter PR Judge`
- `PR Scope Guard` rejects all edits outside `agent.py` and edits that break the
  solve contract or attempt forbidden provider/sampling control.
- `OpenRouter PR Judge` reviews the diff with `openai/gpt-5.4` through
  OpenRouter and requires a score above `JUDGE_MIN_SCORE`.

GitHub PR mode uses 50 duel rounds minimum. If a run is configured lower, the
validator bumps it to 50 and raises the task pool target to match.

The validator keeps two independent 50-task pools: a primary pool for the
first challenger-vs-king duel, and a retest pool used only when the challenger
wins the primary duel. Promotion requires the challenger to also win the retest,
which checks the improvement on a separate task set before changing the king.
Parallel duels run the gathered task set instead of stopping early once an
outcome is mathematically decided. Both pools receive the configured refresh
batch, 5 tasks per hour in production.

The production validator continuously drains queued candidates in queue order
and refreshes on-chain submissions every 10 minutes, adding newly eligible PRs
to the queue. Each duel can run up to 25 round workers with challenger agent
timeouts capped at 600 seconds. If a challenger hits 5 consecutive round
timeouts, the validator stops submitting new rounds for that challenger and
moves on after its already-running rounds finish.

When a PR challenger becomes king, the validator auto-merges that PR into the
watched `unarbos/ninja` base branch, records the king as the resulting base
repo commit while keeping the miner hotkey/PR metadata, flushes the old task
pool, and assigns all validator weight to the winning hotkey on the next
allowed weight-set epoch.

The background pool filler prepares tasks before challengers arrive. It caps
king pool solves at 300 seconds, stores the king patch, and the duel gatherer
chooses from unused cached tasks first.
Once the pool is full, the production validator refreshes it by adding 5 new
valid tasks every hour; the normal prune step then removes the oldest 5 so the
pool stays at the configured target size.

`start_validator.sh` enables this production path with:

```bash
--round-concurrency 25 \
--candidate-timeout-streak-limit 5 \
--poll-interval-seconds 600 \
--watch-github-prs \
--github-pr-only \
--github-pr-repo unarbos/ninja \
--github-pr-base main
```

Use `--hotkey-spent-since-block` or `VALIDATE_HOTKEY_SPENT_SINCE_BLOCK` to
override the spent-history cutoff block.

`--github-pr-only` means normal `unarbos/ninja@sha` commitments are ignored by the live validator. This keeps miner submissions tied to PR review, CI, and the committing hotkey.

Optional PR cleanup can label and close old or invalid open PRs in the watched
repo:

```bash
--github-pr-cleanup \
--github-pr-cleanup-stale-after-hours 24 \
--github-pr-missing-commitment-notice-after-minutes 30
```

The cleanup pass uses GitHub labels as sortable close reasons:

- `close: failed-test` for failed required validator CI
- `close: passed-test-inadequate` for rejected judge checks
- `close: stale-head` when the PR head moved away from the on-chain commitment
- `close: stale-base` for PRs targeting an unwatched base
- `close: hotkey-spent` when the title hotkey already used its one submission
- `close: stale-submission` when an old PR is not live in the validator queue
- `close: promoted-king` when the validator already promoted that PR
- `notice: missing-commitment` for open PRs older than the notice window where
  the title hotkey has not posted a matching on-chain PR commitment

Set `VALIDATE_GITHUB_PR_CLEANUP=1` to enable the same behavior from the
environment. The cleanup uses the owner-scoped GitHub merge token because it
needs write access to add labels, comment, and close PRs.

## Validator Duel Scoring

Each validation task still starts from a mined GitHub commit: `task/original` is the repo before the commit, `task/reference` is the repo after it, and `task/reference.patch` is used to filter out tiny tasks.

For duels, the live validator does not run a baseline solution. The pool filler prepares the task and the current king patch, then each duel round solves the challenger patch on the same task.

Round score is 100% dual LLM diff judgment. The default judges are `openai/gpt-5.4` and `anthropic/claude-sonnet-4.6` through OpenRouter at temperature 0 with medium reasoning effort and a 16000-token output cap. Set `DIFF_JUDGE_MODELS=model-a,model-b` to override them. The validator randomly maps king/challenger to `candidate_a`/`candidate_b` for each judged round before prompting. If the judges disagree, they exchange only public candidate-labeled deliberation notes and retry for up to three rounds; their final JSON decisions remain hidden from each other during deliberation and are omitted from public dashboard/R2 payloads.

The challenger needs more decisive round wins than the current king. By default, `--win-margin 0` means one more challenger win than king win is enough; the deployed live validator currently passes `--win-margin 3`, so the challenger must win by at least four decisive rounds.

The validator still compares `king` to `challenger` separately for copy detection, but that pairwise similarity does not contribute to round score.

## Managed Inference Policy

Docker file agents receive a validator-managed OpenAI-compatible endpoint through `solve(..., model, api_base, api_key)`. The upstream provider key is never passed into miner code.

The proxy forwards to OpenRouter and enforces:

- the validator-selected model, currently `deepseek/deepseek-v4-flash` for solver inference unless overridden by validator config
- `temperature=0.0`
- `top_p=1.0`
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

- `claude` to run the local Claude CLI on the host
- a local `agent.py` file for the Docker file solver
- a local repo root containing `agent.py` for the Docker file solver
- a GitHub repo URL or shorthand like `owner/repo` for the Docker file solver

Example using Claude:

```bash
source .venv/bin/activate
tau solve --task my-task --solution claude-run --agent claude
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
- `--solver-max-requests <int>`
- `--solver-max-total-tokens <int>`
- `--solver-max-cost <float>`
- `--docker-solver-memory 2g`
- `--docker-solver-cpus 2`
- `--docker-solver-no-cache`
- `--agent-timeout <seconds>`
- `--debug`

## Compare Solutions

Compare two saved solutions using changed-lines-only similarity:

```bash
source .venv/bin/activate
tau compare --task my-task --solutions claude-run baseline
```

Comma-separated values also work:

```bash
source .venv/bin/activate
tau compare --task my-task --solutions claude-run,baseline
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
tau solve --task demo-task --solution run-1 --agent claude
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

## Notes

- `generate` needs `GITHUB_TOKEN` or `GH_TOKEN`.
- `tau solve --agent claude` needs the `claude` CLI installed on the host.
- Docker file solves and `eval` need `OPENROUTER_API_KEY`.
- `compare` reads saved solution artifacts and does not call a model.
- Docker-backed solves use Docker, so Docker must be installed and running.
- Generated task, solution, and evaluation paths are printed by the CLI after each command finishes.
