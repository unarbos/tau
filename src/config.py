from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_str(*names: str) -> str | None:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _env_int(*names: str) -> int | None:
    value = _env_str(*names)
    if value is None:
        return None
    return int(value)


def _env_int_default(name: str, default: int) -> int:
    value = _env_int(name)
    return default if value is None else value


def _env_float(*names: str) -> float | None:
    value = _env_str(*names)
    if value is None:
        return None
    return float(value)


def _env_bool(*names: str, default: bool = False) -> bool:
    value = _env_str(*names)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_bool_optional(*names: str) -> bool | None:
    value = _env_str(*names)
    if value is None:
        return None
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(slots=True)
class SolverAgentSource:
    raw: str
    kind: str
    local_path: str | None = None
    repo_url: str | None = None
    agent_file: str | None = None
    commit_sha: str | None = None

    def to_dict(self) -> dict[str, str]:
        payload = {
            "raw": self.raw,
            "kind": self.kind,
        }
        if self.local_path:
            payload["local_path"] = self.local_path
        if self.repo_url:
            payload["repo_url"] = self.repo_url
        if self.agent_file:
            payload["agent_file"] = self.agent_file
        if self.commit_sha:
            payload["commit_sha"] = self.commit_sha
        return payload


@dataclass(slots=True)
class RunConfig:
    """Runtime configuration for staged SWE commands."""

    workspace_root: Path = field(default_factory=Path.cwd)
    github_token: str | None = field(
        default_factory=lambda: os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN"),
    )
    github_tokens: str | None = field(
        default_factory=lambda: os.environ.get("GITHUB_TOKENS"),
    )
    # Dedicated owner-scoped token used only for write paths (auto-merging the
    # winning challenger PR into the watched base repo). Kept separate from the
    # rotation pool in `github_tokens` so a non-owner rotation token can never
    # be selected for the merge call (which would 404). Falls back to
    # GITHUB_TOKEN_UNARBOS, then GITHUB_TOKEN, then the first token in
    # GITHUB_TOKENS.
    github_merge_token: str | None = field(
        default_factory=lambda: (
            os.environ.get("GITHUB_MERGE_TOKEN")
            or os.environ.get("GITHUB_TOKEN_UNARBOS")
            or os.environ.get("GITHUB_TOKEN")
        ),
    )
    openrouter_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY"))
    cursor_api_key: str | None = field(default_factory=lambda: os.environ.get("CURSOR_API_KEY"))
    baseline_model: str | None = field(default_factory=lambda: _env_str("BASELINE_MODEL", "OPENROUTER_BASELINE_MODEL"))
    generator_model: str | None = field(default_factory=lambda: _env_str("GENERATOR_MODEL", "OPENROUTER_GENERATOR_MODEL"))
    solver_model: str | None = None
    eval_model: str | None = field(default_factory=lambda: _env_str("EVAL_MODEL", "OPENROUTER_EVAL_MODEL"))
    agent_timeout: int = 600
    solver_max_requests: int | None = field(default_factory=lambda: _env_int("SOLVER_MAX_REQUESTS"))
    solver_max_total_tokens: int | None = field(default_factory=lambda: _env_int("SOLVER_MAX_TOTAL_TOKENS"))
    solver_max_prompt_tokens: int | None = field(default_factory=lambda: _env_int("SOLVER_MAX_PROMPT_TOKENS"))
    solver_max_completion_tokens: int | None = field(default_factory=lambda: _env_int("SOLVER_MAX_COMPLETION_TOKENS"))
    solver_max_cost: float | None = field(default_factory=lambda: _env_float("SOLVER_MAX_COST"))
    solver_max_tokens_per_request: int | None = field(default_factory=lambda: _env_int("SOLVER_MAX_TOKENS_PER_REQUEST"))
    solver_provider_sort: str | None = field(default_factory=lambda: _env_str("SOLVER_PROVIDER_SORT", "OPENROUTER_PROVIDER_SORT"))
    solver_provider_only: str | None = field(default_factory=lambda: _env_str("SOLVER_PROVIDER_ONLY", "OPENROUTER_PROVIDER_ONLY"))
    solver_provider_allow_fallbacks: bool | None = field(
        default_factory=lambda: _env_bool_optional("SOLVER_PROVIDER_ALLOW_FALLBACKS", "OPENROUTER_PROVIDER_ALLOW_FALLBACKS"),
    )
    solver_provider_min_throughput_p50: float | None = field(
        default_factory=lambda: _env_float("SOLVER_PROVIDER_MIN_THROUGHPUT_P50", "OPENROUTER_PROVIDER_MIN_THROUGHPUT_P50"),
    )
    solver_provider_min_throughput_p90: float | None = field(
        default_factory=lambda: _env_float("SOLVER_PROVIDER_MIN_THROUGHPUT_P90", "OPENROUTER_PROVIDER_MIN_THROUGHPUT_P90"),
    )
    random_seed: int | None = None
    max_mining_attempts: int = 50
    http_timeout: float = 30.0
    solver_backend: str = "claude"
    solve_agent: str | None = None
    docker_solver_image: str | None = None
    solver_agent_source: SolverAgentSource | None = None
    docker_solver_memory: str = "2g"
    docker_solver_cpus: str = "1"
    docker_solver_pids_limit: int = 256
    docker_solver_tmp_size: str = "128m"
    docker_solver_workdir_size: str = "2g"
    docker_solver_nofile_limit: int = 4096
    docker_solver_max_output_bytes: int = 1_000_000
    docker_solver_drop_caps: bool = True
    docker_solver_no_new_privileges: bool = True
    docker_solver_read_only_rootfs: bool = True
    docker_solver_user: str | None = None
    docker_solver_no_cache: bool = False
    validate_netuid: int = 66
    validate_network: str | None = None
    validate_subtensor_endpoint: str | None = None
    validate_duel_rounds: int = 50
    validate_win_margin: int = 0
    validate_max_concurrency: int = 1
    validate_round_concurrency: int = 25
    validate_candidate_timeout_streak_limit: int = 5
    validate_task_pool_target: int = 50
    validate_task_pool_static: bool = True
    validate_pool_filler_concurrency: int = 24
    validate_task_pool_refresh_count: int = 0
    validate_task_pool_refresh_interval_seconds: int = 0
    validate_task_pool_fill_from_saved: bool = field(default_factory=lambda: _env_bool("VALIDATE_TASK_POOL_FILL_FROM_SAVED"))
    validate_task_cleanup_min_age_seconds: int = 3600
    validate_weight_interval_blocks: int = 360
    validate_king_window_size: int = 5
    validate_poll_interval_seconds: int = 600
    validate_duel_timeout_seconds: int = 3600
    validate_max_duels: int | None = None
    validate_min_commitment_block: int | None = None
    validate_hotkey_spent_since_block: int | None = field(default_factory=lambda: _env_int_default("VALIDATE_HOTKEY_SPENT_SINCE_BLOCK", 8_104_340))
    validate_queue_size: int | None = None
    validate_wallet_name: str | None = None
    validate_wallet_hotkey: str | None = None
    validate_wallet_path: str | None = None
    validate_github_pr_watch: bool = field(default_factory=lambda: _env_bool("VALIDATE_GITHUB_PR_WATCH"))
    validate_github_pr_repo: str = field(default_factory=lambda: _env_str("VALIDATE_GITHUB_PR_REPO") or "unarbos/ninja")
    validate_github_pr_base: str = field(default_factory=lambda: _env_str("VALIDATE_GITHUB_PR_BASE") or "main")
    validate_github_pr_require_checks: bool = field(default_factory=lambda: _env_bool("VALIDATE_GITHUB_PR_REQUIRE_CHECKS", default=True))
    validate_github_pr_include_drafts: bool = field(default_factory=lambda: _env_bool("VALIDATE_GITHUB_PR_INCLUDE_DRAFTS"))
    validate_github_pr_only: bool = field(default_factory=lambda: _env_bool("VALIDATE_GITHUB_PR_ONLY"))
    validate_github_conflict_resolver_max_tokens: int = field(
        default_factory=lambda: _env_int_default("VALIDATE_GITHUB_CONFLICT_RESOLVER_MAX_TOKENS", 32_000)
    )
    validate_github_pr_cleanup: bool = field(default_factory=lambda: _env_bool("VALIDATE_GITHUB_PR_CLEANUP"))
    validate_github_pr_cleanup_stale_after_hours: int = field(default_factory=lambda: _env_int_default("VALIDATE_GITHUB_PR_CLEANUP_STALE_AFTER_HOURS", 24))
    validate_github_pr_missing_commitment_notice_after_minutes: int = field(default_factory=lambda: _env_int_default("VALIDATE_GITHUB_PR_MISSING_COMMITMENT_NOTICE_AFTER_MINUTES", 30))
    validate_github_pr_cleanup_max_pages: int = field(default_factory=lambda: _env_int_default("VALIDATE_GITHUB_PR_CLEANUP_MAX_PAGES", 3))
    debug: bool = False

    @property
    def tasks_root(self) -> Path:
        return self.workspace_root / "workspace" / "tasks"

    @property
    def task_generation_timeout(self) -> int:
        return max(self.agent_timeout, 300)

    @property
    def validate_root(self) -> Path:
        return self.workspace_root / "workspace" / "validate" / f"netuid-{self.validate_netuid}"

    @property
    def use_docker_solver(self) -> bool:
        return self.solver_backend == "docker-file"

    @property
    def use_cursor_solver(self) -> bool:
        return self.solver_backend == "cursor"

    @property
    def use_claw_solver(self) -> bool:
        return self.solver_backend == "claw"

    @property
    def use_claude_solver(self) -> bool:
        return self.solver_backend == "claude"
