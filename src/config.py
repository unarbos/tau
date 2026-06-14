from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from tau.io.upstream_request_policy import UpstreamRequestPolicy, build_upstream_request_policy


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


def _env_bytes_default(name: str, default: int) -> int:
    value = _env_str(name)
    if value is None:
        return default
    return _parse_bytes(value)


def _parse_bytes(value: str) -> int:
    clean = value.strip().lower().replace("_", "")
    if not clean:
        raise ValueError("byte value cannot be empty")
    units = {
        "": 1,
        "b": 1,
        "k": 1024,
        "kb": 1024,
        "m": 1024**2,
        "mb": 1024**2,
        "g": 1024**3,
        "gb": 1024**3,
        "t": 1024**4,
        "tb": 1024**4,
    }
    number = clean
    multiplier = 1
    for suffix, scale in sorted(units.items(), key=lambda item: len(item[0]), reverse=True):
        if suffix and clean.endswith(suffix):
            number = clean[: -len(suffix)]
            multiplier = scale
            break
    return int(float(number) * multiplier)


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


def _split_env_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.replace("\n", ",").split(",") if item.strip()]


def _github_merge_token_env() -> str | None:
    return _env_str("GITHUB_MERGE_TOKEN", "GITHUB_TOKEN_UNARBOS")


def _github_read_token_env() -> str | None:
    reserved = _github_merge_token_env()
    for value in (
        _env_str("GITHUB_TASK_TOKEN"),
        _env_str("GITHUB_READ_TOKEN"),
        _env_str("GITHUB_TOKEN"),
        _env_str("GH_TOKEN"),
    ):
        if value and value != reserved:
            return value
    return None


def _github_read_tokens_env() -> str | None:
    reserved = _github_merge_token_env()
    tokens = [token for token in _split_env_list(os.environ.get("GITHUB_TOKENS")) if token != reserved]
    return ",".join(tokens) or None


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
        default_factory=_github_read_token_env,
    )
    github_tokens: str | None = field(
        default_factory=_github_read_tokens_env,
    )
    # Dedicated owner-scoped token used only for write paths (publishing the
    # winning private submission into the public base repo). This intentionally
    # does not fall back to task-generation read tokens; set GITHUB_MERGE_TOKEN
    # or GITHUB_TOKEN_UNARBOS for promotion publishing.
    github_merge_token: str | None = field(
        default_factory=_github_merge_token_env,
    )
    openrouter_api_key: str | None = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY"))
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
    solver_text_only: bool = field(
        default_factory=lambda: _env_bool("SOLVER_TEXT_ONLY", "OPENROUTER_SOLVER_TEXT_ONLY"),
    )
    solver_shell_tools: bool = field(
        default_factory=lambda: _env_bool("SOLVER_SHELL_TOOLS", "OPENROUTER_SOLVER_SHELL_TOOLS"),
    )
    solver_empty_response_retries: int | None = field(
        default_factory=lambda: _env_int("SOLVER_EMPTY_RESPONSE_RETRIES", "OPENROUTER_SOLVER_EMPTY_RESPONSE_RETRIES"),
    )
    solver_rate_limit_retries: int | None = field(
        default_factory=lambda: _env_int("SOLVER_RATE_LIMIT_RETRIES", "OPENROUTER_RATE_LIMIT_RETRIES"),
    )
    solver_temperature: float | None = field(
        default_factory=lambda: _env_float("SOLVER_TEMPERATURE", "OPENROUTER_SOLVER_TEMPERATURE"),
    )
    solver_proxy_cache_dir: Path | None = field(
        default_factory=lambda: (
            Path(value).expanduser() if (value := os.environ.get("PROXY_CACHE_DIR")) else None
        ),
    )
    solver_proxy_replay_dir: Path | None = field(
        default_factory=lambda: (
            Path(value).expanduser() if (value := os.environ.get("PROXY_REPLAY_DIR")) else None
        ),
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
    docker_solver_start_timeout_seconds: int = field(
        default_factory=lambda: _env_int_default("TAU_DOCKER_START_TIMEOUT_SECONDS", 90),
    )
    docker_solver_start_retries: int = field(
        default_factory=lambda: _env_int_default("TAU_DOCKER_START_RETRIES", 2),
    )
    docker_solver_start_retry_delay_seconds: float = field(
        default_factory=lambda: _env_float("TAU_DOCKER_START_RETRY_DELAY_SECONDS") or 5.0,
    )
    docker_solver_start_concurrency: int = field(
        default_factory=lambda: _env_int_default("TAU_DOCKER_START_CONCURRENCY", 8),
    )
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
    validate_task_archive_enabled: bool = field(default_factory=lambda: _env_bool("VALIDATE_TASK_ARCHIVE_ENABLED"))
    validate_task_archive_hf_dataset: str | None = field(default_factory=lambda: _env_str("VALIDATE_TASK_ARCHIVE_HF_DATASET"))
    validate_task_archive_hf_token_env: str = field(default_factory=lambda: _env_str("VALIDATE_TASK_ARCHIVE_HF_TOKEN_ENV") or "HF_TOKEN")
    validate_task_archive_per_hour: int = field(default_factory=lambda: _env_int_default("VALIDATE_TASK_ARCHIVE_PER_HOUR", 10))
    validate_min_free_disk_bytes: int = field(
        default_factory=lambda: _env_bytes_default("VALIDATE_MIN_FREE_DISK_BYTES", 100 * 1024**3),
    )
    validate_disk_cleanup_max_dirs_per_pass: int = field(
        default_factory=lambda: _env_int_default("VALIDATE_DISK_CLEANUP_MAX_DIRS_PER_PASS", 100),
    )
    record_rollouts: bool = field(default_factory=lambda: _env_bool("TAU_RECORD_ROLLOUTS"))
    rollout_root: Path | None = field(
        default_factory=lambda: (
            Path(value).expanduser()
            if (value := os.environ.get("TAU_ROLLOUT_ROOT"))
            else None
        ),
    )
    push_rollouts_to_hf: bool = field(default_factory=lambda: _env_bool("TAU_PUSH_ROLLOUTS_TO_HF"))
    rollout_hf_dataset: str | None = field(default_factory=lambda: _env_str("TAU_ROLLOUT_HF_DATASET"))
    rollout_hf_token_env: str = field(default_factory=lambda: _env_str("TAU_ROLLOUT_HF_TOKEN_ENV") or "HF_TOKEN")
    rollout_export_format: str = field(default_factory=lambda: _env_str("TAU_ROLLOUT_EXPORT_FORMAT") or "jsonl")
    clear_uploaded_rollouts: bool = field(default_factory=lambda: _env_bool("TAU_CLEAR_UPLOADED_ROLLOUTS"))
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
    validate_github_conflict_resolver_max_tokens: int = field(
        default_factory=lambda: _env_int_default("VALIDATE_GITHUB_CONFLICT_RESOLVER_MAX_TOKENS", 32_000)
    )
    validate_publish_repo: str = field(default_factory=lambda: _env_str("VALIDATE_PUBLISH_REPO") or "unarbos/ninja")
    validate_publish_base: str = field(default_factory=lambda: _env_str("VALIDATE_PUBLISH_BASE") or "main")
    validate_private_submission_watch: bool = field(default_factory=lambda: _env_bool("VALIDATE_PRIVATE_SUBMISSION_WATCH"))
    validate_private_submission_only: bool = field(default_factory=lambda: _env_bool("VALIDATE_PRIVATE_SUBMISSION_ONLY"))
    validate_private_submission_root: Path | None = field(
        default_factory=lambda: (
            Path(value).expanduser()
            if (value := os.environ.get("VALIDATE_PRIVATE_SUBMISSION_ROOT"))
            else None
        ),
    )
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

    def resolved_rollout_root(self) -> Path:
        return self.rollout_root or (self.workspace_root / "workspace" / "rollouts")

    @property
    def use_docker_solver(self) -> bool:
        return self.solver_backend == "docker-file"

    @property
    def use_claw_solver(self) -> bool:
        return self.solver_backend == "claw"

    @property
    def use_claude_solver(self) -> bool:
        return self.solver_backend == "claude"

    @property
    def solver_upstream_request_policy(self) -> UpstreamRequestPolicy | None:
        return build_upstream_request_policy(
            text_only=self.solver_text_only,
            shell_tools=self.solver_shell_tools,
            empty_response_retries=self.solver_empty_response_retries,
            rate_limit_retries=self.solver_rate_limit_retries,
        )
