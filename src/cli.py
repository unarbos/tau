from __future__ import annotations

import argparse
import os
from pathlib import Path
from urllib.parse import urlparse

from config import RunConfig, SolverAgentSource
from pipeline import (
    compare_task_run,
    delete_task_run,
    evaluate_task_run,
    generate_task_run,
    solve_task_run,
)

_DEFAULT_CONCURRENCY = min(os.cpu_count() or 4, 8)
_DEFAULT_AGENT_FILE = "agent.py"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate, solve, compare, and evaluate SWE tasks as independent stages.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="Mine a commit and materialize a named task.")
    _add_shared_args(generate)
    generate.add_argument("--task", required=True, help="Unique name for the generated task.")
    generate.add_argument("--generator-model", help="Optional model override for task generation.")
    generate.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for repeatable mining order.",
    )
    generate.add_argument(
        "--max-mining-attempts",
        type=int,
        default=25,
        help="How many GitHub event/commit retries to attempt while mining a task.",
    )

    solve = subparsers.add_parser("solve", help="Run a named task with a named solver agent.")
    _add_shared_args(solve)
    solve.add_argument("--task", required=True, help="Existing generated task name.")
    solve.add_argument("--solution", required=True, help="Unique name for this solver run.")
    solve.add_argument(
        "--agent",
        required=True,
        help=(
            "Solver backend selector. Use 'claude' for the host Claude CLI, "
            "or pass a local agent.py file / repo root / GitHub repo URL for the Docker file solver."
        ),
    )
    _add_solver_args(solve)

    evaluate = subparsers.add_parser("eval", help="Evaluate ordered solution pairs for one named task.")
    _add_shared_args(evaluate)
    evaluate.add_argument("--task", required=True, help="Existing generated task name.")
    evaluate.add_argument(
        "--solutions",
        required=True,
        nargs="+",
        help="Ordered solution names to compare. Supports '--solutions A B' and '--solutions A,B'.",
    )
    evaluate.add_argument("--eval-model", help="Optional model override for evaluation.")
    evaluate.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for deterministic blind-candidate ordering.",
    )

    compare = subparsers.add_parser("compare", help="Compare two saved solutions by changed-line similarity.")
    _add_shared_args(compare)
    compare.add_argument("--task", required=True, help="Existing generated task name.")
    compare.add_argument(
        "--solutions",
        required=True,
        nargs="+",
        help="Exactly two solution names to compare. Supports '--solutions A B' and '--solutions A,B'.",
    )

    delete = subparsers.add_parser("delete", help="Delete saved task workspaces and related artifacts.")
    _add_shared_args(delete)
    delete.add_argument(
        "resource",
        nargs="?",
        choices=["task"],
        help="Optional resource type. Use 'task' for forms like 'tau delete task --all'.",
    )
    delete_group = delete.add_mutually_exclusive_group(required=True)
    delete_group.add_argument("--task", help="Delete one saved task by name.")
    delete_group.add_argument("--all", action="store_true", help="Delete all saved task workspaces.")

    validate = subparsers.add_parser(
        "validate",
        help="Run the live king-of-the-hill validator loop for subnet commitments.",
    )
    _add_shared_args(validate)
    _add_solver_args(validate)
    validate.set_defaults(agent_timeout=1800, docker_solver_max_output_bytes=100000000)
    validate.add_argument("--netuid", type=int, default=66, help="Subnet netuid to validate.")
    validate.add_argument("--network", help="Optional Bittensor network name or websocket endpoint.")
    validate.add_argument(
        "--subtensor-endpoint",
        help="Optional websocket endpoint that overrides --network for chain access.",
    )
    validate.add_argument("-N", "--duel-rounds", type=int, default=50, help="Decisive rounds per duel.")
    validate.add_argument("-K", "--win-margin", type=int, default=0, help="Extra decisive round wins over the king required to dethrone.")
    validate.add_argument("--max-concurrency", type=int, default=1, help="Max parallel duels (1 = serialized).")
    validate.add_argument("--round-concurrency", type=int, default=25, help="Max parallel rounds within a single duel.")
    validate.add_argument("--candidate-timeout-streak-limit", type=int, default=5, help="Stop submitting new rounds for a challenger after this many consecutive round timeouts.")
    validate.add_argument("--task-pool-target", type=int, default=50, help="Pre-solved tasks to keep in pool.")
    validate.add_argument("--pool-filler-concurrency", type=int, default=24, help="Parallel pool-filler threads.")
    validate.add_argument("--task-pool-refresh-count", type=int, default=5, help="Full-pool tasks to replace each refresh interval.")
    validate.add_argument("--task-pool-refresh-interval-seconds", type=int, default=3600, help="Seconds between full-pool refresh batches.")
    validate.add_argument(
        "--task-pool-fill-from-saved",
        action="store_true",
        help="Fill validator task pools by round-robin reusing saved task workspaces instead of fetching new tasks.",
    )
    validate.add_argument("--task-cleanup-min-age-seconds", type=int, default=3600, help="Minimum age before non-pool validate task dirs can be pruned.")
    validate.add_argument("--weight-interval-blocks", type=int, default=360, help="Blocks between weight sets.")
    validate.add_argument("--king-window-size", type=int, default=5, help="Number of recent kings to share emissions across (each gets 1/N).")
    validate.add_argument("--poll-interval-seconds", type=int, default=600, help="Seconds between chain submission refreshes.")
    validate.add_argument("--duel-timeout", type=int, default=7200, help="Max seconds a single duel may run before being cancelled.")
    validate.add_argument("--max-duels", type=int, help="Stop after this many completed duels.")
    validate.add_argument("--min-commitment-block", type=int, default=0, help="Ignore submissions before this block (0 = auto-set to current block at startup).")
    validate.add_argument("--hotkey-spent-since-block", type=int, help="Block cutoff for hotkey-spent history.")
    validate.add_argument("--queue-size", type=int, help="Max queued challengers.")
    validate.add_argument("--watch-github-prs", action="store_true", default=None, help="Accept eligible GitHub PR commitments as validator challengers.")
    validate.add_argument("--github-pr-repo", help="Repository whose PRs should be watched, in owner/name form.")
    validate.add_argument("--github-pr-base", help="Base branch for watched PRs.")
    validate.add_argument("--github-pr-no-require-checks", action="store_true", help="Queue watched PRs before required CI checks pass.")
    validate.add_argument("--github-pr-include-drafts", action="store_true", default=None, help="Include draft PRs in the watched PR queue.")
    validate.add_argument("--github-pr-only", action="store_true", default=None, help="Use only GitHub PR commitments as submissions.")
    validate.add_argument("--github-pr-cleanup", action="store_true", default=None, help="Label and close stale or invalid watched GitHub PRs.")
    validate.add_argument("--github-pr-cleanup-stale-after-hours", type=int, help="Close non-active watched PRs older than this many hours.")
    validate.add_argument("--github-pr-missing-commitment-notice-after-minutes", type=int, help="Comment on open watched PRs older than this many minutes when no matching on-chain PR commitment exists.")
    validate.add_argument("--github-pr-cleanup-max-pages", type=int, help="Maximum open PR pages to scan per cleanup pass.")
    validate.add_argument("--wallet-name", required=True, help="Wallet coldkey name.")
    validate.add_argument("--wallet-hotkey", required=True, help="Wallet hotkey name.")
    validate.add_argument("--wallet-path", help="Wallet path override.")

    fill_pool = subparsers.add_parser(
        "fill-pool",
        aliases=["generate-pool"],
        help="Run only the central validator task-pool generator, without duels or weights.",
    )
    _add_shared_args(fill_pool)
    _add_solver_args(fill_pool)
    fill_pool.set_defaults(agent_timeout=1800, docker_solver_max_output_bytes=100000000)
    fill_pool.add_argument("--netuid", type=int, default=66, help="Subnet netuid whose validate workspace should be filled.")
    fill_pool.add_argument("--generator-model", help="Optional model override for task generation.")
    fill_pool.add_argument("--max-mining-attempts", type=int, default=50, help="How many GitHub event/commit retries to attempt while mining a task.")
    fill_pool.add_argument("--task-pool-target", type=int, default=200, help="Pre-solved tasks to keep in the central pool.")
    fill_pool.add_argument("--pool-filler-concurrency", type=int, default=24, help="Parallel central pool generator threads.")
    fill_pool.add_argument("--task-pool-refresh-count", type=int, default=6, help="Tasks to replace each refresh interval after the pool is full.")
    fill_pool.add_argument("--task-pool-refresh-interval-seconds", type=int, default=3600, help="Seconds between refresh batches after the pool is full.")
    fill_pool.add_argument(
        "--task-pool-fill-from-saved",
        dest="task_pool_fill_from_saved",
        action="store_true",
        default=True,
        help="Prefer saved task workspaces before generating fresh tasks.",
    )
    fill_pool.add_argument(
        "--no-task-pool-fill-from-saved",
        dest="task_pool_fill_from_saved",
        action="store_false",
        help="Generate fresh tasks only.",
    )
    return parser


def main() -> None:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    try:
        if args.command == "generate":
            result = generate_task_run(task_name=args.task, config=_build_generate_config(args))
            print(f"generated {result.task_name}: {result.repo}@{result.commit_sha[:12]}")
            print(result.task_root)
            return
        if args.command == "solve":
            result = solve_task_run(
                task_name=args.task,
                solution_name=args.solution,
                config=_build_solve_config(args),
            )
            status = "success" if result.success else "failed"
            print(
                f"solved {result.task_name}/{result.solution_name}: "
                f"{result.repo}@{result.commit_sha[:12]} -> {status}"
            )
            print(result.solution_root)
            return
        if args.command == "eval":
            result = evaluate_task_run(
                task_name=args.task,
                solution_names=_normalize_solution_names(args.solutions),
                config=_build_eval_config(args),
            )
            print(
                f"evaluated {result.task_name}/{result.eval_name}: "
                f"{result.repo}@{result.commit_sha[:12]} -> {result.comparison_count} comparisons"
            )
            print(result.eval_root)
            return
        if args.command == "compare":
            result = compare_task_run(
                task_name=args.task,
                solution_names=_normalize_compare_solution_names(args.solutions),
                config=_build_compare_config(args),
            )
            print(
                f"compared {result.task_name}/{result.comparison_name}: "
                f"{result.repo}@{result.commit_sha[:12]} -> "
                f"{result.matched_changed_lines}/{result.scored_positions} matching changed lines "
                f"({result.similarity_ratio:.2%})"
            )
            print(result.comparison_root)
            return
        if args.command == "delete":
            result = delete_task_run(
                task_name=getattr(args, "task", None),
                delete_all=getattr(args, "all", False),
                config=_build_delete_config(args),
            )
            if result.deleted_all:
                print(f"deleted {result.deleted_count} task workspace(s)")
            else:
                print(f"deleted task {result.deleted_tasks[0]}")
            return
        if args.command == "validate":
            from validate import validate_loop_run

            result = validate_loop_run(config=_build_validate_config(args))
            print(
                f"validate loop exited with king uid={result.king_uid} "
                f"hotkey={result.king_hotkey} repo={result.king_repo}"
            )
            print(result.validate_root)
            return
        if args.command in {"fill-pool", "generate-pool"}:
            from validate import fill_task_pool_run

            result = fill_task_pool_run(config=_build_fill_pool_config(args))
            print(
                f"pool generator exited with king uid={result.king_uid} "
                f"hotkey={result.king_hotkey} repo={result.king_repo}"
            )
            print(result.validate_root)
            return
        parser.error(f"Unknown command: {args.command}")
    except Exception as exc:  # noqa: BLE001
        if getattr(args, "debug", False):
            raise
        parser.exit(1, f"error: {exc}\n")


def _add_shared_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path.cwd(),
        help="Root directory that will receive workspace/tasks/... artifacts.",
    )
    parser.add_argument(
        "--agent-timeout",
        type=int,
        default=600,
        help="Timeout in seconds for each model or solver invocation.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging for the selected stage.",
    )


def _build_generate_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        generator_model=_arg_or_env(args.generator_model, "GENERATOR_MODEL", "OPENROUTER_GENERATOR_MODEL"),
        agent_timeout=args.agent_timeout,
        random_seed=args.seed,
        max_mining_attempts=args.max_mining_attempts,
        debug=args.debug,
    )


def _build_solve_config(args: argparse.Namespace) -> RunConfig:
    solver_backend, agent_source = _resolve_solve_target(args.agent, cwd=Path.cwd())
    defaults = RunConfig()
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        solver_model=args.solver_model,
        agent_timeout=args.agent_timeout,
        solver_max_requests=_arg_or_env_int(args.solver_max_requests, "SOLVER_MAX_REQUESTS"),
        solver_max_total_tokens=_arg_or_env_int(args.solver_max_total_tokens, "SOLVER_MAX_TOTAL_TOKENS"),
        solver_max_prompt_tokens=_arg_or_env_int(args.solver_max_prompt_tokens, "SOLVER_MAX_PROMPT_TOKENS"),
        solver_max_completion_tokens=_arg_or_env_int(args.solver_max_completion_tokens, "SOLVER_MAX_COMPLETION_TOKENS"),
        solver_max_cost=_arg_or_env_float(args.solver_max_cost, "SOLVER_MAX_COST"),
        solver_max_tokens_per_request=_arg_or_env_int(
            args.solver_max_tokens_per_request,
            "SOLVER_MAX_TOKENS_PER_REQUEST",
        ),
        solver_provider_sort=_arg_or_env(args.solver_provider_sort, "SOLVER_PROVIDER_SORT", "OPENROUTER_PROVIDER_SORT"),
        solver_provider_only=_arg_or_env(args.solver_provider_only, "SOLVER_PROVIDER_ONLY", "OPENROUTER_PROVIDER_ONLY"),
        solver_provider_allow_fallbacks=(
            False if args.solver_provider_disable_fallbacks else defaults.solver_provider_allow_fallbacks
        ),
        solver_provider_min_throughput_p50=_arg_or_env_float(
            args.solver_provider_min_throughput_p50,
            "SOLVER_PROVIDER_MIN_THROUGHPUT_P50",
            "OPENROUTER_PROVIDER_MIN_THROUGHPUT_P50",
        ),
        solver_provider_min_throughput_p90=_arg_or_env_float(
            args.solver_provider_min_throughput_p90,
            "SOLVER_PROVIDER_MIN_THROUGHPUT_P90",
            "OPENROUTER_PROVIDER_MIN_THROUGHPUT_P90",
        ),
        random_seed=args.seed,
        solver_backend=solver_backend,
        solve_agent=args.agent,
        docker_solver_image=args.docker_solver_image,
        solver_agent_source=agent_source,
        docker_solver_memory=args.docker_solver_memory,
        docker_solver_cpus=args.docker_solver_cpus,
        docker_solver_pids_limit=args.docker_solver_pids_limit,
        docker_solver_tmp_size=args.docker_solver_tmp_size,
        docker_solver_workdir_size=args.docker_solver_workdir_size,
        docker_solver_nofile_limit=args.docker_solver_nofile_limit,
        docker_solver_max_output_bytes=args.docker_solver_max_output_bytes,
        docker_solver_drop_caps=not args.docker_solver_keep_caps,
        docker_solver_no_new_privileges=not args.docker_solver_allow_privilege_escalation,
        docker_solver_read_only_rootfs=not args.docker_solver_writeable_rootfs,
        docker_solver_user=args.docker_solver_user,
        docker_solver_no_cache=args.docker_solver_no_cache,
        debug=args.debug,
    )


def _build_eval_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        eval_model=_arg_or_env(args.eval_model, "EVAL_MODEL", "OPENROUTER_EVAL_MODEL"),
        agent_timeout=args.agent_timeout,
        random_seed=args.seed,
        debug=args.debug,
    )


def _build_compare_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        agent_timeout=args.agent_timeout,
        debug=args.debug,
    )


def _build_delete_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        agent_timeout=args.agent_timeout,
        debug=args.debug,
    )


def _build_validate_config(args: argparse.Namespace) -> RunConfig:
    defaults = RunConfig()
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        solver_model=args.solver_model,
        agent_timeout=args.agent_timeout,
        solver_max_requests=_arg_or_env_int(args.solver_max_requests, "SOLVER_MAX_REQUESTS"),
        solver_max_total_tokens=_arg_or_env_int(args.solver_max_total_tokens, "SOLVER_MAX_TOTAL_TOKENS"),
        solver_max_prompt_tokens=_arg_or_env_int(args.solver_max_prompt_tokens, "SOLVER_MAX_PROMPT_TOKENS"),
        solver_max_completion_tokens=_arg_or_env_int(args.solver_max_completion_tokens, "SOLVER_MAX_COMPLETION_TOKENS"),
        solver_max_cost=_arg_or_env_float(args.solver_max_cost, "SOLVER_MAX_COST"),
        solver_max_tokens_per_request=_arg_or_env_int(
            args.solver_max_tokens_per_request,
            "SOLVER_MAX_TOKENS_PER_REQUEST",
        ),
        solver_provider_sort=_arg_or_env(args.solver_provider_sort, "SOLVER_PROVIDER_SORT", "OPENROUTER_PROVIDER_SORT"),
        solver_provider_only=_arg_or_env(args.solver_provider_only, "SOLVER_PROVIDER_ONLY", "OPENROUTER_PROVIDER_ONLY"),
        solver_provider_allow_fallbacks=(
            False if args.solver_provider_disable_fallbacks else defaults.solver_provider_allow_fallbacks
        ),
        solver_provider_min_throughput_p50=_arg_or_env_float(
            args.solver_provider_min_throughput_p50,
            "SOLVER_PROVIDER_MIN_THROUGHPUT_P50",
            "OPENROUTER_PROVIDER_MIN_THROUGHPUT_P50",
        ),
        solver_provider_min_throughput_p90=_arg_or_env_float(
            args.solver_provider_min_throughput_p90,
            "SOLVER_PROVIDER_MIN_THROUGHPUT_P90",
            "OPENROUTER_PROVIDER_MIN_THROUGHPUT_P90",
        ),
        random_seed=args.seed,
        docker_solver_image=args.docker_solver_image,
        docker_solver_memory=args.docker_solver_memory,
        docker_solver_cpus=args.docker_solver_cpus,
        docker_solver_pids_limit=args.docker_solver_pids_limit,
        docker_solver_tmp_size=args.docker_solver_tmp_size,
        docker_solver_workdir_size=args.docker_solver_workdir_size,
        docker_solver_nofile_limit=args.docker_solver_nofile_limit,
        docker_solver_max_output_bytes=args.docker_solver_max_output_bytes,
        docker_solver_drop_caps=not args.docker_solver_keep_caps,
        docker_solver_no_new_privileges=not args.docker_solver_allow_privilege_escalation,
        docker_solver_read_only_rootfs=not args.docker_solver_writeable_rootfs,
        docker_solver_user=args.docker_solver_user,
        docker_solver_no_cache=args.docker_solver_no_cache,
        validate_netuid=args.netuid,
        validate_network=args.network,
        validate_subtensor_endpoint=args.subtensor_endpoint,
        validate_duel_rounds=args.duel_rounds,
        validate_win_margin=args.win_margin,
        validate_max_concurrency=args.max_concurrency,
        validate_round_concurrency=args.round_concurrency,
        validate_candidate_timeout_streak_limit=args.candidate_timeout_streak_limit,
        validate_task_pool_target=args.task_pool_target,
        validate_pool_filler_concurrency=args.pool_filler_concurrency,
        validate_task_pool_refresh_count=args.task_pool_refresh_count,
        validate_task_pool_refresh_interval_seconds=args.task_pool_refresh_interval_seconds,
        validate_task_pool_fill_from_saved=args.task_pool_fill_from_saved,
        validate_task_cleanup_min_age_seconds=args.task_cleanup_min_age_seconds,
        validate_weight_interval_blocks=args.weight_interval_blocks,
        validate_king_window_size=args.king_window_size,
        validate_poll_interval_seconds=args.poll_interval_seconds,
        validate_duel_timeout_seconds=args.duel_timeout,
        validate_max_duels=args.max_duels,
        validate_min_commitment_block=args.min_commitment_block,
        validate_hotkey_spent_since_block=(
            args.hotkey_spent_since_block
            if args.hotkey_spent_since_block is not None
            else defaults.validate_hotkey_spent_since_block
        ),
        validate_queue_size=args.queue_size,
        validate_wallet_name=args.wallet_name,
        validate_wallet_hotkey=args.wallet_hotkey,
        validate_wallet_path=args.wallet_path,
        validate_github_pr_watch=(
            defaults.validate_github_pr_watch
            if args.watch_github_prs is None
            else bool(args.watch_github_prs)
        ),
        validate_github_pr_repo=args.github_pr_repo or defaults.validate_github_pr_repo,
        validate_github_pr_base=args.github_pr_base or defaults.validate_github_pr_base,
        validate_github_pr_require_checks=(
            False
            if args.github_pr_no_require_checks
            else defaults.validate_github_pr_require_checks
        ),
        validate_github_pr_include_drafts=(
            defaults.validate_github_pr_include_drafts
            if args.github_pr_include_drafts is None
            else bool(args.github_pr_include_drafts)
        ),
        validate_github_pr_only=(
            defaults.validate_github_pr_only
            if args.github_pr_only is None
            else bool(args.github_pr_only)
        ),
        validate_github_pr_cleanup=(
            defaults.validate_github_pr_cleanup
            if args.github_pr_cleanup is None
            else bool(args.github_pr_cleanup)
        ),
        validate_github_pr_cleanup_stale_after_hours=(
            args.github_pr_cleanup_stale_after_hours
            if args.github_pr_cleanup_stale_after_hours is not None
            else defaults.validate_github_pr_cleanup_stale_after_hours
        ),
        validate_github_pr_missing_commitment_notice_after_minutes=(
            args.github_pr_missing_commitment_notice_after_minutes
            if args.github_pr_missing_commitment_notice_after_minutes is not None
            else defaults.validate_github_pr_missing_commitment_notice_after_minutes
        ),
        validate_github_pr_cleanup_max_pages=(
            args.github_pr_cleanup_max_pages
            if args.github_pr_cleanup_max_pages is not None
            else defaults.validate_github_pr_cleanup_max_pages
        ),
        debug=args.debug,
    )


def _build_fill_pool_config(args: argparse.Namespace) -> RunConfig:
    defaults = RunConfig()
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        generator_model=_arg_or_env(args.generator_model, "GENERATOR_MODEL", "OPENROUTER_GENERATOR_MODEL"),
        solver_model=args.solver_model,
        agent_timeout=args.agent_timeout,
        max_mining_attempts=args.max_mining_attempts,
        solver_max_requests=_arg_or_env_int(args.solver_max_requests, "SOLVER_MAX_REQUESTS"),
        solver_max_total_tokens=_arg_or_env_int(args.solver_max_total_tokens, "SOLVER_MAX_TOTAL_TOKENS"),
        solver_max_prompt_tokens=_arg_or_env_int(args.solver_max_prompt_tokens, "SOLVER_MAX_PROMPT_TOKENS"),
        solver_max_completion_tokens=_arg_or_env_int(args.solver_max_completion_tokens, "SOLVER_MAX_COMPLETION_TOKENS"),
        solver_max_cost=_arg_or_env_float(args.solver_max_cost, "SOLVER_MAX_COST"),
        solver_max_tokens_per_request=_arg_or_env_int(
            args.solver_max_tokens_per_request,
            "SOLVER_MAX_TOKENS_PER_REQUEST",
        ),
        solver_provider_sort=_arg_or_env(args.solver_provider_sort, "SOLVER_PROVIDER_SORT", "OPENROUTER_PROVIDER_SORT"),
        solver_provider_only=_arg_or_env(args.solver_provider_only, "SOLVER_PROVIDER_ONLY", "OPENROUTER_PROVIDER_ONLY"),
        solver_provider_allow_fallbacks=(
            False if args.solver_provider_disable_fallbacks else defaults.solver_provider_allow_fallbacks
        ),
        solver_provider_min_throughput_p50=_arg_or_env_float(
            args.solver_provider_min_throughput_p50,
            "SOLVER_PROVIDER_MIN_THROUGHPUT_P50",
            "OPENROUTER_PROVIDER_MIN_THROUGHPUT_P50",
        ),
        solver_provider_min_throughput_p90=_arg_or_env_float(
            args.solver_provider_min_throughput_p90,
            "SOLVER_PROVIDER_MIN_THROUGHPUT_P90",
            "OPENROUTER_PROVIDER_MIN_THROUGHPUT_P90",
        ),
        random_seed=args.seed,
        docker_solver_image=args.docker_solver_image,
        docker_solver_memory=args.docker_solver_memory,
        docker_solver_cpus=args.docker_solver_cpus,
        docker_solver_pids_limit=args.docker_solver_pids_limit,
        docker_solver_tmp_size=args.docker_solver_tmp_size,
        docker_solver_workdir_size=args.docker_solver_workdir_size,
        docker_solver_nofile_limit=args.docker_solver_nofile_limit,
        docker_solver_max_output_bytes=args.docker_solver_max_output_bytes,
        docker_solver_drop_caps=not args.docker_solver_keep_caps,
        docker_solver_no_new_privileges=not args.docker_solver_allow_privilege_escalation,
        docker_solver_read_only_rootfs=not args.docker_solver_writeable_rootfs,
        docker_solver_user=args.docker_solver_user,
        docker_solver_no_cache=args.docker_solver_no_cache,
        validate_netuid=args.netuid,
        validate_task_pool_target=args.task_pool_target,
        validate_pool_filler_concurrency=args.pool_filler_concurrency,
        validate_task_pool_refresh_count=args.task_pool_refresh_count,
        validate_task_pool_refresh_interval_seconds=args.task_pool_refresh_interval_seconds,
        validate_task_pool_fill_from_saved=args.task_pool_fill_from_saved,
        debug=args.debug,
    )


def _arg_or_env(value: str | None, *env_names: str) -> str | None:
    if value:
        return value
    for name in env_names:
        env_value = os.environ.get(name)
        if env_value:
            return env_value
    return None


def _arg_or_env_int(value: int | None, *env_names: str) -> int | None:
    if value is not None:
        return value
    env_value = _arg_or_env(None, *env_names)
    return int(env_value) if env_value is not None else None


def _arg_or_env_float(value: float | None, *env_names: str) -> float | None:
    if value is not None:
        return value
    env_value = _arg_or_env(None, *env_names)
    return float(env_value) if env_value is not None else None


def _normalize_solution_names(raw_values: list[str]) -> list[str]:
    names: list[str] = []
    for raw_value in raw_values:
        parts = [part.strip() for part in raw_value.split(",")]
        names.extend(part for part in parts if part)
    if len(names) < 2:
        raise ValueError("eval requires at least two solution names")
    return names


def _normalize_compare_solution_names(raw_values: list[str]) -> list[str]:
    names = _normalize_solution_names(raw_values)
    if len(names) != 2:
        raise ValueError("compare requires exactly two solution names")
    return names


def _resolve_solve_target(raw_value: str, *, cwd: Path) -> tuple[str, SolverAgentSource | None]:
    normalized = raw_value.strip().lower()
    if normalized == "claude":
        return "claude", None
    if normalized == "claw":
        return "claw", None
    return "docker-file", _resolve_agent_source(raw_value, cwd=cwd)


def _resolve_agent_source(raw_value: str, *, cwd: Path) -> SolverAgentSource:
    value = raw_value.strip()
    if not value:
        raise ValueError("--agent cannot be empty")

    candidate_path = Path(value).expanduser()
    if candidate_path.exists():
        resolved = _resolve_local_agent_file(candidate_path.resolve())
        return SolverAgentSource(
            raw=value,
            kind="local_file",
            local_path=str(resolved),
            agent_file=resolved.name,
        )

    if candidate_path.is_absolute():
        raise ValueError(f"--agent local path does not exist: {candidate_path}")

    relative_candidate = (cwd / candidate_path).resolve()
    if relative_candidate.exists():
        resolved = _resolve_local_agent_file(relative_candidate)
        return SolverAgentSource(
            raw=value,
            kind="local_file",
            local_path=str(resolved),
            agent_file=resolved.name,
        )

    repo_url, agent_file, commit_sha = _normalize_github_agent_source(value)
    if repo_url is None:
        raise ValueError(
            "--agent must be an existing Python file, a directory containing agent.py, "
            "or a GitHub repo URL/shorthand like "
            "'github.com/org/repo', 'org/repo@commit', or "
            "'https://github.com/org/repo/commit/<sha>'"
        )

    return SolverAgentSource(
        raw=value,
        kind="github_repo",
        repo_url=repo_url,
        agent_file=agent_file,
        commit_sha=commit_sha,
    )


def _normalize_github_agent_source(raw_value: str) -> tuple[str | None, str, str | None]:
    cleaned = raw_value.strip().rstrip("/")
    pinned_match = _split_repo_commit_ref(cleaned)
    if pinned_match is not None:
        repo_path, commit_sha = pinned_match
        return f"https://github.com/{repo_path}.git", _DEFAULT_AGENT_FILE, commit_sha

    if "://" not in cleaned and cleaned.count("/") >= 1 and not cleaned.startswith("github.com/"):
        parts = [part for part in cleaned.split("/") if part]
        if len(parts) >= 2:
            repo_path = "/".join(parts[:2])
            return f"https://github.com/{repo_path}.git", _DEFAULT_AGENT_FILE, None
        return None, _DEFAULT_AGENT_FILE, None

    parsed = urlparse(cleaned if "://" in cleaned else f"https://{cleaned}")
    if parsed.netloc.lower() != "github.com":
        return None, _DEFAULT_AGENT_FILE, None

    parts = [part for part in parsed.path.split("/") if part]
    if len(parts) < 2:
        return None, _DEFAULT_AGENT_FILE, None

    if len(parts) >= 4 and parts[2] == "commit":
        repo_path = "/".join(parts[:2])
        return f"https://github.com/{repo_path}.git", _DEFAULT_AGENT_FILE, parts[3]

    repo_parts = parts[:2]
    if len(parts) >= 5 and parts[2] == "blob":
        repo_path = "/".join(repo_parts)
        return f"https://github.com/{repo_path}.git", "/".join(parts[4:]), None

    repo_path = "/".join(repo_parts)
    return f"https://github.com/{repo_path}.git", _DEFAULT_AGENT_FILE, None


def _split_repo_commit_ref(raw_value: str) -> tuple[str, str] | None:
    if "@" not in raw_value or "://" in raw_value or raw_value.startswith("github.com/"):
        return None
    repo_path, commit_sha = raw_value.rsplit("@", 1)
    parts = [part for part in repo_path.split("/") if part]
    if len(parts) != 2 or not commit_sha:
        return None
    return "/".join(parts), commit_sha


def _resolve_local_agent_file(candidate: Path) -> Path:
    if candidate.is_file():
        if candidate.suffix != ".py":
            raise ValueError(f"--agent local file must be a Python file: {candidate}")
        return candidate
    if candidate.is_dir():
        agent_file = candidate / _DEFAULT_AGENT_FILE
        if agent_file.is_file():
            return agent_file
        raise ValueError(f"--agent local directory must contain {_DEFAULT_AGENT_FILE}: {candidate}")
    raise ValueError(f"--agent local path must be a Python file or directory: {candidate}")


def _add_solver_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--solver-model", help="Optional model override for solving.")
    parser.add_argument(
        "--seed",
        type=int,
        help="Optional random seed for deterministic solver-side choices.",
    )
    parser.add_argument(
        "--solver-max-requests",
        type=int,
        help="Maximum number of proxied OpenRouter requests allowed for a solve run.",
    )
    parser.add_argument(
        "--solver-max-total-tokens",
        type=int,
        help="Maximum total OpenRouter tokens allowed for a solve run.",
    )
    parser.add_argument(
        "--solver-max-prompt-tokens",
        type=int,
        help="Maximum prompt tokens allowed for a solve run.",
    )
    parser.add_argument(
        "--solver-max-completion-tokens",
        type=int,
        help="Maximum completion tokens allowed for a solve run.",
    )
    parser.add_argument(
        "--solver-max-cost",
        type=float,
        help="Maximum OpenRouter cost allowed for a solve run.",
    )
    parser.add_argument(
        "--solver-max-tokens-per-request",
        type=int,
        help="Maximum completion tokens to allow on any single proxied request.",
    )
    parser.add_argument(
        "--solver-provider-sort",
        choices=("price", "throughput", "latency"),
        help="OpenRouter provider sort policy for proxied solver requests.",
    )
    parser.add_argument(
        "--solver-provider-only",
        help="Comma-separated OpenRouter provider slugs to allow for proxied solver requests.",
    )
    parser.add_argument(
        "--solver-provider-disable-fallbacks",
        action="store_true",
        help="Disable OpenRouter fallbacks outside the ordered/allowed provider policy.",
    )
    parser.add_argument(
        "--solver-provider-min-throughput-p50",
        type=float,
        help="Prefer providers with at least this p50 throughput in tokens/sec.",
    )
    parser.add_argument(
        "--solver-provider-min-throughput-p90",
        type=float,
        help="Prefer providers with at least this p90 throughput in tokens/sec.",
    )
    parser.add_argument(
        "--docker-solver-image",
        help="Optional Docker image tag for the solver image. If omitted, one is derived.",
    )
    parser.add_argument(
        "--docker-solver-memory",
        default="2g",
        help="Docker memory limit for the solver container.",
    )
    parser.add_argument(
        "--docker-solver-cpus",
        default="2",
        help="Docker CPU limit for the solver container.",
    )
    parser.add_argument(
        "--docker-solver-pids-limit",
        type=int,
        default=256,
        help="Maximum number of processes allowed inside the solver container.",
    )
    parser.add_argument(
        "--docker-solver-tmp-size",
        default="128m",
        help="Maximum writable size of /tmp inside the solver container.",
    )
    parser.add_argument(
        "--docker-solver-workdir-size",
        default="2g",
        help="Maximum writable size of /work inside the solver container.",
    )
    parser.add_argument(
        "--docker-solver-nofile-limit",
        type=int,
        default=4096,
        help="Maximum number of open files allowed inside the solver container.",
    )
    parser.add_argument(
        "--docker-solver-max-output-bytes",
        type=int,
        default=1000000,
        help="Maximum combined stdout or stderr bytes allowed from the solver command before it is killed.",
    )
    parser.add_argument(
        "--docker-solver-user",
        help="Optional user to run the solver container as.",
    )
    parser.add_argument(
        "--docker-solver-keep-caps",
        action="store_true",
        help="Do not drop Linux capabilities in the solver container.",
    )
    parser.add_argument(
        "--docker-solver-allow-privilege-escalation",
        action="store_true",
        help="Do not set no-new-privileges on the solver container.",
    )
    parser.add_argument(
        "--docker-solver-writeable-rootfs",
        action="store_true",
        help="Do not force the solver container root filesystem to read-only mode.",
    )
    parser.add_argument(
        "--docker-solver-no-cache",
        action="store_true",
        help="Build the solver Docker image with --no-cache.",
    )


def _load_dotenv() -> None:
    dotenv_path = Path(__file__).resolve().parents[1] / ".env"
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue

        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]

        os.environ.setdefault(key, value)


if __name__ == "__main__":
    main()
