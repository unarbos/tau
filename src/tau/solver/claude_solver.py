import logging
from pathlib import Path

from config import RunConfig
from openrouter_proxy import SolveBudget
from task_generation import GeneratedTask
from tau.runners.claude_runner import run_claude
from tau.solver.base import Solver, SolveRequest, SolveResult
from tau.solver.constants import COMPLETED_EXIT_REASON, SOLVER_ERROR_EXIT_REASON
from tau.solver.utils import (
    _parse_claude_json_output,
    _resolve_exit_reason,
    build_solver_prompt,
)
from workspace import git_diff

log = logging.getLogger(__name__)


class ClaudeSolver(Solver):
    def solve(self, request: SolveRequest) -> SolveResult:
        return solve_task(
            repo_dir=request.repo_dir,
            task=request.task,
            model=self.model,
            timeout=self.timeout,
            config=self.config,
        )


def solve_task(
    *,
    repo_dir: Path,
    task: GeneratedTask,
    model: str | None,
    timeout: int,
    config: RunConfig | None = None,
) -> SolveResult:
    prompt = build_solver_prompt(task)
    log.debug("Prepared solver prompt for task %r", task.title)
    result = run_claude(
        prompt=prompt,
        cwd=repo_dir,
        model=model,
        timeout=timeout,
        output_format="text",
        openrouter_api_key=config.openrouter_api_key if config else None,
        solve_budget=SolveBudget.from_config(config),
        cache_dir=config.solver_proxy_replay_dir or config.solver_proxy_cache_dir if config else None,
        cache_replay_only=config.solver_proxy_replay_dir is not None if config else False,
    )

    raw_output, parsed_total_tokens, tool_calls = _parse_claude_json_output(result.stdout)
    if not raw_output:
        raw_output = result.combined_output
    exit_reason = _resolve_exit_reason(result)
    success = result.returncode == 0 and exit_reason == COMPLETED_EXIT_REASON
    if not raw_output.strip() and success:
        raw_output = "Solver returned empty output from Claude"
        exit_reason = SOLVER_ERROR_EXIT_REASON
        success = False
    solution_diff = git_diff(repo_dir)
    usage_summary = result.usage_summary
    log.debug(
        "Solver exited code=%s elapsed=%.2fs total_tokens=%s tool_calls=%s exit_reason=%s",
        result.returncode,
        result.elapsed_seconds,
        usage_summary.total_tokens if usage_summary else parsed_total_tokens,
        tool_calls,
        exit_reason,
    )

    return SolveResult(
        success=success,
        elapsed_seconds=result.elapsed_seconds,
        raw_output=raw_output,
        model=model,
        solution_diff=solution_diff,
        exit_reason=exit_reason,
        usage_summary=usage_summary,
        request_count=usage_summary.request_count if usage_summary else None,
        prompt_tokens=usage_summary.prompt_tokens if usage_summary else None,
        completion_tokens=usage_summary.completion_tokens if usage_summary else None,
        total_tokens=usage_summary.total_tokens if usage_summary else parsed_total_tokens,
        cached_tokens=usage_summary.cached_tokens if usage_summary else None,
        cache_write_tokens=usage_summary.cache_write_tokens if usage_summary else None,
        reasoning_tokens=usage_summary.reasoning_tokens if usage_summary else None,
        cost=usage_summary.cost if usage_summary else None,
        tool_calls=tool_calls,
    )
