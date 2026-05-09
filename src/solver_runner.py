from __future__ import annotations

import json
import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from claude_runner import run_claude
from claw_runner import run_claw
from config import RunConfig
from openrouter_proxy import SolveBudget, SolveUsageSummary
from task_generation import GeneratedTask
from workspace import git_diff

log = logging.getLogger("swe-eval.solver_runner")
COMPLETED_EXIT_REASON = "completed"
TIME_LIMIT_EXIT_REASON = "time_limit_exceeded"
SANDBOX_VIOLATION_EXIT_REASON = "sandbox_violation"
SOLVER_ERROR_EXIT_REASON = "solver_error"


@dataclass(slots=True)
class SolveResult:
    success: bool
    elapsed_seconds: float
    raw_output: str
    model: str | None
    solution_diff: str
    exit_reason: str = COMPLETED_EXIT_REASON
    usage_summary: SolveUsageSummary | None = None
    request_count: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost: float | None = None
    tool_calls: int | None = None
    rollout_output: str | None = None
    rollout_format: str | None = None
    rollout_filename: str | None = None
    session_id: str | None = None
    error_summary: str | None = None
    error_details: dict[str, Any] | None = None

    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "elapsed_seconds": self.elapsed_seconds,
            "raw_output": self.raw_output,
            "model": self.model,
            "solution_diff": self.solution_diff,
            "exit_reason": self.exit_reason,
            "usage_summary": self.usage_summary.to_dict() if self.usage_summary else None,
            "request_count": self.request_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost": self.cost,
            "tool_calls": self.tool_calls,
            "rollout_format": self.rollout_format,
            "rollout_filename": self.rollout_filename,
            "session_id": self.session_id,
            "error_summary": self.error_summary,
            "error_details": self.error_details,
        }


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
        error_summary=_build_solver_error_summary(
            success=success,
            exit_reason=exit_reason,
            returncode=result.returncode,
            timed_out=result.timed_out,
            solution_diff=solution_diff,
            stdout=result.stdout,
            stderr=result.stderr,
            raw_output=raw_output,
            usage_summary=usage_summary,
            tool_calls=tool_calls,
        ),
        error_details=_build_solver_error_details(
            success=success,
            exit_reason=exit_reason,
            returncode=result.returncode,
            timed_out=result.timed_out,
            solution_diff=solution_diff,
            stdout=result.stdout,
            stderr=result.stderr,
            raw_output=raw_output,
            usage_summary=usage_summary,
            tool_calls=tool_calls,
        ),
    )


def solve_task_claw(
    *,
    repo_dir: Path,
    task: GeneratedTask,
    model: str | None,
    timeout: int,
    config: RunConfig | None = None,
) -> SolveResult:
    prompt = build_solver_prompt(task)
    log.debug("Prepared solver prompt for task %r (claw)", task.title)
    result = run_claw(
        prompt=prompt,
        cwd=repo_dir,
        model=model,
        timeout=timeout,
        output_format="text",
        openrouter_api_key=config.openrouter_api_key if config else None,
        solve_budget=SolveBudget.from_config(config),
    )

    raw_output, parsed_total_tokens, tool_calls = _parse_claude_json_output(result.stdout)
    if not raw_output:
        raw_output = result.combined_output
    exit_reason = _resolve_exit_reason(result)
    success = result.returncode == 0 and exit_reason == COMPLETED_EXIT_REASON
    if not raw_output.strip() and success:
        raw_output = "Solver returned empty output from Claw"
        exit_reason = SOLVER_ERROR_EXIT_REASON
        success = False
    solution_diff = git_diff(repo_dir)
    usage_summary = result.usage_summary
    log.debug(
        "Claw solver exited code=%s elapsed=%.2fs total_tokens=%s tool_calls=%s exit_reason=%s",
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
        error_summary=_build_solver_error_summary(
            success=success,
            exit_reason=exit_reason,
            returncode=result.returncode,
            timed_out=result.timed_out,
            solution_diff=solution_diff,
            stdout=result.stdout,
            stderr=result.stderr,
            raw_output=raw_output,
            usage_summary=usage_summary,
            tool_calls=tool_calls,
        ),
        error_details=_build_solver_error_details(
            success=success,
            exit_reason=exit_reason,
            returncode=result.returncode,
            timed_out=result.timed_out,
            solution_diff=solution_diff,
            stdout=result.stdout,
            stderr=result.stderr,
            raw_output=raw_output,
            usage_summary=usage_summary,
            tool_calls=tool_calls,
        ),
    )


def _build_solver_error_summary(
    *,
    success: bool,
    exit_reason: str,
    solution_diff: str,
    returncode: int | None = None,
    timed_out: bool | None = None,
    killed_for_budget: bool | None = None,
    sandbox_violation_reason: str | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    raw_output: str | None = None,
    parsed_output: str | None = None,
    rollout_output: str | None = None,
    reported_success: bool | None = None,
    reported_patch: str | None = None,
    usage_summary: SolveUsageSummary | None = None,
    tool_calls: int | None = None,
    docker_diagnostics: dict[str, Any] | None = None,
) -> str | None:
    if success and exit_reason == COMPLETED_EXIT_REASON:
        return None

    parts: list[str] = []
    if returncode is not None:
        parts.append(f"returncode={returncode}")
    if timed_out:
        parts.append("timed_out=yes")
    if killed_for_budget:
        parts.append("killed_for_budget=yes")
    if sandbox_violation_reason:
        parts.append(f"sandbox_violation={_compact_reason(sandbox_violation_reason)}")
    if rollout_output is not None:
        harness = "yes"
        if reported_success is not None:
            harness = f"yes success={str(reported_success).lower()}"
        parts.append(f"harness_json={harness}")
    else:
        parts.append("harness_json=no")
    if not solution_diff.strip():
        parts.append("patch=empty")
    else:
        parts.append(f"patch_bytes={len(solution_diff.encode('utf-8'))}")
    if reported_patch is not None:
        parts.append(f"reported_patch_bytes={len(reported_patch.encode('utf-8'))}")
    if usage_summary is not None:
        request_bits = (
            f"requests={usage_summary.request_count} "
            f"successes={usage_summary.success_count} "
            f"errors={usage_summary.error_count}"
        )
        if usage_summary.rejected_request_count:
            request_bits += f" rejected={usage_summary.rejected_request_count}"
        parts.append(request_bits)
        if usage_summary.budget_exceeded_reason:
            parts.append(f"budget={usage_summary.budget_exceeded_reason}")
    if tool_calls is not None:
        parts.append(f"tool_calls={tool_calls}")
    docker_summary = _docker_diagnostics_summary(docker_diagnostics)
    if docker_summary:
        parts.append(docker_summary)
    output_sizes = _output_size_summary(
        stdout=stdout,
        stderr=stderr,
        raw_output=raw_output,
        parsed_output=parsed_output,
    )
    if output_sizes:
        parts.append(output_sizes)
    if not parts:
        return exit_reason
    return f"{exit_reason}: " + "; ".join(parts)


def _build_solver_error_details(
    *,
    success: bool,
    exit_reason: str,
    solution_diff: str,
    returncode: int | None = None,
    timed_out: bool | None = None,
    killed_for_budget: bool | None = None,
    sandbox_violation_reason: str | None = None,
    stdout: str | None = None,
    stderr: str | None = None,
    raw_output: str | None = None,
    parsed_output: str | None = None,
    rollout_output: str | None = None,
    reported_success: bool | None = None,
    reported_patch: str | None = None,
    usage_summary: SolveUsageSummary | None = None,
    tool_calls: int | None = None,
    docker_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if success and exit_reason == COMPLETED_EXIT_REASON:
        return None

    details: dict[str, Any] = {
        "exit_reason": exit_reason,
        "failure_kind": _solver_failure_kind(
            exit_reason=exit_reason,
            returncode=returncode,
            timed_out=timed_out,
            killed_for_budget=killed_for_budget,
            sandbox_violation_reason=sandbox_violation_reason,
            rollout_output=rollout_output,
            reported_success=reported_success,
            solution_diff=solution_diff,
            raw_output=raw_output,
        ),
        "returncode": returncode,
        "timed_out": bool(timed_out),
        "killed_for_budget": bool(killed_for_budget),
        "sandbox_violation_reason": sandbox_violation_reason,
        "harness_json_found": rollout_output is not None,
        "harness_reported_success": reported_success,
        "patch_bytes": len(solution_diff.encode("utf-8")) if solution_diff else 0,
        "reported_patch_bytes": len(reported_patch.encode("utf-8")) if reported_patch is not None else None,
        "stdout_bytes": len(stdout.encode("utf-8")) if stdout is not None else None,
        "stderr_bytes": len(stderr.encode("utf-8")) if stderr is not None else None,
        "raw_output_bytes": len(raw_output.encode("utf-8")) if raw_output is not None else None,
        "parsed_output_bytes": len(parsed_output.encode("utf-8")) if parsed_output is not None else None,
        "tool_calls": tool_calls,
        "docker_diagnostics": docker_diagnostics,
    }
    if usage_summary is not None:
        details.update(
            {
                "request_count": usage_summary.request_count,
                "success_count": usage_summary.success_count,
                "error_count": usage_summary.error_count,
                "rejected_request_count": usage_summary.rejected_request_count,
                "first_token_count": usage_summary.first_token_count,
                "prompt_tokens": usage_summary.prompt_tokens,
                "completion_tokens": usage_summary.completion_tokens,
                "total_tokens": usage_summary.total_tokens,
                "reasoning_tokens": usage_summary.reasoning_tokens,
                "cost": usage_summary.cost,
                "budget_exceeded_reason": usage_summary.budget_exceeded_reason,
            }
        )
    return {key: value for key, value in details.items() if value is not None}


def _solver_failure_kind(
    *,
    exit_reason: str,
    solution_diff: str,
    returncode: int | None,
    timed_out: bool | None,
    killed_for_budget: bool | None,
    sandbox_violation_reason: str | None,
    rollout_output: str | None,
    reported_success: bool | None,
    raw_output: str | None,
) -> str:
    if timed_out or exit_reason == TIME_LIMIT_EXIT_REASON:
        return "timeout"
    if sandbox_violation_reason or exit_reason == SANDBOX_VIOLATION_EXIT_REASON:
        return "sandbox_violation"
    if killed_for_budget or exit_reason.endswith("_limit_exceeded"):
        return "budget_exceeded"
    if rollout_output is None:
        return "no_harness_json"
    if reported_success is False:
        return "harness_reported_failure"
    if not solution_diff.strip():
        return "empty_patch"
    if returncode not in (None, 0):
        return "nonzero_exit"
    if not (raw_output or "").strip():
        return "empty_output"
    return "solver_error"


def _compact_reason(reason: str, *, limit: int = 160) -> str:
    compact = " ".join(str(reason).split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3] + "..."


def _output_size_summary(
    *,
    stdout: str | None,
    stderr: str | None,
    raw_output: str | None,
    parsed_output: str | None,
) -> str:
    parts: list[str] = []
    if stdout is not None:
        parts.append(f"stdout={len(stdout.encode('utf-8'))}B")
    if stderr is not None:
        parts.append(f"stderr={len(stderr.encode('utf-8'))}B")
    if raw_output is not None and stdout is None and stderr is None:
        parts.append(f"raw_output={len(raw_output.encode('utf-8'))}B")
    if parsed_output is not None:
        parts.append(f"parsed_output={len(parsed_output.encode('utf-8'))}B")
    return " ".join(parts)


def _docker_diagnostics_summary(docker_diagnostics: dict[str, Any] | None) -> str | None:
    if not docker_diagnostics:
        return None
    state = docker_diagnostics.get("state") if isinstance(docker_diagnostics.get("state"), dict) else {}
    cgroup = docker_diagnostics.get("cgroup") if isinstance(docker_diagnostics.get("cgroup"), dict) else {}
    pieces: list[str] = []
    if state.get("status"):
        pieces.append(f"status={state['status']}")
    if state.get("oom_killed") is True:
        pieces.append("oom_killed=yes")
    if state.get("exit_code") not in (None, 0):
        pieces.append(f"container_exit={state['exit_code']}")
    pids_current = cgroup.get("pids_current")
    pids_max = cgroup.get("pids_max")
    if pids_current is not None or pids_max is not None:
        pieces.append(f"pids={pids_current if pids_current is not None else '?'}"
                      f"/{pids_max if pids_max is not None else '?'}")
    memory_events = cgroup.get("memory_events")
    if isinstance(memory_events, dict):
        oom_kill = memory_events.get("oom_kill")
        oom = memory_events.get("oom")
        if oom_kill not in (None, 0, "0"):
            pieces.append(f"oom_kill_events={oom_kill}")
        elif oom not in (None, 0, "0"):
            pieces.append(f"oom_events={oom}")
    cgroup_error = docker_diagnostics.get("cgroup_error")
    if cgroup_error:
        pieces.append(f"cgroup_error={_compact_reason(str(cgroup_error), limit=80)}")
    inspect_error = docker_diagnostics.get("inspect_error")
    if inspect_error:
        pieces.append(f"inspect_error={_compact_reason(str(inspect_error), limit=80)}")
    if not pieces:
        return None
    return "docker=" + " ".join(pieces)


def build_solver_prompt(task: GeneratedTask) -> str:
    return textwrap.dedent(
        f"""\
        You are solving a software engineering task. Your diff will be scored by
        positional line-level exact matching against a reference solution.
        Score = matched_lines / max(your_lines, reference_lines).

        Task:
        {task.prompt_text}

        Strategy:
        1. Read the files that need to change IN FULL before editing.
        2. Identify the MINIMAL set of changes — every extra line hurts your score.
        3. Make precise, targeted edits. Match existing code style exactly.
        4. Stop. Do not summarize, verify, or re-read files.

        Critical rules:
        - Change ONLY what the task requires. No cosmetic changes, no refactoring.
        - Match indentation, quotes, semicolons, naming, and spacing character-for-character.
        - Do not add comments, docstrings, type annotations, or error handling.
        - Do not reorder imports, rename variables, or fix unrelated issues.
        - Process files in alphabetical path order. Edit top-to-bottom within each file.
        - Do not run tests, builds, or linters.
        - Do not create new files unless the task explicitly requires it.
        - When unsure about a change, leave the code as-is.
        """,
    )


def _parse_claude_json_output(raw_output: str) -> tuple[str, int | None, int | None]:
    text = raw_output.strip()
    if not text:
        return "", None, None

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return text, None, None

    if not isinstance(payload, dict):
        return text, None, None

    extracted_text = _extract_text(payload).strip() or text
    token_count = _extract_token_count(payload)
    tool_calls = _count_tool_calls(payload)
    return extracted_text, token_count, tool_calls


def _extract_text(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, list):
        parts = [_extract_text(item).strip() for item in payload]
        return "\n".join(part for part in parts if part)
    if isinstance(payload, dict):
        for key in ("result", "content", "text", "message", "completion"):
            value = payload.get(key)
            if value:
                return _extract_text(value)
        if payload.get("type") == "text":
            return str(payload.get("text") or "")
        if isinstance(payload.get("content"), list):
            return _extract_text(payload["content"])
    return ""


def _extract_token_count(payload: Any) -> int | None:
    usage = _find_usage_dict(payload)
    if not usage:
        return None
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    prompt_tokens = usage.get("input_tokens")
    completion_tokens = usage.get("output_tokens")
    if isinstance(prompt_tokens, int) or isinstance(completion_tokens, int):
        return int(prompt_tokens or 0) + int(completion_tokens or 0)
    return None


def _find_usage_dict(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        usage = payload.get("usage")
        if isinstance(usage, dict):
            return usage
        for value in payload.values():
            nested = _find_usage_dict(value)
            if nested:
                return nested
    elif isinstance(payload, list):
        for item in payload:
            nested = _find_usage_dict(item)
            if nested:
                return nested
    return None


def _count_tool_calls(payload: Any) -> int | None:
    count = _count_tool_calls_inner(payload)
    return count or None


def _count_tool_calls_inner(payload: Any) -> int:
    if isinstance(payload, list):
        return sum(_count_tool_calls_inner(item) for item in payload)
    if not isinstance(payload, dict):
        return 0

    count = 0
    entry_type = payload.get("type")
    if entry_type in {"tool_call", "tool_use"}:
        count += 1
    tool_calls = payload.get("tool_calls")
    if isinstance(tool_calls, list):
        count += len(tool_calls)
    for value in payload.values():
        count += _count_tool_calls_inner(value)
    return count


def _resolve_exit_reason(result) -> str:
    if result.timed_out:
        return TIME_LIMIT_EXIT_REASON
    if result.budget_exceeded_reason:
        return result.budget_exceeded_reason
    if result.returncode == 0:
        return COMPLETED_EXIT_REASON
    return SOLVER_ERROR_EXIT_REASON
