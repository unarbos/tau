from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path

from config import RunConfig
from openrouter_proxy import SolveUsageSummary
from task_generation import GeneratedTask

from .constants import COMPLETED_EXIT_REASON


class Solver(abc.ABC):

    def __init__(self, model: str | None, timeout: int, config: RunConfig) -> None:
        self.model = model
        self.timeout = timeout
        self.config = config


    @abc.abstractmethod
    def solve(self, request: SolveRequest) -> SolveResult: ...



@dataclass(slots=True)
class SolveRequest:
    repo_dir: Path
    task: GeneratedTask
    task_name: str | None = None
    solution_name: str | None = None
    repo_full_name: str | None = None
    commit_sha: str | None = None
    run_label: str | None = None



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
    rollout_id: str | None = None
    rollout_path: str | None = None

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
            "rollout_id": self.rollout_id,
            "rollout_path": self.rollout_path,
        }


