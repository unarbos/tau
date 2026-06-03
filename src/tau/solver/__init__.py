from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

COMPLETED_EXIT_REASON = "completed"
TIME_LIMIT_EXIT_REASON = "time_limit_exceeded"
SANDBOX_VIOLATION_EXIT_REASON = "sandbox_violation"
SOLVER_ERROR_EXIT_REASON = "solver_error"
PROVIDER_ENDPOINT_ERROR_EXIT_REASON = "provider_endpoint_error"
PROVIDER_ACCOUNT_ERROR_EXIT_REASON = "provider_account_error"


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


@dataclass(slots=True)
class SolveUsageSummary:
    request_count: int = 0
    rejected_request_count: int = 0
    first_token_count: int = 0
    success_count: int = 0
    error_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    budget_exceeded_reason: str | None = None
    requests: list[ProxyRequestRecord] = field(default_factory=list)

    def snapshot(self) -> SolveUsageSummary:
        return SolveUsageSummary(
            request_count=self.request_count,
            rejected_request_count=self.rejected_request_count,
            first_token_count=self.first_token_count,
            success_count=self.success_count,
            error_count=self.error_count,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            cached_tokens=self.cached_tokens,
            cache_write_tokens=self.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens,
            cost=self.cost,
            budget_exceeded_reason=self.budget_exceeded_reason,
            requests=[ProxyRequestRecord(**request.to_dict()) for request in self.requests],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "rejected_request_count": self.rejected_request_count,
            "first_token_count": self.first_token_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost": self.cost,
            "budget_exceeded_reason": self.budget_exceeded_reason,
            "requests": [request.to_dict() for request in self.requests],
        }


@dataclass(slots=True)
class ProxyRequestRecord:
    method: str
    path: str
    status_code: int | None
    latency_ms: int
    request_model: str | None = None
    response_model: str | None = None
    generation_id: str | None = None
    first_token_latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost: float | None = None
    rejected: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "path": self.path,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
            "request_model": self.request_model,
            "response_model": self.response_model,
            "generation_id": self.generation_id,
            "first_token_latency_ms": self.first_token_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost": self.cost,
            "rejected": self.rejected,
            "error": self.error,
        }
