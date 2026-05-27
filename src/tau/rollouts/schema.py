from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from tau.rollouts.ids import event_id
from tau.rollouts.redaction import redact_value

SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def with_event_ids(rollout_id_value: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "event_id": str(event.get("event_id") or event_id(
                rollout_id_value=rollout_id_value,
                event_index=index,
                event_type=str(event.get("type") or "event"),
            )),
            "event_index": index,
            **event,
        }
        for index, event in enumerate(events)
    ]


def event_sort_key(indexed_event: tuple[int, dict[str, Any]]) -> tuple[str, int]:
    index, event = indexed_event
    timestamp = event.get("started_at") or event.get("finished_at") or event.get("created_at")
    return str(timestamp or "9999-12-31T23:59:59+00:00"), index


def order_trajectory_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [event for _, event in sorted(enumerate(events), key=event_sort_key)]


def build_llm_event(
    *,
    method: str,
    path: str,
    request_payload: Any,
    response_payload: Any,
    status_code: int | None,
    latency_ms: int,
    request_model: str | None,
    response_model: str | None,
    usage: dict[str, Any],
    cost: float | None,
    started_at: str,
    finished_at: str,
    secrets: tuple[str, ...] = (),
) -> dict[str, Any]:
    return {
        "type": "llm_call",
        "source": "tau_proxy",
        "started_at": started_at,
        "finished_at": finished_at,
        "method": method,
        "path": path,
        "status_code": status_code,
        "latency_ms": latency_ms,
        "request": redact_value(request_payload, secrets),
        "response": redact_value(response_payload, secrets),
        "usage": usage,
        "cost": cost,
        "model_requested": request_model,
        "model_effective": response_model or request_model,
    }


def build_rollout_record(
    *,
    rollout_id_value: str,
    task_name: str,
    solution_name: str,
    role: str | None,
    repo: str | None,
    commit_sha: str | None,
    issue: str,
    agent_hash: str | None,
    agent_source: dict[str, Any] | None,
    started_at: str,
    finished_at: str,
    trajectory: list[dict[str, Any]],
    final_patch: str,
    miner_logs: str | None,
    steps: int | None,
    cost: float | None,
    success: bool,
    exit_reason: str,
    runner: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "rollout_id": rollout_id_value,
        "task_name": task_name,
        "solution_name": solution_name,
        "repo": repo,
        "commit_sha": commit_sha,
        "issue": issue,
        "agent_hash": agent_hash,
        "agent_source": agent_source,
        "role": role or role_from_solution_name(solution_name),
        "started_at": started_at,
        "finished_at": finished_at,
        "runner": runner,
        "trajectory": with_event_ids(rollout_id_value, order_trajectory_events(trajectory)),
        "final_patch": final_patch,
        "miner_logs": miner_logs,
        "steps": steps,
        "cost": cost,
        "success": success,
        "exit_reason": exit_reason,
        "visibility": "public_after_task_retired",
    }


def role_from_solution_name(solution_name: str) -> str:
    if solution_name == "king":
        return "king"
    if solution_name.startswith("challenger"):
        return "challenger"
    if solution_name in {"baseline", "reference"}:
        return solution_name
    return "offline"


def attach_judge_outcome(
    record: dict[str, Any],
    *,
    duel_id: int | None,
    judge: dict[str, Any],
    pairwise: dict[str, Any] | None = None,
) -> dict[str, Any]:
    updated = {**record, "judge": judge}
    if duel_id is not None:
        updated["duel_id"] = duel_id
    if pairwise is not None:
        updated["pairwise"] = pairwise
    return updated
