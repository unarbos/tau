from __future__ import annotations

from typing import Any


def prompt_text(*records: dict[str, Any]) -> str | None:
    for record in records:
        issue = record.get("issue")
        if issue:
            return str(issue)
    return None


def llm_response_texts(record: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    for event in record.get("trajectory") or []:
        if not isinstance(event, dict) or event.get("type") != "llm_call":
            continue
        response = event.get("response")
        if not isinstance(response, dict):
            continue
        for choice in response.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
            content = message.get("content")
            if isinstance(content, str) and content:
                texts.append(content)
    return texts


def response_text(record: dict[str, Any]) -> str:
    patch = record.get("final_patch")
    if isinstance(patch, str) and patch.strip():
        return patch
    return "\n\n".join(llm_response_texts(record))


def reward_value(record: dict[str, Any]) -> float | int | bool | None:
    direct = record.get("judge_score")
    if direct is not None:
        return direct
    judge = record.get("judge")
    role = record.get("role")
    if isinstance(judge, dict) and role in {"king", "challenger"}:
        role_score = judge.get(f"{role}_score")
        if role_score is not None:
            return role_score
    if "success" in record:
        return bool(record.get("success"))
    return None


def rollout_training_ref(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "rollout_id": record.get("rollout_id"),
        "task_name": record.get("task_name"),
        "role": record.get("role"),
        "agent_hash": record.get("agent_hash"),
        "success": record.get("success"),
        "reward": reward_value(record),
        "response": response_text(record),
        "trajectory": record.get("trajectory") or [],
        "final_patch": record.get("final_patch") or "",
    }
