from __future__ import annotations

import hashlib
import json
from typing import Any


def stable_id(prefix: str, payload: dict[str, Any], *, length: int = 24) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{digest}"


def rollout_id(
    *,
    task_name: str,
    solution_name: str,
    agent_hash: str | None,
    started_at: str,
) -> str:
    return stable_id(
        "rol",
        {
            "task_name": task_name,
            "solution_name": solution_name,
            "agent_hash": agent_hash,
            "started_at": started_at,
        },
    )


def event_id(*, rollout_id_value: str, event_index: int, event_type: str) -> str:
    return stable_id(
        "evt",
        {
            "rollout_id": rollout_id_value,
            "event_index": event_index,
            "event_type": event_type,
        },
    )
