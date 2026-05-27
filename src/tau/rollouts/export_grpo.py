from __future__ import annotations

from typing import Any

from tau.rollouts.training import prompt_text, rollout_training_ref


def grpo_row(*, task_name: str, group_id: str, rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "group_id": group_id,
        "prompt": prompt_text(*rollouts),
        "responses": [rollout_training_ref(row) for row in rollouts],
        "rollouts": rollouts,
    }
