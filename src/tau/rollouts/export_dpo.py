from __future__ import annotations

from typing import Any

from tau.rollouts.training import prompt_text, reward_value, response_text, rollout_training_ref


def dpo_row(*, task_name: str, chosen: dict[str, Any], rejected: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "task_name": task_name,
        "prompt": prompt_text(chosen, rejected),
        "chosen_rollout_id": chosen.get("rollout_id"),
        "rejected_rollout_id": rejected.get("rollout_id"),
        "chosen": response_text(chosen),
        "rejected": response_text(rejected),
        "chosen_reward": reward_value(chosen),
        "rejected_reward": reward_value(rejected),
        "chosen_rollout": rollout_training_ref(chosen),
        "rejected_rollout": rollout_training_ref(rejected),
        "preference_source": source,
    }
