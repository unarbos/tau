from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tau.io.chat_completion import normalize_message_text

DEFAULT_EMPTY_RESPONSE_RETRIES = 3
DEFAULT_RATE_LIMIT_RETRIES = 6

SOLVER_SHELL_TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run exactly one shell command in the task repository.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Single bash command to execute.",
                },
            },
            "required": ["command"],
        },
    },
}


@dataclass(frozen=True, slots=True)
class UpstreamRequestPolicy:
    """Validator-side tweaks applied to proxied chat completion requests."""

    text_only: bool = False
    shell_tools: bool = False
    empty_response_retries: int = DEFAULT_EMPTY_RESPONSE_RETRIES
    rate_limit_retries: int = DEFAULT_RATE_LIMIT_RETRIES

    def __post_init__(self) -> None:
        if self.empty_response_retries < 1:
            raise ValueError("empty_response_retries must be at least 1")
        if self.rate_limit_retries < 1:
            raise ValueError("rate_limit_retries must be at least 1")
        if self.text_only and self.shell_tools:
            raise ValueError("text_only and shell_tools are mutually exclusive")


def build_upstream_request_policy(
    *,
    text_only: bool = False,
    shell_tools: bool = False,
    empty_response_retries: int | None = None,
    rate_limit_retries: int | None = None,
) -> UpstreamRequestPolicy | None:
    if (
        not text_only
        and not shell_tools
        and empty_response_retries is None
        and rate_limit_retries is None
    ):
        return None
    return UpstreamRequestPolicy(
        text_only=text_only,
        shell_tools=shell_tools,
        empty_response_retries=(
            empty_response_retries
            if empty_response_retries is not None
            else DEFAULT_EMPTY_RESPONSE_RETRIES
        ),
        rate_limit_retries=(
            rate_limit_retries
            if rate_limit_retries is not None
            else DEFAULT_RATE_LIMIT_RETRIES
        ),
    )


def apply_upstream_request_policy(
    payload: dict[str, Any],
    policy: UpstreamRequestPolicy,
) -> None:
    if policy.shell_tools:
        payload.pop("functions", None)
        payload["tools"] = [SOLVER_SHELL_TOOL]
        payload["tool_choice"] = "auto"
        payload["parallel_tool_calls"] = False
        messages = payload.get("messages")
        if isinstance(messages, list):
            payload["messages"] = drop_empty_assistant_messages(messages)
        return
    if policy.text_only:
        payload["tool_choice"] = "none"
        payload.pop("tools", None)
        payload.pop("functions", None)
        payload["parallel_tool_calls"] = False
        messages = payload.get("messages")
        if isinstance(messages, list):
            payload["messages"] = drop_empty_assistant_messages(messages)


def drop_empty_assistant_messages(messages: list[Any]) -> list[Any]:
    cleaned: list[Any] = []
    for message in messages:
        if not isinstance(message, dict):
            cleaned.append(message)
            continue
        if message.get("role") != "assistant":
            cleaned.append(message)
            continue
        if normalize_message_text(message.get("content")).strip():
            cleaned.append(message)
            continue
        if message.get("tool_calls"):
            cleaned.append(message)
    return cleaned
