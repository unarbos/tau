from __future__ import annotations

import logging
import os
from typing import Any

import httpx

log = logging.getLogger("swe-eval.openrouter_client")

def _normalize_openrouter_base_url(raw: str | None) -> str:
    base = (raw or "https://openrouter.ai/api/v1").rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/v1"):
        return base
    return base + "/v1"


_DEFAULT_MODEL = "deepseek/deepseek-v4-flash"


def _openrouter_url() -> str:
    return _normalize_openrouter_base_url(
        os.environ.get("OPENROUTER_UPSTREAM_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL"),
    ) + "/chat/completions"


def complete_text(
    *,
    prompt: str,
    model: str | None,
    timeout: int,
    openrouter_api_key: str,
    system_prompt: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
    max_tokens: int | None = None,
    reasoning: dict[str, Any] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "model": _resolve_model(model),
        "messages": _build_messages(system_prompt=system_prompt, prompt=prompt),
    }
    if temperature is not None:
        payload["temperature"] = temperature
    if top_p is not None:
        payload["top_p"] = top_p
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if reasoning is not None:
        payload["reasoning"] = reasoning
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json",
        "X-Title": "swe-eval",
    }
    log.debug("Calling OpenRouter model=%s timeout=%ss", payload["model"], timeout)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(_openrouter_url(), headers=headers, json=payload)
        response.raise_for_status()
    data = response.json()
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError(_no_choices_error(data))
    message = choices[0].get("message") or {}
    content = message.get("content")
    text = _extract_text(content)
    if not text.strip():
        raise RuntimeError(_empty_content_error(data))
    return text


def _resolve_model(model: str | None) -> str:
    if not model:
        return _DEFAULT_MODEL
    if model.startswith("openrouter/"):
        return model.split("/", 1)[1]
    return model


def _build_messages(*, system_prompt: str | None, prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    return messages


def _extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text" and item.get("text"):
                parts.append(str(item["text"]))
        return "".join(parts)
    return ""


def _no_choices_error(data: dict[str, Any]) -> str:
    error = data.get("error") if isinstance(data.get("error"), dict) else {}
    return (
        "OpenRouter returned no choices "
        f"(error_code={error.get('code')!r}, "
        f"error_message={_truncate_error_text(error.get('message'))!r}, "
        f"response_keys={sorted(data.keys())})"
    )


def _truncate_error_text(raw: Any, limit: int = 240) -> str | None:
    if raw is None:
        return None
    text = str(raw)
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _empty_content_error(data: dict[str, Any]) -> str:
    choice = (data.get("choices") or [{}])[0]
    message = choice.get("message") if isinstance(choice, dict) else {}
    message = message if isinstance(message, dict) else {}
    usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    completion_details = (
        usage.get("completion_tokens_details")
        if isinstance(usage.get("completion_tokens_details"), dict)
        else {}
    )
    return (
        "OpenRouter returned empty content "
        f"(finish_reason={choice.get('finish_reason')!r}, "
        f"native_finish_reason={choice.get('native_finish_reason')!r}, "
        f"message_keys={sorted(message.keys())}, "
        f"completion_tokens={usage.get('completion_tokens')!r}, "
        f"reasoning_tokens={completion_details.get('reasoning_tokens')!r})"
    )
