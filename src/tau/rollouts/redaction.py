from __future__ import annotations

from typing import Any

_REDACTED = "[redacted]"


def redact_text(text: str | None, secrets: tuple[str, ...]) -> str | None:
    if text is None:
        return None
    redacted = text
    for secret in sorted((item for item in secrets if len(item) >= 8), key=len, reverse=True):
        redacted = redacted.replace(secret, _REDACTED)
    return redacted


def redact_value(value: Any, secrets: tuple[str, ...]) -> Any:
    if isinstance(value, str):
        return redact_text(value, secrets)
    if isinstance(value, list):
        return [redact_value(item, secrets) for item in value]
    if isinstance(value, dict):
        return {
            str(key): redact_value(item, secrets)
            for key, item in value.items()
            if str(key).lower() not in {"authorization", "x-api-key", "api_key", "api_key_header"}
        }
    return value


def public_rollout(record: dict[str, Any]) -> dict[str, Any]:
    """Return a public rollout row.

    Rollouts are public only after task-set retirement. Once eligible, the full
    trajectory is intentionally preserved for training. Validator-internal auth
    material remains stripped.
    """
    return redact_value(record, ())
