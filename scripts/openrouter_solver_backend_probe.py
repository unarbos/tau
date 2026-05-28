from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import RunConfig  # noqa: E402
from docker_solver import _solver_model_id, _solver_provider_preferences  # noqa: E402
from openrouter_client import _normalize_openrouter_base_url  # noqa: E402
from openrouter_proxy import _MINER_CONTROLLED_SAMPLING_PARAMS, _VALIDATOR_SAMPLING_PARAMS  # noqa: E402
from workspace import build_task_paths  # noqa: E402


def openrouter_chat_url(raw_base_url: str | None) -> str:
    return _normalize_openrouter_base_url(raw_base_url) + "/chat/completions"


def load_task_prompt(*, tasks_root: Path, task_name: str) -> str:
    task_paths = build_task_paths(tasks_root, task_name)
    if not task_paths.task_txt_path.exists():
        raise FileNotFoundError(f"Task prompt is missing at {task_paths.task_txt_path}")
    return task_paths.task_txt_path.read_text().strip()


def build_probe_messages(*, task_name: str, task_prompt: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an inference backend probe. Answer briefly and do not solve the task."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Task example: {task_name}\n\n"
                f"{task_prompt}\n\n"
                "Return one sentence confirming the backend received this task prompt."
            ),
        },
    ]


def strip_miner_sampling_params(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if key not in _MINER_CONTROLLED_SAMPLING_PARAMS
    }


def build_validator_equivalent_payload(
    *,
    config: RunConfig,
    task_name: str,
    task_prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    base_payload = {
        "model": _solver_model_id(config.solver_model),
        "messages": build_probe_messages(task_name=task_name, task_prompt=task_prompt),
        "max_tokens": max_tokens,
    }
    provider = _solver_provider_preferences(config)
    provider_payload = {**base_payload, "provider": provider} if provider is not None else base_payload
    return {
        **strip_miner_sampling_params(provider_payload),
        **_VALIDATOR_SAMPLING_PARAMS,
    }


def post_openrouter_probe(
    *,
    payload: dict[str, Any],
    api_key: str,
    base_url: str | None,
    timeout: float,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "swe-eval-backend-probe",
    }
    with httpx.Client(timeout=timeout) as client:
        response = client.post(openrouter_chat_url(base_url), headers=headers, json=payload)
        response.raise_for_status()
        return response.json()


def summarize_response(payload: dict[str, Any]) -> dict[str, Any]:
    choices = payload.get("choices") if isinstance(payload.get("choices"), list) else []
    first_choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = first_choice.get("message") if isinstance(first_choice.get("message"), dict) else {}
    usage = payload.get("usage") if isinstance(payload.get("usage"), dict) else {}
    return {
        "id": payload.get("id"),
        "model": payload.get("model"),
        "finish_reason": first_choice.get("finish_reason"),
        "native_finish_reason": first_choice.get("native_finish_reason"),
        "content": message.get("content"),
        "usage": usage,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Post a single OpenRouter chat request using the same model, provider, "
            "and validator-owned sampling controls as the regular Docker solver proxy."
        ),
    )
    parser.add_argument("--task", required=True, help="Existing task name under --tasks-root.")
    parser.add_argument("--tasks-root", type=Path, default=Path("workspace/tasks"))
    parser.add_argument("--model", default=os.environ.get("SOLVER_MODEL") or os.environ.get("OPENROUTER_SOLVER_MODEL"))
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true", help="Print the request payload without posting.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(solver_model=args.model)
    task_prompt = load_task_prompt(tasks_root=args.tasks_root, task_name=args.task)
    payload = build_validator_equivalent_payload(
        config=config,
        task_name=args.task,
        task_prompt=task_prompt,
        max_tokens=args.max_tokens,
    )
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    api_key = config.openrouter_api_key
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is not set")
    response = post_openrouter_probe(
        payload=payload,
        api_key=api_key,
        base_url=os.environ.get("OPENROUTER_UPSTREAM_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL"),
        timeout=args.timeout,
    )
    print(json.dumps(summarize_response(response), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
