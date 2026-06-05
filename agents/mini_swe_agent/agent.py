from __future__ import annotations

import os
import signal
import shutil
import subprocess
import tempfile
import time
from pathlib import Path


def _git_diff(repo_path: str) -> str:
    proc = subprocess.run(
        ["git", "diff", "--binary", "--", "."],
        cwd=repo_path,
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    return proc.stdout or ""


def _mini_model_name(model: str) -> str:
    if "/" in model and not model.startswith(("openai/", "azure/", "openrouter/")):
        return f"openai/{model}"
    if "/" not in model:
        return f"openai/{model}"
    return model


def _mini_env(*, api_base: str, api_key: str) -> dict[str, str]:
    return os.environ | {
        "MSWEA_CONFIGURED": "true",
        "MSWEA_API_KEY": api_key,
        "MSWEA_COST_TRACKING": "ignore_errors",
        "NO_COLOR": "1",
        "OPENAI_API_KEY": api_key,
        "OPENAI_BASE_URL": api_base,
        "OPENAI_API_BASE": api_base,
        "PIP_PROGRESS_BAR": "off",
        "TQDM_DISABLE": "1",
    }


def _cleanup_reserve_seconds(timeout_seconds: int | None) -> int:
    if timeout_seconds is None or timeout_seconds <= 0:
        return 30
    return max(20, min(90, int(timeout_seconds * 0.15)))


def _inner_timeout_seconds(*, timeout_seconds: int | None, deadline_epoch: float | None) -> int | None:
    candidates: list[int] = []
    if timeout_seconds is not None and timeout_seconds > 0:
        candidates.append(timeout_seconds)
    if deadline_epoch is not None and deadline_epoch > 0:
        candidates.append(int(deadline_epoch - time.time()))
    if not candidates:
        return None
    outer = max(1, min(candidates))
    return max(1, outer - _cleanup_reserve_seconds(outer))


def _time_budget_note(inner_timeout_seconds: int | None) -> str:
    if inner_timeout_seconds is None or inner_timeout_seconds <= 0:
        return ""
    minutes = max(1, int(inner_timeout_seconds // 60))
    return (
        f"You have at most {minutes} minutes before this wrapper stops mini-swe-agent. "
        "Prefer a small correct patch over extended exploration, and finish by submitting once a viable fix is in place.\n\n"
    )


def _mini_command(*, issue: str, model: str, api_base: str, api_key: str, output_path: Path, timeout_seconds: int | None) -> list[str]:
    return [
        "mini",
        "-m",
        _mini_model_name(model),
        "-t",
        _time_budget_note(timeout_seconds) + issue,
        "-y",
        "--exit-immediately",
        "-o",
        str(output_path),
        "-c",
        "mini.yaml",
        "-c",
        "model.model_class=litellm",
        "-c",
        f"model.model_kwargs.api_base={api_base}",
        "-c",
        f"model.model_kwargs.api_key={api_key}",
        "-c",
        "agent.cost_limit=0",
    ]


def _terminate_process_group(proc: subprocess.Popen, *, grace_seconds: int = 8) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    proc.wait()


def _run_mini(
    *,
    repo_path: str,
    issue: str,
    model: str,
    api_base: str,
    api_key: str,
    output_path: Path,
    timeout_seconds: int | None,
) -> tuple[int | None, str, str, bool]:
    proc = subprocess.Popen(
        _mini_command(
            issue=issue,
            model=model,
            api_base=api_base,
            api_key=api_key,
            output_path=output_path,
            timeout_seconds=timeout_seconds,
        ),
        cwd=repo_path,
        env=_mini_env(api_base=api_base, api_key=api_key),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=timeout_seconds)
        return proc.returncode, stdout or "", stderr or "", False
    except subprocess.TimeoutExpired:
        _terminate_process_group(proc)
        stdout, stderr = proc.communicate()
        return proc.returncode, stdout or "", stderr or "", True


def _require_mini() -> None:
    if shutil.which("mini") is None:
        raise RuntimeError("The real mini-swe-agent CLI is missing. Install the official mini-swe-agent package.")


def solve(
    repo_path: str,
    issue: str,
    model: str,
    api_base: str,
    api_key: str,
    timeout_seconds: int | None = None,
    deadline_epoch: float | None = None,
) -> dict:
    _require_mini()
    inner_timeout = _inner_timeout_seconds(
        timeout_seconds=timeout_seconds,
        deadline_epoch=deadline_epoch,
    )
    with tempfile.TemporaryDirectory(prefix="tau-mini-swe-agent-") as tmp:
        output_path = Path(tmp) / "trajectory.json"
        returncode, _stdout, _stderr, timed_out = _run_mini(
            repo_path=repo_path,
            issue=issue,
            model=model,
            api_base=api_base,
            api_key=api_key,
            output_path=output_path,
            timeout_seconds=inner_timeout,
        )
    diff = _git_diff(repo_path)
    success = bool(diff.strip())
    status = "timed out" if timed_out else "exited"
    return {
        "success": success,
        "message": (
            f"mini {status} {returncode}; "
            f"inner_timeout={inner_timeout}; diff_bytes={len(diff.encode('utf-8'))}"
        ),
        "diff": diff,
    }
