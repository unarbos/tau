from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from openrouter_proxy import OpenRouterProxy, SolveBudget, SolveUsageSummary

log = logging.getLogger("swe-eval.claw_runner")


@dataclass(slots=True)
class ClawRunResult:
    returncode: int
    stdout: str
    stderr: str
    elapsed_seconds: float
    usage_summary: SolveUsageSummary | None = None
    budget_exceeded_reason: str | None = None
    timed_out: bool = False

    @property
    def combined_output(self) -> str:
        return ((self.stdout or "") + (self.stderr or "")).strip()


def run_claw(
    *,
    prompt: str,
    cwd: Path,
    model: str | None,
    timeout: int,
    output_format: str = "text",
    openrouter_api_key: str | None = None,
    solve_budget: SolveBudget | None = None,
    cache_dir: Path | None = None,
    cache_replay_only: bool = False,
    additional_dirs: list[Path] | None = None,
) -> ClawRunResult:
    cmd = [
        "claw",
        "--dangerously-skip-permissions",
        "--output-format",
        output_format,
    ]
    if model:
        cmd.extend(["--model", model])

    allowed_dirs = _dedupe_paths(additional_dirs or [])
    for directory in allowed_dirs:
        cmd.extend(["--add-dir", str(directory)])

    cmd.extend(["-p", prompt])

    log.debug(
        "Running claw in %s (model=%s, timeout=%ss, format=%s)",
        cwd,
        model,
        timeout,
        output_format,
    )
    start = time.monotonic()
    env = os.environ.copy()
    proxy: OpenRouterProxy | None = None
    returncode = 1
    stdout = ""
    stderr = ""
    timed_out = False

    if openrouter_api_key:
        proxy = OpenRouterProxy(
            openrouter_api_key=openrouter_api_key,
            solve_budget=solve_budget,
            cache_dir=cache_dir,
            cache_replay_only=cache_replay_only,
        )
        proxy.start()
        env["OPENAI_BASE_URL"] = f"http://127.0.0.1:{proxy.port}/v1"
        env["OPENAI_API_KEY"] = proxy.auth_token
        env.pop("ANTHROPIC_API_KEY", None)
        env.pop("ANTHROPIC_BASE_URL", None)
        env.pop("ANTHROPIC_AUTH_TOKEN", None)

    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
        returncode = result.returncode
        stdout = result.stdout or ""
        stderr = result.stderr or ""
    except FileNotFoundError as exc:
        raise RuntimeError("`claw` CLI not found on PATH") from exc
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        returncode = 124
        stdout = _coerce_process_output(exc.stdout)
        timeout_message = f"Claw timed out after {timeout}s"
        existing_stderr = _coerce_process_output(exc.stderr)
        stderr = f"{existing_stderr}\n{timeout_message}".strip()
    finally:
        elapsed = time.monotonic() - start
        usage_summary = proxy.usage_snapshot() if proxy is not None else None
        budget_exceeded_reason = proxy.budget_exceeded_reason if proxy is not None else None
        if proxy is not None:
            proxy.stop()

    log.debug("Claw exited with code %s in %.2fs", returncode, elapsed)
    return ClawRunResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        elapsed_seconds=elapsed,
        usage_summary=usage_summary,
        budget_exceeded_reason=budget_exceeded_reason,
        timed_out=timed_out,
    )


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(resolved)
    return unique


def _coerce_process_output(raw_output: str | bytes | None) -> str:
    if isinstance(raw_output, bytes):
        return raw_output.decode("utf-8", errors="replace")
    elif isinstance(raw_output, str):
        return raw_output
    return ""
