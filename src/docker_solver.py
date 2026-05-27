from __future__ import annotations

import hashlib
import json
import logging
import os
import shlex
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from config import RunConfig
from openrouter_proxy import OpenRouterProxy, SolveBudget
from solver_runner import (
    COMPLETED_EXIT_REASON,
    PROVIDER_ACCOUNT_ERROR_EXIT_REASON,
    PROVIDER_ENDPOINT_ERROR_EXIT_REASON,
    SANDBOX_VIOLATION_EXIT_REASON,
    SOLVER_ERROR_EXIT_REASON,
    TIME_LIMIT_EXIT_REASON,
    SolveResult,
)
from task_generation import GeneratedTask
from tau.rollouts.ids import rollout_id as make_rollout_id
from tau.rollouts.redaction import redact_value
from tau.rollouts.schema import build_rollout_record, utc_now
from tau.rollouts.store import append_rollout
from workspace import ensure_tree_has_no_symlinks, git_diff

log = logging.getLogger("swe-eval.docker_solver")

_DOCKERFILE_TEMPLATE = """\
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \\
    bash git ca-certificates \\
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

CMD ["bash"]
"""

_CONTAINER_ROOT = "/work"
_CONTAINER_REPO_DIR = f"{_CONTAINER_ROOT}/repo"
_CONTAINER_AGENT_FILE = f"{_CONTAINER_ROOT}/agent.py"
_CONTAINER_PROMPT_FILE = f"{_CONTAINER_ROOT}/task.txt"
_CONTAINER_RUNNER_FILE = f"{_CONTAINER_ROOT}/run_single_file_agent.py"
_CONTAINER_PROXY_SOCKET_DIR = "/proxy-socket"
_CONTAINER_PROXY_SOCKET_FILE = f"{_CONTAINER_PROXY_SOCKET_DIR}/openrouter-proxy.sock"
_CONTAINER_PROXY_BRIDGE_FILE = f"{_CONTAINER_ROOT}/proxy_bridge.py"
_CONTAINER_RUNNER_EVENTS_FILE = f"{_CONTAINER_ROOT}/tau_events.jsonl"
_CONTAINER_PROXY_PORT = 4318
_DEFAULT_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"
_DEFAULT_AGENT_FILE = "agent.py"
_HARNESS_ROLLOUT_FILENAME = "harness.json"
_SHARED_DOCKER_TEMP_ROOT = Path.home() / ".cache" / "swe-eval"
_REDACTED = "[redacted]"
_DOCKER_SOLVER_HARD_TIMEOUT_SECONDS = 300


@dataclass(slots=True)
class _DockerSolverCommandResult:
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool = False
    killed_for_budget: bool = False
    sandbox_violation_reason: str | None = None
    parsed_output: str | None = None
    rollout_output: str | None = None
    session_id: str | None = None
    tool_calls: int | None = None
    reported_patch: str | None = None
    reported_success: bool | None = None
    trusted_events_output: str | None = None

    @property
    def combined_output(self) -> str:
        return ((self.stdout or "") + (self.stderr or "")).strip()


@dataclass(slots=True)
class _DockerProxyTransport:
    bind_host: str | None
    unix_socket_path: str | None
    container_network: str
    mount_socket_dir: bool
    container_host_name: str | None = None
    relay_container_name: str | None = None
    relay_network_name: str | None = None
    fixed_container_port: bool = False

    def container_base_url(self, proxy: OpenRouterProxy) -> str:
        if self.mount_socket_dir:
            return f"http://127.0.0.1:{_CONTAINER_PROXY_PORT}/v1"
        if not self.container_host_name:
            raise RuntimeError("Container proxy host name is not configured")
        if self.fixed_container_port:
            return f"http://{self.container_host_name}:{_CONTAINER_PROXY_PORT}/v1"
        return proxy.container_base_url(self.container_host_name)


def solve_task_in_docker(
    *,
    repo_dir: Path,
    task: GeneratedTask,
    model: str | None,
    timeout: int,
    config: RunConfig,
    run_label: str | None = None,
    task_name: str | None = None,
    solution_name: str | None = None,
    repo_full_name: str | None = None,
    commit_sha: str | None = None,
) -> SolveResult:
    if not config.openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not set. Load it from .env or export it before running swe-eval.")

    issue = task.prompt_text
    image_tag = _resolve_image_tag(config)
    model_id = _solver_model_id(config.solver_model)
    log.debug("Prepared Docker file solver issue for task %r", task.title)

    start = time.monotonic()
    container_id: str | None = None
    container_force_killed = False
    solver_run = _DockerSolverCommandResult(returncode=1, stdout="", stderr="")
    solution_diff = ""
    budget = SolveBudget.from_config(config)
    with tempfile.TemporaryDirectory(prefix="swe-eval-agent-src-") as agent_src_dir, tempfile.TemporaryDirectory(
        prefix="swe-eval-proxy-socket-",
        dir=_shared_docker_temp_root(),
    ) as proxy_socket_dir:
        # Widen permissions so the Docker container user can reach the socket.
        os.chmod(proxy_socket_dir, 0o755)
        agent_src_path = Path(agent_src_dir)
        agent_file = _materialize_agent_source(config=config, target_dir=agent_src_path)
        rollout_started_at = utc_now()
        agent_hash = _file_sha256(agent_file)
        rollout_id_value = make_rollout_id(
            task_name=task_name or run_label or "unknown-task",
            solution_name=solution_name or run_label or "solution",
            agent_hash=agent_hash,
            started_at=rollout_started_at,
        )
        rollout_events: list[dict[str, Any]] = []
        rollout_events_lock = threading.Lock()

        def _append_rollout_event(event: dict[str, Any]) -> None:
            with rollout_events_lock:
                rollout_events.append(event)

        proxy_transport = _resolve_proxy_transport(proxy_socket_dir=Path(proxy_socket_dir))
        with OpenRouterProxy(
            openrouter_api_key=config.openrouter_api_key,
            solve_budget=budget,
            bind_host=proxy_transport.bind_host,
            unix_socket_path=proxy_transport.unix_socket_path,
            enforced_model=model_id,
            enforced_provider=_solver_provider_preferences(config),
            require_auth=True,
            rollout_event_sink=_append_rollout_event if config.record_rollouts else None,
            rollout_capture_bodies=config.record_rollouts,
        ) as proxy:
            sensitive_values = _sensitive_values(config=config, proxy=proxy)
            try:
                if proxy_transport.relay_network_name:
                    _create_proxy_relay_network(network_name=proxy_transport.relay_network_name)
                if proxy_transport.relay_container_name:
                    _start_proxy_relay_container(proxy_transport=proxy_transport, proxy=proxy)
                _build_image(image_tag=image_tag, config=config)
                container_id = _start_container(
                    image_tag=image_tag,
                    config=config,
                    run_label=run_label,
                    proxy_transport=proxy_transport,
                    proxy_socket_dir=Path(proxy_socket_dir),
                )
                _copy_repo_to_container(repo_dir=repo_dir, container_id=container_id)
                _copy_agent_file_to_container(agent_file=agent_file, container_id=container_id)
                _copy_prompt_to_container(prompt=issue, container_id=container_id)
                _copy_harness_runner_to_container(container_id=container_id)
                solver_run = _run_solver_command(
                    container_id=container_id,
                    proxy=proxy,
                    timeout=timeout,
                    max_output_bytes=config.docker_solver_max_output_bytes,
                    use_proxy_bridge=proxy_transport.mount_socket_dir,
                    proxy_base_url=proxy_transport.container_base_url(proxy),
                    model_id=model_id,
                )
                solver_run = replace(
                    solver_run,
                    trusted_events_output=_read_runner_events_from_container(container_id=container_id),
                )
                solver_run = _redact_solver_run(solver_run, sensitive_values)
                if container_id is not None and _container_is_running(container_id):
                    symlink_violation = _find_repo_symlinks_in_container(container_id=container_id)
                    if symlink_violation:
                        solver_run.sandbox_violation_reason = (
                            f"repository contains symbolic links: {symlink_violation}"
                        )
                        solver_run.stderr = (
                            f"{solver_run.stderr}\nDocker tau solver sandbox violation: "
                            f"{solver_run.sandbox_violation_reason}"
                        ).strip()
                        _kill_container(container_id)
                        container_force_killed = True
                    else:
                        solution_diff = _collect_repo_patch_from_container(container_id=container_id)
                        if not solution_diff.strip() and solver_run.reported_patch:
                            solution_diff = solver_run.reported_patch
                        solution_diff = _redact_sensitive_text(solution_diff, sensitive_values)
                        _kill_container(container_id)
                        container_force_killed = True
                        _apply_patch_to_repo(repo_dir=repo_dir, patch_text=solution_diff)
            finally:
                if container_id is not None:
                    if not container_force_killed:
                        _kill_container(container_id)
                    _remove_container(container_id)
                if proxy_transport.relay_container_name:
                    _remove_container(proxy_transport.relay_container_name)
                if proxy_transport.relay_network_name:
                    _remove_network(proxy_transport.relay_network_name)

    elapsed = time.monotonic() - start
    if not solution_diff:
        solution_diff = git_diff(repo_dir)
    solution_diff = _redact_sensitive_text(solution_diff, sensitive_values)
    usage_summary = proxy.usage_snapshot()
    exit_reason = _resolve_exit_reason(solver_run=solver_run, proxy=proxy)
    success = solver_run.returncode == 0 and exit_reason == COMPLETED_EXIT_REASON
    rollout_path: str | None = None
    if config.record_rollouts:
        runner_events = _parse_runner_events(solver_run.trusted_events_output)
        trajectory = [*rollout_events, *runner_events]
        record = build_rollout_record(
            rollout_id_value=rollout_id_value,
            task_name=task_name or run_label or "unknown-task",
            solution_name=solution_name or run_label or "solution",
            role=None,
            repo=repo_full_name,
            commit_sha=commit_sha,
            issue=issue,
            agent_hash=agent_hash,
            agent_source=config.solver_agent_source.to_dict() if config.solver_agent_source else None,
            started_at=rollout_started_at,
            finished_at=utc_now(),
            trajectory=redact_value(trajectory, sensitive_values),
            final_patch=solution_diff or "",
            miner_logs=_redact_sensitive_text(_build_solver_raw_output(solver_run), sensitive_values),
            steps=solver_run.tool_calls,
            cost=usage_summary.cost,
            success=success,
            exit_reason=exit_reason,
            runner={
                "backend": "docker-file",
                "image": image_tag,
                "timeout_seconds": timeout,
                "container_network": proxy_transport.container_network,
            },
        )
        rollout_path = str(append_rollout(config.resolved_rollout_root(), record))

    return SolveResult(
        success=success,
        elapsed_seconds=elapsed,
        raw_output=_redact_sensitive_text(_build_solver_raw_output(solver_run), sensitive_values),
        model=model,
        solution_diff=solution_diff,
        exit_reason=exit_reason,
        usage_summary=usage_summary,
        request_count=usage_summary.request_count,
        prompt_tokens=usage_summary.prompt_tokens,
        completion_tokens=usage_summary.completion_tokens,
        total_tokens=usage_summary.total_tokens,
        cached_tokens=usage_summary.cached_tokens,
        cache_write_tokens=usage_summary.cache_write_tokens,
        reasoning_tokens=usage_summary.reasoning_tokens,
        cost=usage_summary.cost,
        tool_calls=solver_run.tool_calls,
        rollout_output=_redact_sensitive_text(solver_run.rollout_output, sensitive_values),
        rollout_format="single-file-json" if solver_run.rollout_output else None,
        rollout_filename=_HARNESS_ROLLOUT_FILENAME if solver_run.rollout_output else None,
        session_id=solver_run.session_id,
        rollout_id=rollout_id_value if config.record_rollouts else None,
        rollout_path=rollout_path,
    )


def _build_image(*, image_tag: str, config: RunConfig) -> None:
    inspect_result = _run(
        ["docker", "image", "inspect", image_tag],
        timeout=30,
        check=False,
    )
    if inspect_result.returncode == 0 and not config.docker_solver_no_cache:
        log.debug("Reusing existing Docker solver image %s", image_tag)
        return

    with tempfile.TemporaryDirectory(prefix="swe-eval-docker-build-") as build_dir:
        build_path = Path(build_dir)
        dockerfile = _DOCKERFILE_TEMPLATE
        (build_path / "Dockerfile").write_text(dockerfile)

        cmd = ["docker", "build", "-t", image_tag]
        if config.docker_solver_no_cache:
            cmd.append("--no-cache")
        cmd.append(".")
        result = _run(cmd, cwd=build_path, timeout=1800, check=False)
        if result.returncode != 0:
            output = ((result.stdout or "") + (result.stderr or "")).strip()
            raise RuntimeError(f"Docker solver image build failed: {output[-500:]}")


def _start_container(
    *,
    image_tag: str,
    config: RunConfig,
    run_label: str | None,
    proxy_transport: _DockerProxyTransport,
    proxy_socket_dir: Path,
) -> str:
    name = _container_name(image_tag, run_label=run_label)
    cmd = [
        "docker",
        "run",
        "-d",
        "--network",
        proxy_transport.container_network,
        "--memory",
        config.docker_solver_memory,
        "--memory-swap",
        config.docker_solver_memory,
        "--cpus",
        config.docker_solver_cpus,
        "--pids-limit",
        str(config.docker_solver_pids_limit),
        "--tmpfs",
        f"/tmp:exec,mode=1777,size={config.docker_solver_tmp_size}",
        "--ulimit",
        f"nofile={config.docker_solver_nofile_limit}:{config.docker_solver_nofile_limit}",
        "--name",
        name,
    ]
    if proxy_transport.mount_socket_dir:
        cmd.extend(["--mount", f"type=bind,src={proxy_socket_dir},dst={_CONTAINER_PROXY_SOCKET_DIR}"])
    if config.docker_solver_drop_caps:
        cmd.extend(["--cap-drop", "ALL"])
    if config.docker_solver_no_new_privileges:
        cmd.extend(["--security-opt", "no-new-privileges:true"])
    if config.docker_solver_read_only_rootfs:
        cmd.append("--read-only")
        cmd.extend(["--tmpfs", f"/work:exec,mode=1777,size={config.docker_solver_workdir_size}"])
    if config.docker_solver_user:
        cmd.extend(["--user", config.docker_solver_user])
    cmd.extend([image_tag, "sleep", "3600"])
    result = _run(cmd, timeout=30, check=False)
    if result.returncode != 0:
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        raise RuntimeError(f"Failed to start Docker solver container: {output[-500:]}")
    return result.stdout.strip()


def _copy_repo_to_container(*, repo_dir: Path, container_id: str) -> None:
    _run(
        ["docker", "exec", container_id, "bash", "-lc", f"rm -rf {_CONTAINER_REPO_DIR} && mkdir -p {_CONTAINER_REPO_DIR}"],
        timeout=30,
    )
    _copy_directory_to_container(source_dir=repo_dir, container_id=container_id, target_dir=_CONTAINER_REPO_DIR)
    _sanitize_repo_git_metadata_in_container(container_id=container_id, repo_dir=_CONTAINER_REPO_DIR)


def _copy_agent_file_to_container(*, agent_file: Path, container_id: str) -> None:
    _write_text_to_container(
        container_id=container_id,
        target_path=_CONTAINER_AGENT_FILE,
        content=agent_file.read_text(encoding="utf-8"),
    )


def _copy_prompt_to_container(*, prompt: str, container_id: str) -> None:
    _write_text_to_container(
        container_id=container_id,
        target_path=_CONTAINER_PROMPT_FILE,
        content=prompt.rstrip("\n") + "\n",
    )


def _copy_proxy_bridge_script(*, container_id: str) -> None:
    _write_text_to_container(
        container_id=container_id,
        target_path=_CONTAINER_PROXY_BRIDGE_FILE,
        content=_proxy_bridge_script() + "\n",
    )


def _copy_harness_runner_to_container(*, container_id: str) -> None:
    _write_text_to_container(
        container_id=container_id,
        target_path=_CONTAINER_RUNNER_FILE,
        content=_harness_runner_script() + "\n",
    )


def _copy_directory_to_container(
    *,
    source_dir: Path,
    container_id: str,
    target_dir: str,
    exclude_names: set[str] | None = None,
) -> None:
    if not source_dir.is_dir():
        raise RuntimeError(f"Directory to copy does not exist: {source_dir}")

    extract_script = textwrap.dedent(
        """\
        import sys
        import tarfile
        from pathlib import Path

        target = Path(sys.argv[1])
        target.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=sys.stdin.buffer, mode="r|*") as archive:
            archive.extractall(target)
        """,
    ).strip()

    with tempfile.NamedTemporaryFile(suffix=".tar") as tar_file:
        with tarfile.open(fileobj=tar_file, mode="w") as archive:
            archive.add(source_dir, arcname=".", filter=_tar_filter(exclude_names))
        tar_file.flush()
        tar_file.seek(0)
        result = subprocess.run(
            ["docker", "exec", "-i", container_id, "python3", "-c", extract_script, target_dir],
            stdin=tar_file,
            capture_output=True,
            text=True,
            timeout=300,
        )
    if result.returncode != 0:
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        raise RuntimeError(f"Failed to copy directory into container: {output[-500:]}")


def _sanitize_repo_git_metadata_in_container(*, container_id: str, repo_dir: str) -> None:
    _run(
        ["docker", "exec", container_id, "bash", "-lc", _git_metadata_sanitize_script(repo_dir)],
        timeout=120,
    )


def _git_metadata_sanitize_script(repo_dir: str) -> str:
    quoted_repo = shlex.quote(repo_dir)
    return textwrap.dedent(
        f"""\
        set -euo pipefail
        repo={quoted_repo}
        git_dir="$repo/.git"
        if [ ! -d "$git_dir" ]; then
            exit 0
        fi

        head_sha="$(git -C "$repo" -c safe.directory="$repo" rev-parse --verify HEAD)"
        git -C "$repo" -c safe.directory="$repo" checkout --detach "$head_sha" >/dev/null 2>&1

        rm -f \
            "$git_dir/FETCH_HEAD" \
            "$git_dir/ORIG_HEAD" \
            "$git_dir/MERGE_HEAD" \
            "$git_dir/CHERRY_PICK_HEAD" \
            "$git_dir/REBASE_HEAD" \
            "$git_dir/packed-refs" \
            "$git_dir/objects/info/alternates"
        rm -rf "$git_dir/refs" "$git_dir/logs"
        mkdir -p "$git_dir/refs"

        git -C "$repo" -c safe.directory="$repo" reflog expire --expire=now --all >/dev/null 2>&1 || true
        git -C "$repo" -c safe.directory="$repo" gc --prune=now >/dev/null 2>&1 || true
        """
    ).strip()


def _tar_filter(exclude_names: set[str] | None):
    def filter_member(tar_info: tarfile.TarInfo) -> tarfile.TarInfo | None:
        if not exclude_names:
            return tar_info
        parts = [part for part in Path(tar_info.name).parts if part not in {".", ""}]
        if any(part in exclude_names for part in parts):
            return None
        return tar_info

    return filter_member


def _write_text_to_container(*, container_id: str, target_path: str, content: str) -> None:
    parent_dir = str(Path(target_path).parent)
    quoted_parent_dir = shlex.quote(parent_dir)
    _run(
        ["docker", "exec", container_id, "bash", "-lc", f"mkdir -p {quoted_parent_dir}"],
        timeout=30,
    )
    write_script = textwrap.dedent(
        """\
        import sys
        from pathlib import Path

        target = Path(sys.argv[1])
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(sys.stdin.read())
        """,
    ).strip()
    result = subprocess.run(
        ["docker", "exec", "-i", container_id, "python3", "-c", write_script, target_path],
        input=content,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        raise RuntimeError(f"Failed to write file into container: {output[-500:]}")


def _run_solver_command(
    *,
    container_id: str,
    proxy: OpenRouterProxy,
    timeout: int,
    max_output_bytes: int,
    use_proxy_bridge: bool,
    proxy_base_url: str,
    model_id: str,
) -> _DockerSolverCommandResult:
    prompt_cmd = _build_solver_command(
        use_proxy_bridge=use_proxy_bridge,
    )
    env_cmd = [
        "docker",
        "exec",
        "-e",
        f"TAU_REPO_DIR={_CONTAINER_REPO_DIR}",
        "-e",
        f"TAU_PROMPT_FILE={_CONTAINER_PROMPT_FILE}",
        "-e",
        f"TAU_AGENT_FILE={_CONTAINER_AGENT_FILE}",
        "-e",
        f"TAU_HARNESS_RUNNER={_CONTAINER_RUNNER_FILE}",
        "-e",
        f"OPENAI_BASE_URL={proxy_base_url}",
        "-e",
        f"OPENAI_API_KEY={proxy.auth_token}",
        "-e",
        f"AGENT_API_BASE={proxy_base_url}",
        "-e",
        f"AGENT_API_KEY={proxy.auth_token}",
        "-e",
        f"NINJA_INFERENCE_BASE_URL={proxy_base_url}",
        "-e",
        f"NINJA_INFERENCE_API_KEY={proxy.auth_token}",
        "-e",
        f"NINJA_MODEL={model_id}",
        "-e",
        f"AGENT_MODEL={model_id}",
        container_id,
        "bash",
        "-lc",
        prompt_cmd,
    ]
    if use_proxy_bridge:
        _copy_proxy_bridge_script(container_id=container_id)
        # Insert proxy env vars before the container_id arg
        _cid_idx = env_cmd.index(container_id)
        env_cmd[_cid_idx:_cid_idx] = [
            "-e",
            f"TAU_PROXY_BRIDGE={_CONTAINER_PROXY_BRIDGE_FILE}",
            "-e",
            f"TAU_PROXY_SOCKET_PATH={_CONTAINER_PROXY_SOCKET_FILE}",
            "-e",
            f"TAU_PROXY_LISTEN_PORT={_CONTAINER_PROXY_PORT}",
        ]
    log.info("Docker exec command: %s", " ".join(env_cmd[:6]) + " ... " + " ".join(env_cmd[-4:]))
    log.info("Prompt cmd (first 200): %s", prompt_cmd[:200])
    start = time.monotonic()
    first_model_activity_at: float | None = None
    hard_timeout = max(timeout, _DOCKER_SOLVER_HARD_TIMEOUT_SECONDS)
    with tempfile.NamedTemporaryFile("w+", prefix="swe-eval-solver-stdout-", encoding="utf-8") as stdout_file, tempfile.NamedTemporaryFile(
        "w+",
        prefix="swe-eval-solver-stderr-",
        encoding="utf-8",
    ) as stderr_file:
        try:
            process = subprocess.Popen(
                env_cmd,
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(f"Required command not found: {env_cmd[0]}") from exc

        killed_for_budget = False
        timed_out = False
        stop_requested_at: float | None = None
        exec_terminate_sent = False
        timeout_message: str | None = None
        sandbox_violation_reason: str | None = None
        while process.poll() is None:
            now = time.monotonic()
            if first_model_activity_at is None:
                usage = proxy.usage_snapshot()
                if usage.first_token_count > 0 or usage.success_count > 0:
                    first_model_activity_at = now
            if proxy.budget_exceeded_reason and not killed_for_budget:
                killed_for_budget = True
                stop_requested_at = now
                _stop_solver_processes(container_id=container_id)
            elif not timed_out and first_model_activity_at is not None and now - first_model_activity_at > timeout:
                timed_out = True
                timeout_message = (
                    f"Docker tau solver active timeout after {timeout}s "
                    "from first model token"
                )
                stop_requested_at = now
                _stop_solver_processes(container_id=container_id)
            elif not timed_out and now - start > hard_timeout:
                timed_out = True
                timeout_message = f"Docker tau solver hard timeout after {hard_timeout}s wall-clock"
                stop_requested_at = now
                _stop_solver_processes(container_id=container_id)
            elif stop_requested_at is not None and not exec_terminate_sent and now - stop_requested_at > 3:
                exec_terminate_sent = True
                process.terminate()
            time.sleep(0.2)

        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)

        stdout = _read_limited_output(Path(stdout_file.name), max_output_bytes=max_output_bytes)
        stderr = _read_limited_output(Path(stderr_file.name), max_output_bytes=max_output_bytes)

        if timed_out:
            stderr = f"{stderr}\n{timeout_message or f'Docker tau solver timed out after {timeout}s'}".strip()
        if sandbox_violation_reason:
            stderr = f"{stderr}\nDocker tau solver sandbox violation: {sandbox_violation_reason}".strip()
        parsed_output, rollout_output, session_id, tool_calls, reported_patch, reported_success = _parse_harness_json_output(stdout)
        return _DockerSolverCommandResult(
            returncode=process.returncode or 0,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
            killed_for_budget=killed_for_budget,
            sandbox_violation_reason=sandbox_violation_reason,
            parsed_output=parsed_output,
            rollout_output=rollout_output,
            session_id=session_id,
            tool_calls=tool_calls,
            reported_patch=reported_patch,
            reported_success=reported_success,
        )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_runner_events_from_container(*, container_id: str) -> str:
    result = _run(
        ["docker", "exec", container_id, "bash", "-lc", f"cat {_CONTAINER_RUNNER_EVENTS_FILE} 2>/dev/null || true"],
        timeout=30,
        check=False,
    )
    return result.stdout or ""


def _parse_runner_events(raw_output: str | None) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in (raw_output or "").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _remove_container(container_id: str) -> None:
    _run(["docker", "rm", "-f", container_id], timeout=30, check=False)


def _kill_container(container_id: str) -> None:
    _run(["docker", "kill", container_id], timeout=30, check=False)


def _stop_solver_processes(*, container_id: str) -> None:
    """Stop the solver exec without stopping the container.

    Keeping the container alive lets the caller collect a best-effort repo diff
    from timed-out runs before teardown.
    """
    stop_script = textwrap.dedent(
        """\
        import os
        import signal
        import time

        current = os.getpid()
        parent = os.getppid()
        targets = []
        needles = (
            b"run_single_file_agent.py",
            b"/work/agent.py",
            b"TAU_HARNESS_RUNNER",
        )
        for name in os.listdir("/proc"):
            if not name.isdigit():
                continue
            pid = int(name)
            if pid in {1, current, parent}:
                continue
            try:
                cmdline = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\\0", b" ")
            except OSError:
                continue
            if any(needle in cmdline for needle in needles):
                targets.append(pid)

        for sig in (signal.SIGTERM, signal.SIGKILL):
            for pid in list(targets):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    pass
            time.sleep(0.5)
        """,
    ).strip()
    _run(
        ["docker", "exec", container_id, "python3", "-c", stop_script],
        timeout=15,
        check=False,
    )


def _resolve_image_tag(config: RunConfig) -> str:
    if config.docker_solver_image:
        return config.docker_solver_image
    digest = hashlib.sha256()
    digest.update(_DOCKERFILE_TEMPLATE.encode("utf-8"))
    digest.update(_harness_runner_script().encode("utf-8"))
    return f"swe-eval/file-solver:{digest.hexdigest()[:12]}"


def _container_name(image_tag: str, *, run_label: str | None) -> str:
    digest = hashlib.sha256(image_tag.encode("utf-8")).hexdigest()[:12]
    label = hashlib.sha256((run_label or str(time.time_ns())).encode("utf-8")).hexdigest()[:10]
    return f"swe-eval-tau-{digest}-{label}"


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
    timeout: int,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    log.debug("Running command: %s", " ".join(cmd))
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required command not found: {cmd[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Command timed out after {timeout}s: {' '.join(cmd[:3])}") from exc

    if check and result.returncode != 0:
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        raise RuntimeError(f"Command failed ({' '.join(cmd[:3])}): {output[-500:]}")
    return result


def _build_solver_command(*, use_proxy_bridge: bool) -> str:
    prefix = "true"
    if use_proxy_bridge:
        proxy_ready_check = shlex.quote(
            'import os, socket; '
            'sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM); '
            'sock.settimeout(0.1); '
            'sock.connect(("127.0.0.1", int(os.environ["TAU_PROXY_LISTEN_PORT"]))); '
            'sock.close()',
        )
        # Use semicolon before '&' so the PATH export stays in the main shell.
        # 'cmd1 && cmd2 &' backgrounds both; 'cmd1; cmd2 &' only backgrounds cmd2.
        prefix = (
            'python3 "$TAU_PROXY_BRIDGE" & BRIDGE_PID=$!'
            + " && trap 'kill $BRIDGE_PID >/dev/null 2>&1 || true' EXIT"
            + ' && for _ in $(seq 1 50); do '
            + f'python3 -c {proxy_ready_check} && break || sleep 0.1; '
            + 'done'
        )
    return " && ".join([prefix, _clean_harness_command()])


def _clean_harness_command() -> str:
    return (
        "env -i "
        "PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin "
        "HOME=/tmp "
        "TMPDIR=/tmp "
        "LANG=C.UTF-8 "
        "PYTHONUNBUFFERED=1 "
        'TAU_REPO_DIR="$TAU_REPO_DIR" '
        'TAU_PROMPT_FILE="$TAU_PROMPT_FILE" '
        'TAU_AGENT_FILE="$TAU_AGENT_FILE" '
        'TAU_HARNESS_RUNNER="$TAU_HARNESS_RUNNER" '
        'OPENAI_BASE_URL="$OPENAI_BASE_URL" '
        'OPENAI_API_KEY="$OPENAI_API_KEY" '
        'AGENT_API_BASE="$AGENT_API_BASE" '
        'AGENT_API_KEY="$AGENT_API_KEY" '
        'NINJA_INFERENCE_BASE_URL="$NINJA_INFERENCE_BASE_URL" '
        'NINJA_INFERENCE_API_KEY="$NINJA_INFERENCE_API_KEY" '
        'NINJA_MODEL="$NINJA_MODEL" '
        'AGENT_MODEL="$AGENT_MODEL" '
        'python3 "$TAU_HARNESS_RUNNER"'
    )


def _build_solver_raw_output(solver_run: _DockerSolverCommandResult) -> str:
    parts: list[str] = []
    if solver_run.parsed_output:
        parts.append(solver_run.parsed_output.strip())
    if solver_run.stderr:
        parts.append(solver_run.stderr.strip())
    if parts:
        return "\n\n".join(part for part in parts if part)
    return solver_run.combined_output


def _parse_harness_json_output(
    raw_output: str,
) -> tuple[str, str | None, str | None, int | None, str | None, bool | None]:
    if not raw_output.strip():
        return "", None, None, None, None, None

    payloads: list[dict[str, Any]] = []
    for line in raw_output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            payloads.append(payload)

    if not payloads:
        return raw_output.strip(), None, None, None, None, None

    payload = payloads[-1]
    result = payload.get("result") if isinstance(payload.get("result"), dict) else payload
    if not isinstance(result, dict):
        return json.dumps(payload, sort_keys=True), json.dumps(payload, sort_keys=True), None, None, None, None

    logs = str(result.get("logs") or "").strip()
    steps = _coerce_int(result.get("steps"))
    cost = result.get("cost")
    success = result.get("success")
    reported_success = bool(success) if isinstance(success, bool) else None
    reported_patch = result.get("patch") if isinstance(result.get("patch"), str) else None

    header = f"single-file harness success={success} steps={steps} cost={cost}"
    parsed_output = "\n\n".join(part for part in (header, logs) if part.strip())
    rollout_output = json.dumps(payload, sort_keys=True)
    return parsed_output, rollout_output, None, steps, reported_patch, reported_success


def _sensitive_values(*, config: RunConfig, proxy: OpenRouterProxy) -> tuple[str, ...]:
    values = {
        proxy.auth_token,
        config.openrouter_api_key or "",
    }
    return tuple(sorted((value for value in values if len(value) >= 8), key=len, reverse=True))


def _redact_sensitive_text(text: str | None, sensitive_values: tuple[str, ...]) -> str | None:
    if text is None or not sensitive_values:
        return text
    redacted = text
    for value in sensitive_values:
        redacted = redacted.replace(value, _REDACTED)
    return redacted


def _redact_solver_run(
    solver_run: _DockerSolverCommandResult,
    sensitive_values: tuple[str, ...],
) -> _DockerSolverCommandResult:
    if not sensitive_values:
        return solver_run
    return replace(
        solver_run,
        stdout=_redact_sensitive_text(solver_run.stdout, sensitive_values) or "",
        stderr=_redact_sensitive_text(solver_run.stderr, sensitive_values) or "",
        parsed_output=_redact_sensitive_text(solver_run.parsed_output, sensitive_values),
        rollout_output=_redact_sensitive_text(solver_run.rollout_output, sensitive_values),
        reported_patch=_redact_sensitive_text(solver_run.reported_patch, sensitive_values),
        trusted_events_output=_redact_sensitive_text(solver_run.trusted_events_output, sensitive_values),
    )


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _resolve_proxy_transport(*, proxy_socket_dir: Path) -> _DockerProxyTransport:
    if sys.platform.startswith("linux"):
        return _DockerProxyTransport(
            bind_host=None,
            unix_socket_path=str(proxy_socket_dir / "openrouter-proxy.sock"),
            container_network="none",
            mount_socket_dir=True,
        )

    token = hashlib.sha256(str(time.time_ns()).encode("utf-8")).hexdigest()[:12]
    relay_name = f"swe-eval-proxy-relay-{token}"
    network_name = f"swe-eval-solver-net-{token}"
    return _DockerProxyTransport(
        bind_host="0.0.0.0",
        unix_socket_path=None,
        container_network=network_name,
        mount_socket_dir=False,
        container_host_name=relay_name,
        relay_container_name=relay_name,
        relay_network_name=network_name,
        fixed_container_port=True,
    )


def _shared_docker_temp_root() -> str:
    # Keep bind-mounted socket paths under the user's home directory so Docker
    # Desktop can mount them without exposing a host TCP listener.
    _SHARED_DOCKER_TEMP_ROOT.mkdir(parents=True, exist_ok=True)
    return str(_SHARED_DOCKER_TEMP_ROOT)


def _create_proxy_relay_network(*, network_name: str) -> None:
    _run(["docker", "network", "create", "--internal", network_name], timeout=30)


def _start_proxy_relay_container(*, proxy_transport: _DockerProxyTransport, proxy: OpenRouterProxy) -> None:
    if not proxy_transport.relay_container_name or not proxy_transport.relay_network_name:
        raise RuntimeError("Proxy relay transport is missing relay container metadata")

    relay_target = f"host.docker.internal:{proxy.port}"
    _run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            proxy_transport.relay_container_name,
            "--network",
            "bridge",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--pids-limit",
            "64",
            "--memory",
            "64m",
            "--memory-swap",
            "64m",
            "--cpus",
            "0.5",
            "alpine/socat:latest",
            "-d",
            "-d",
            f"TCP-LISTEN:{_CONTAINER_PROXY_PORT},fork,reuseaddr,bind=0.0.0.0",
            f"TCP:{relay_target}",
        ],
        timeout=30,
    )
    _run(
        [
            "docker",
            "network",
            "connect",
            "--alias",
            proxy_transport.relay_container_name,
            proxy_transport.relay_network_name,
            proxy_transport.relay_container_name,
        ],
        timeout=30,
    )


def _remove_network(network_name: str) -> None:
    _run(["docker", "network", "rm", network_name], timeout=30, check=False)


def _solver_model_id(model: str | None) -> str:
    if not model:
        return _DEFAULT_OPENROUTER_MODEL
    if model.startswith("openrouter/"):
        return model.split("/", 1)[1]
    return model


def _split_provider_slugs(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _solver_provider_preferences(config: RunConfig) -> dict[str, Any] | None:
    provider: dict[str, Any] = {}
    if config.solver_provider_sort:
        provider["sort"] = config.solver_provider_sort
    only = _split_provider_slugs(config.solver_provider_only)
    if only:
        provider["only"] = only
    if config.solver_provider_allow_fallbacks is not None:
        provider["allow_fallbacks"] = config.solver_provider_allow_fallbacks
    preferred_min_throughput: dict[str, float] = {}
    if config.solver_provider_min_throughput_p50 is not None:
        preferred_min_throughput["p50"] = config.solver_provider_min_throughput_p50
    if config.solver_provider_min_throughput_p90 is not None:
        preferred_min_throughput["p90"] = config.solver_provider_min_throughput_p90
    if preferred_min_throughput:
        provider["preferred_min_throughput"] = preferred_min_throughput
    return provider or None


def _resolve_exit_reason(*, solver_run: _DockerSolverCommandResult, proxy: OpenRouterProxy) -> str:
    if solver_run.timed_out:
        return TIME_LIMIT_EXIT_REASON
    if solver_run.sandbox_violation_reason:
        return SANDBOX_VIOLATION_EXIT_REASON
    if proxy.budget_exceeded_reason:
        return proxy.budget_exceeded_reason
    if solver_run.returncode == 0:
        return COMPLETED_EXIT_REASON
    usage_summary = proxy.usage_snapshot()
    if _proxy_usage_has_provider_account_error(usage_summary):
        return PROVIDER_ACCOUNT_ERROR_EXIT_REASON
    if _proxy_usage_has_provider_endpoint_error(usage_summary):
        return PROVIDER_ENDPOINT_ERROR_EXIT_REASON
    return SOLVER_ERROR_EXIT_REASON


def _proxy_usage_has_provider_account_error(usage_summary: Any) -> bool:
    return any(_proxy_request_is_provider_account_error(request) for request in usage_summary.requests)


def _proxy_usage_has_provider_endpoint_error(usage_summary: Any) -> bool:
    return any(_proxy_request_is_provider_endpoint_error(request) for request in usage_summary.requests)


def _proxy_request_is_provider_account_error(request: Any) -> bool:
    status_code = request.status_code
    error_text = str(request.error or "").lower()
    return (
        status_code in {401, 402, 403}
        or "insufficient credit" in error_text
        or "insufficient balance" in error_text
        or "billing" in error_text
        or "payment" in error_text
        or "quota" in error_text
        or "unauthorized" in error_text
        or "invalid api key" in error_text
        or "invalid_api_key" in error_text
    )


def _proxy_request_is_provider_endpoint_error(request: Any) -> bool:
    status_code = request.status_code
    error_text = str(request.error or "").lower()
    if status_code is None:
        return bool(error_text)
    return (
        status_code == 429
        or 500 <= status_code <= 599
        or "provider returned error" in error_text
        or "no endpoints" in error_text
        or "temporarily unavailable" in error_text
    )


def _harness_runner_script() -> str:
    return textwrap.dedent(
        """\
        import importlib.util
        import json
        import os
        import subprocess
        import sys
        import tempfile
        import time
        import traceback
        from pathlib import Path

        _ORIGINAL_SUBPROCESS_RUN = subprocess.run
        _ORIGINAL_SUBPROCESS_CALL = subprocess.call
        _ORIGINAL_SUBPROCESS_CHECK_CALL = subprocess.check_call
        _ORIGINAL_SUBPROCESS_CHECK_OUTPUT = subprocess.check_output
        _ORIGINAL_SUBPROCESS_POPEN = subprocess.Popen
        _ORIGINAL_OS_SYSTEM = os.system
        _ORIGINAL_OS_POPEN = os.popen
        _TAU_SUPPRESS_POPEN_EVENT = {"value": False}
        _MAX_EVENT_TEXT_CHARS = 64000
        _MAX_EVENT_DIFF_CHARS = 256000
        _RUNNER_EVENTS_OUTPUT_PATH = Path(os.environ.get("TAU_HARNESS_RUNNER", "/work/run_single_file_agent.py")).with_name("tau_events.jsonl")
        _RUNNER_EVENTS_BUFFER = tempfile.TemporaryFile("w+", encoding="utf-8")


        def _load_agent(path):
            spec = importlib.util.spec_from_file_location("submitted_agent", str(path))
            if spec is None or spec.loader is None:
                raise RuntimeError(f"Unable to import agent file: {path}")
            module = importlib.util.module_from_spec(spec)
            sys.modules["submitted_agent"] = module
            spec.loader.exec_module(module)
            solve = getattr(module, "solve", None)
            if not callable(solve):
                raise RuntimeError("Agent file must define solve(repo_path, issue, model, api_base, api_key)")
            return module


        def _append_event(event):
            try:
                _RUNNER_EVENTS_BUFFER.write(json.dumps(event, sort_keys=True) + "\\n")
                _RUNNER_EVENTS_BUFFER.flush()
            except Exception:
                pass


        def _flush_events_to_output():
            try:
                _RUNNER_EVENTS_BUFFER.flush()
                _RUNNER_EVENTS_BUFFER.seek(0)
                payload = _RUNNER_EVENTS_BUFFER.read()
                _RUNNER_EVENTS_OUTPUT_PATH.write_text(payload, encoding="utf-8")
            except Exception:
                pass


        def _truncate_text(value, limit):
            if value is None:
                return ""
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            text = str(value)
            if len(text) <= limit:
                return text
            half = max(1, limit // 2)
            return text[:half] + f"\\n...[truncated {len(text) - limit} chars]...\\n" + text[-half:]


        def _repo_diff(repo_dir):
            previous = _TAU_SUPPRESS_POPEN_EVENT["value"]
            _TAU_SUPPRESS_POPEN_EVENT["value"] = True
            try:
                proc = _ORIGINAL_SUBPROCESS_RUN(
                    ["git", "diff", "--binary", "--", "."],
                    cwd=str(repo_dir),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                diff = proc.stdout or ""
                untracked = _ORIGINAL_SUBPROCESS_RUN(
                    ["git", "ls-files", "--others", "--exclude-standard", "-z"],
                    cwd=str(repo_dir),
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                for relative_path in [item for item in (untracked.stdout or "").split("\\0") if item]:
                    file_diff = _ORIGINAL_SUBPROCESS_RUN(
                        ["git", "diff", "--binary", "--no-index", "--", "/dev/null", relative_path],
                        cwd=str(repo_dir),
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=False,
                    )
                    if file_diff.returncode in (0, 1):
                        diff += file_diff.stdout or ""
                return diff
            finally:
                _TAU_SUPPRESS_POPEN_EVENT["value"] = previous


        def _diff_hash(text):
            import hashlib
            return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


        def _cmd_text(cmd):
            if isinstance(cmd, (list, tuple)):
                return " ".join(str(part) for part in cmd)
            return str(cmd)


        def _cwd_text(kwargs):
            return str(kwargs.get("cwd") or os.getcwd())


        def _emit_edit_if_changed(repo_dir, last_diff_hash):
            diff = _repo_diff(repo_dir)
            current_hash = _diff_hash(diff)
            if current_hash == last_diff_hash["value"]:
                return
            last_diff_hash["value"] = current_hash
            _append_event({
                "type": "edit",
                "source": "tau_runner_process_hook",
                "finished_at": _iso_now(),
                "repo_diff_sha256": current_hash,
                "diff": _truncate_text(diff, _MAX_EVENT_DIFF_CHARS),
                "truncated": len(diff) > _MAX_EVENT_DIFF_CHARS,
            })


        def _raw_text(value):
            if value is None:
                return ""
            if isinstance(value, bytes):
                return value.decode("utf-8", errors="replace")
            return str(value)


        def _output_was_truncated(value, limit):
            return len(_raw_text(value)) > limit


        def _completed_stdout(result):
            return getattr(result, "stdout", "")


        def _completed_stderr(result):
            return getattr(result, "stderr", "")


        def _write_stream_value(stream, value):
            if value in (None, "", b""):
                return
            try:
                if isinstance(value, bytes):
                    buffer = getattr(stream, "buffer", None)
                    if buffer is not None:
                        buffer.write(value)
                        buffer.flush()
                        return
                    value = value.decode("utf-8", errors="replace")
                stream.write(str(value))
                stream.flush()
            except Exception:
                pass


        def _replay_captured_output(stdout, stderr):
            _write_stream_value(sys.stdout, stdout)
            _write_stream_value(sys.stderr, stderr)


        def _run_should_capture_for_event(kwargs):
            return (
                "stdout" not in kwargs
                and "stderr" not in kwargs
                and not bool(kwargs.get("capture_output"))
            )


        class _TauPopenProxy:
            def __init__(self, process, *, emit_command, cmd, cwd, started_at, started):
                self._process = process
                self._emit_command = emit_command
                self._cmd = cmd
                self._cwd = cwd
                self._started_at = started_at
                self._started = started
                self._emitted = False

            def _emit_once(self, *, stdout="", stderr="", timed_out=False, error=None):
                if self._emitted:
                    return
                self._emitted = True
                self._emit_command(
                    cmd=self._cmd,
                    cwd=self._cwd,
                    started_at=self._started_at,
                    started=self._started,
                    result=self._process,
                    stdout=stdout,
                    stderr=stderr,
                    timed_out=timed_out,
                    error=error,
                )

            def communicate(self, *args, **kwargs):
                try:
                    stdout, stderr = self._process.communicate(*args, **kwargs)
                except subprocess.TimeoutExpired as exc:
                    self._emit_once(
                        stdout=getattr(exc, "stdout", None) or getattr(exc, "output", None) or "",
                        stderr=_raw_text(getattr(exc, "stderr", None)) + f"\\nCommand timed out after {exc.timeout}s.",
                        timed_out=True,
                        error=exc,
                    )
                    raise
                except Exception as exc:
                    self._emit_once(error=exc)
                    raise
                self._emit_once(stdout=stdout, stderr=stderr)
                return stdout, stderr

            def wait(self, *args, **kwargs):
                try:
                    returncode = self._process.wait(*args, **kwargs)
                except subprocess.TimeoutExpired as exc:
                    self._emit_once(
                        stderr=f"Command timed out after {exc.timeout}s.",
                        timed_out=True,
                        error=exc,
                    )
                    raise
                except Exception as exc:
                    self._emit_once(error=exc)
                    raise
                self._emit_once()
                return returncode

            def poll(self):
                return self._process.poll()

            def kill(self):
                return self._process.kill()

            def terminate(self):
                return self._process.terminate()

            def send_signal(self, signal):
                return self._process.send_signal(signal)

            def __enter__(self):
                self._process.__enter__()
                return self

            def __exit__(self, exc_type, exc, tb):
                result = self._process.__exit__(exc_type, exc, tb)
                if self._process.poll() is not None:
                    self._emit_once(error=exc)
                return result

            def __getattr__(self, name):
                return getattr(self._process, name)


        def _install_process_event_hooks(repo_dir):
            last_diff_hash = {"value": _diff_hash(_repo_diff(repo_dir))}

            def emit_command(*, cmd, cwd, started_at, started, result=None, returncode=None, stdout="", stderr="", timed_out=False, error=None):
                finished_at = _iso_now()
                raw_stdout = stdout if stdout not in (None, "", b"") else _completed_stdout(result)
                raw_stderr = stderr if stderr not in (None, "", b"") else _completed_stderr(result)
                event = {
                    "type": "command",
                    "source": "tau_runner_process_hook",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "cmd": _cmd_text(cmd),
                    "cwd": cwd,
                    "exit_code": returncode if returncode is not None else getattr(result, "returncode", None),
                    "stdout": _truncate_text(raw_stdout, _MAX_EVENT_TEXT_CHARS),
                    "stderr": _truncate_text(raw_stderr, _MAX_EVENT_TEXT_CHARS),
                    "duration_ms": int((time.time() - started) * 1000),
                    "timed_out": timed_out,
                    "error": str(error) if error is not None else None,
                }
                event["stdout_truncated"] = _output_was_truncated(raw_stdout, _MAX_EVENT_TEXT_CHARS)
                event["stderr_truncated"] = _output_was_truncated(raw_stderr, _MAX_EVENT_TEXT_CHARS)
                _append_event(event)
                _emit_edit_if_changed(repo_dir, last_diff_hash)

            def wrapped_run(*args, **kwargs):
                cmd = args[0] if args else kwargs.get("args")
                run_kwargs = dict(kwargs)
                capture_for_event = _run_should_capture_for_event(run_kwargs)
                if capture_for_event:
                    run_kwargs["capture_output"] = True
                started = time.time()
                started_at = _iso_now()
                cwd = _cwd_text(run_kwargs)
                previous = _TAU_SUPPRESS_POPEN_EVENT["value"]
                _TAU_SUPPRESS_POPEN_EVENT["value"] = True
                try:
                    result = _ORIGINAL_SUBPROCESS_RUN(*args, **run_kwargs)
                except subprocess.TimeoutExpired as exc:
                    if capture_for_event:
                        _replay_captured_output(exc.stdout or getattr(exc, "output", None), exc.stderr)
                    emit_command(
                        cmd=cmd,
                        cwd=cwd,
                        started_at=started_at,
                        started=started,
                        returncode=124,
                        stdout=exc.stdout or getattr(exc, "output", None) or "",
                        stderr=_raw_text(exc.stderr) + f"\\nCommand timed out after {exc.timeout}s.",
                        timed_out=True,
                        error=exc,
                    )
                    raise
                except subprocess.CalledProcessError as exc:
                    if capture_for_event:
                        _replay_captured_output(exc.output, exc.stderr)
                    emit_command(
                        cmd=cmd,
                        cwd=cwd,
                        started_at=started_at,
                        started=started,
                        returncode=exc.returncode,
                        stdout=exc.output or "",
                        stderr=exc.stderr or "",
                        error=exc,
                    )
                    raise
                except Exception as exc:
                    emit_command(cmd=cmd, cwd=cwd, started_at=started_at, started=started, returncode=None, error=exc)
                    raise
                finally:
                    _TAU_SUPPRESS_POPEN_EVENT["value"] = previous
                if capture_for_event:
                    _replay_captured_output(result.stdout, result.stderr)
                emit_command(
                    cmd=cmd,
                    cwd=cwd,
                    started_at=started_at,
                    started=started,
                    result=result,
                    stdout=result.stdout,
                    stderr=result.stderr,
                )
                if not capture_for_event:
                    return result
                return subprocess.CompletedProcess(result.args, result.returncode)

            def wrapped_call(*args, **kwargs):
                result = wrapped_run(*args, **kwargs)
                return int(getattr(result, "returncode", result))

            def wrapped_check_call(*args, **kwargs):
                result = wrapped_run(*args, **kwargs)
                returncode = int(getattr(result, "returncode", result))
                if returncode != 0:
                    raise subprocess.CalledProcessError(returncode, args[0] if args else kwargs.get("args"))
                return 0

            def wrapped_check_output(*args, **kwargs):
                kwargs = dict(kwargs)
                kwargs.setdefault("stdout", subprocess.PIPE)
                result = wrapped_run(*args, **kwargs)
                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, args[0] if args else kwargs.get("args"), output=result.stdout, stderr=result.stderr)
                return result.stdout

            def wrapped_popen(*args, **kwargs):
                if _TAU_SUPPRESS_POPEN_EVENT["value"]:
                    return _ORIGINAL_SUBPROCESS_POPEN(*args, **kwargs)
                cmd = args[0] if args else kwargs.get("args")
                started = time.time()
                started_at = _iso_now()
                cwd = _cwd_text(kwargs)
                try:
                    process = _ORIGINAL_SUBPROCESS_POPEN(*args, **kwargs)
                except Exception as exc:
                    emit_command(cmd=cmd, cwd=cwd, started_at=started_at, started=started, returncode=None, error=exc)
                    raise
                return _TauPopenProxy(process, emit_command=emit_command, cmd=cmd, cwd=cwd, started_at=started_at, started=started)

            def wrapped_os_system(command):
                started = time.time()
                started_at = _iso_now()
                returncode = _ORIGINAL_OS_SYSTEM(command)
                emit_command(cmd=command, cwd=os.getcwd(), started_at=started_at, started=started, returncode=returncode)
                return returncode

            def wrapped_os_popen(command, mode="r", buffering=-1):
                started = time.time()
                started_at = _iso_now()
                try:
                    handle = _ORIGINAL_OS_POPEN(command, mode, buffering)
                except Exception as exc:
                    emit_command(cmd=command, cwd=os.getcwd(), started_at=started_at, started=started, returncode=None, error=exc)
                    raise
                emit_command(cmd=command, cwd=os.getcwd(), started_at=started_at, started=started, returncode=None)
                return handle

            subprocess.run = wrapped_run
            subprocess.call = wrapped_call
            subprocess.check_call = wrapped_check_call
            subprocess.check_output = wrapped_check_output
            subprocess.Popen = wrapped_popen
            os.system = wrapped_os_system
            os.popen = wrapped_os_popen


        def _iso_now():
            from datetime import UTC, datetime
            return datetime.now(tz=UTC).isoformat()


        def main():
            exit_code = 1
            try:
                agent_file = Path(os.environ["TAU_AGENT_FILE"])
                repo_dir = Path(os.environ["TAU_REPO_DIR"])
                issue = Path(os.environ["TAU_PROMPT_FILE"]).read_text(encoding="utf-8")
                model = _required_env("AGENT_MODEL")
                api_base = _required_env("OPENAI_BASE_URL")
                api_key = _required_env("OPENAI_API_KEY")
                _install_process_event_hooks(repo_dir)
                module = _load_agent(agent_file)
                solve = getattr(module, "solve")
                result = solve(
                    repo_path=str(repo_dir),
                    issue=issue,
                    model=model,
                    api_base=api_base,
                    api_key=api_key,
                )
                if not isinstance(result, dict):
                    raise RuntimeError(f"solve() must return a dict, got {type(result).__name__}")
                print(json.dumps({"ok": True, "result": result}, sort_keys=True), flush=True)
                exit_code = 0 if result.get("success") else 1
            except Exception:
                print(json.dumps({"ok": False, "error": traceback.format_exc()}, sort_keys=True), flush=True)
                exit_code = 1
            finally:
                _flush_events_to_output()
            return exit_code


        def _required_env(name):
            value = os.environ.get(name)
            if not value:
                raise RuntimeError(f"{name} is required for the validator-managed inference proxy")
            return value


        if __name__ == "__main__":
            raise SystemExit(main())
        """,
    ).strip()


def _proxy_bridge_script() -> str:
    return textwrap.dedent(
        """\
        import os
        import sys
        import socket
        import threading
        import time

        LISTEN_HOST = "127.0.0.1"
        LISTEN_PORT = int(os.environ["TAU_PROXY_LISTEN_PORT"])
        SOCKET_PATH = os.environ["TAU_PROXY_SOCKET_PATH"]


        def _log(msg):
            print(f"[bridge] {msg}", file=sys.stderr, flush=True)


        def _pipe(source, destination):
            try:
                while True:
                    chunk = source.recv(65536)
                    if not chunk:
                        try:
                            destination.shutdown(socket.SHUT_WR)
                        except OSError:
                            pass
                        break
                    destination.sendall(chunk)
            except OSError:
                pass
            finally:
                try:
                    destination.close()
                except OSError:
                    pass
                try:
                    source.close()
                except OSError:
                    pass


        def _handle(client):
            upstream = None
            try:
                for attempt in range(5):
                    try:
                        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                        upstream.connect(SOCKET_PATH)
                        break
                    except OSError as e:
                        if upstream:
                            try:
                                upstream.close()
                            except OSError:
                                pass
                            upstream = None
                        if attempt < 4:
                            time.sleep(0.2 * (attempt + 1))
                        else:
                            _log(f"upstream connect failed after 5 attempts: {e}")
                            client.close()
                            return
                threading.Thread(target=_pipe, args=(client, upstream), daemon=True).start()
                threading.Thread(target=_pipe, args=(upstream, client), daemon=True).start()
            except Exception as e:
                _log(f"handle error: {e}")
                if upstream:
                    try:
                        upstream.close()
                    except OSError:
                        pass
                try:
                    client.close()
                except OSError:
                    pass


        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((LISTEN_HOST, LISTEN_PORT))
        server.listen(32)
        _log(f"listening on {LISTEN_HOST}:{LISTEN_PORT} -> {SOCKET_PATH}")

        while True:
            try:
                client, _ = server.accept()
                threading.Thread(target=_handle, args=(client,), daemon=True).start()
            except Exception as e:
                _log(f"accept error: {e}")
        """,
    ).strip()


def _read_limited_output(path: Path, *, max_output_bytes: int | None = None) -> str:
    if not path.exists():
        return ""
    raw_bytes = path.read_bytes()
    if max_output_bytes is not None and len(raw_bytes) > max_output_bytes:
        raw_bytes = raw_bytes[-max_output_bytes:]
    return raw_bytes.decode("utf-8", errors="replace")


def _container_is_running(container_id: str) -> bool:
    result = _run(
        ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
        timeout=30,
        check=False,
    )
    return result.returncode == 0 and result.stdout.strip().lower() == "true"


def _collect_repo_patch_from_container(*, container_id: str) -> str:
    patch_cmd = (
        'cd "$TAU_REPO_DIR" && '
        'git diff --binary && '
        'while IFS= read -r -d \'\' path; do '
        'git diff --binary --no-index -- /dev/null "$path" || test $? -eq 1; '
        'done < <(git ls-files --others --exclude-standard -z)'
    )
    result = _run(
        [
            "docker",
            "exec",
            "-e",
            f"TAU_REPO_DIR={_CONTAINER_REPO_DIR}",
            container_id,
            "bash",
            "-lc",
            patch_cmd,
        ],
        timeout=120,
        check=False,
    )
    if result.returncode not in (0, 1):
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        raise RuntimeError(f"Failed to collect solver patch from container: {output[-500:]}")
    return result.stdout or ""


def _find_repo_symlinks_in_container(*, container_id: str) -> str | None:
    check_script = textwrap.dedent(
        """\
        import os
        from pathlib import Path

        repo_dir = Path(os.environ["TAU_REPO_DIR"])
        symlinks = []
        for current_root, dirnames, filenames in os.walk(repo_dir, topdown=True, followlinks=False):
            current_dir = Path(current_root)
            for name in [*dirnames, *filenames]:
                candidate = current_dir / name
                if candidate.is_symlink():
                    symlinks.append(str(candidate.relative_to(repo_dir)))

        print("\\n".join(sorted(symlinks[:10])))
        """,
    ).strip()
    result = _run(
        [
            "docker",
            "exec",
            "-e",
            f"TAU_REPO_DIR={_CONTAINER_REPO_DIR}",
            container_id,
            "python3",
            "-c",
            check_script,
        ],
        timeout=120,
    )
    symlinks = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not symlinks:
        return None
    sample = ", ".join(symlinks)
    if len(symlinks) == 10:
        sample = f"{sample}, ..."
    return sample


def _apply_patch_to_repo(*, repo_dir: Path, patch_text: str) -> None:
    if not patch_text.strip():
        return
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as temp_file:
        temp_file.write(patch_text)
        temp_file.write("\n")
        temp_path = Path(temp_file.name)
    try:
        _run(
            ["git", "apply", "--binary", "--whitespace=nowarn", str(temp_path)],
            cwd=repo_dir,
            timeout=120,
        )
        ensure_tree_has_no_symlinks(repo_dir, label="solver output tree")
    finally:
        temp_path.unlink(missing_ok=True)


def _validate_agent_file(agent_file: Path) -> Path:
    if not agent_file.is_file():
        raise RuntimeError(f"Agent file does not exist: {agent_file}")
    if agent_file.suffix != ".py":
        raise RuntimeError(f"Agent file must be a Python file: {agent_file}")
    return agent_file


def _materialize_agent_source(*, config: RunConfig, target_dir: Path) -> Path:
    agent = config.solver_agent_source
    if agent is None:
        raise RuntimeError("Docker solver agent is not configured")

    if agent.kind in {"local_file", "local_path"}:
        if not agent.local_path:
            raise RuntimeError("Docker solver local agent file is missing")
        agent_file = Path(agent.local_path).expanduser().resolve()
        if agent_file.is_dir():
            agent_file = agent_file / (agent.agent_file or _DEFAULT_AGENT_FILE)
        return _validate_agent_file(agent_file)

    if agent.kind == "github_repo":
        if not agent.repo_url:
            raise RuntimeError("Docker solver GitHub repo URL is missing")
        target_dir.mkdir(parents=True, exist_ok=True)
        if agent.commit_sha:
            clone_result = _run(
                ["git", "clone", "--filter=blob:none", "--no-checkout", agent.repo_url, str(target_dir)],
                timeout=300,
                check=False,
            )
            if clone_result.returncode != 0:
                output = ((clone_result.stdout or "") + (clone_result.stderr or "")).strip()
                raise RuntimeError(f"Failed to clone agent repository: {output[-500:]}")

            # Resolve short SHA to full SHA via ls-remote, since git fetch --depth=1
            # requires a full SHA or ref name (short SHAs fail as remote refs).
            commit_ref = agent.commit_sha
            if len(commit_ref) < 40:
                ls_result = _run(
                    ["git", "ls-remote", "origin"],
                    cwd=target_dir,
                    timeout=60,
                    check=False,
                )
                if ls_result.returncode == 0 and ls_result.stdout:
                    for line in ls_result.stdout.strip().splitlines():
                        full_sha = line.split("\t")[0]
                        if full_sha.startswith(commit_ref):
                            commit_ref = full_sha
                            break

            fetch_result = _run(
                ["git", "fetch", "--depth=1", "origin", commit_ref],
                cwd=target_dir,
                timeout=180,
                check=False,
            )
            if fetch_result.returncode != 0:
                output = ((fetch_result.stdout or "") + (fetch_result.stderr or "")).strip()
                raise RuntimeError(f"Failed to fetch pinned agent commit: {output[-500:]}")

            checkout_result = _run(
                ["git", "checkout", "--detach", "FETCH_HEAD"],
                cwd=target_dir,
                timeout=120,
                check=False,
            )
            if checkout_result.returncode != 0:
                output = ((checkout_result.stdout or "") + (checkout_result.stderr or "")).strip()
                raise RuntimeError(f"Failed to checkout pinned agent commit: {output[-500:]}")

            head_result = _run(
                ["git", "rev-parse", "HEAD"],
                cwd=target_dir,
                timeout=30,
                check=False,
            )
            if head_result.returncode != 0:
                output = ((head_result.stdout or "") + (head_result.stderr or "")).strip()
                raise RuntimeError(f"Failed to verify pinned agent commit: {output[-500:]}")

            resolved_head = head_result.stdout.strip()
            if not resolved_head.startswith(agent.commit_sha):
                raise RuntimeError(
                    f"Pinned agent commit mismatch: requested {agent.commit_sha}, got {resolved_head}"
                )
        else:
            clone_result = _run(
                ["git", "clone", "--depth=1", agent.repo_url, str(target_dir)],
                timeout=300,
                check=False,
            )
            if clone_result.returncode != 0:
                output = ((clone_result.stdout or "") + (clone_result.stderr or "")).strip()
                raise RuntimeError(f"Failed to clone agent repository: {output[-500:]}")

        agent_file = target_dir / (agent.agent_file or _DEFAULT_AGENT_FILE)
        if not agent_file.is_file():
            raise RuntimeError(f"Resolved agent file does not exist in cloned repo: {agent_file}")
        return _validate_agent_file(agent_file)

    raise RuntimeError(f"Unsupported docker solver agent kind: {agent.kind}")
