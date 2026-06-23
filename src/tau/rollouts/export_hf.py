from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from tau.rollouts.redaction import public_rollout
from tau.rollouts.store import load_task_rollouts, write_gzip_jsonl

log = logging.getLogger("swe-eval.rollouts.export-hf")

DEFAULT_HF_BATCH_SIZE = 64
DEFAULT_HF_RATE_LIMIT_COOLDOWN_SECONDS = 3600


def export_hour(dt: datetime | None = None) -> str:
    current = dt or datetime.now(tz=UTC)
    return current.astimezone(UTC).strftime("%Y-%m-%d-%H")


def task_rollout_hf_path(task_name: str, hour: str | None = None) -> str:
    return f"rollouts/{hour or export_hour()}/{task_name}.jsonl.gz"


def rollout_export_enabled(config: Any) -> bool:
    if not getattr(config, "push_rollouts_to_hf", False):
        return False
    if not getattr(config, "rollout_hf_dataset", None):
        return False
    token_env = getattr(config, "rollout_hf_token_env", None) or "HF_TOKEN"
    return bool(os.environ.get(token_env))


def rollout_export_manifest_path(root: Path) -> Path:
    return root / "hf-exported-rollouts.json"


def rollout_export_cooldown_path(root: Path) -> Path:
    return root / "hf-export-cooldown.json"


def load_export_manifest(root: Path) -> dict[str, Any]:
    path = rollout_export_manifest_path(root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"tasks": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), dict):
        return {"tasks": {}}
    return {"tasks": dict(payload["tasks"])}


def write_export_manifest(root: Path, manifest: dict[str, Any]) -> None:
    path = rollout_export_manifest_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hf_rate_limit_cooldown_seconds() -> int:
    raw = os.environ.get("TAU_ROLLOUT_HF_RATE_LIMIT_COOLDOWN_SECONDS")
    if not raw:
        return DEFAULT_HF_RATE_LIMIT_COOLDOWN_SECONDS
    try:
        value = int(raw)
    except ValueError:
        log.warning("Ignoring invalid TAU_ROLLOUT_HF_RATE_LIMIT_COOLDOWN_SECONDS=%r", raw)
        return DEFAULT_HF_RATE_LIMIT_COOLDOWN_SECONDS
    return max(60, value)


def _rollout_export_cooldown_until(root: Path, now: datetime | None = None) -> datetime | None:
    path = rollout_export_cooldown_path(root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_until = payload.get("cooldown_until")
        until = datetime.fromisoformat(str(raw_until))
    except Exception:
        return None
    if until.tzinfo is None:
        until = until.replace(tzinfo=UTC)
    current = now or datetime.now(tz=UTC)
    if until <= current:
        path.unlink(missing_ok=True)
        return None
    return until


def _set_rollout_export_cooldown(root: Path, *, reason: str) -> datetime:
    until = datetime.now(tz=UTC) + timedelta(seconds=_hf_rate_limit_cooldown_seconds())
    path = rollout_export_cooldown_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "cooldown_until": until.isoformat(),
                "reason": reason,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return until


def _exception_is_hf_rate_limit(exc: BaseException) -> bool:
    seen: set[int] = set()
    stack: list[BaseException] = [exc]
    while stack:
        current = stack.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        response = getattr(current, "response", None)
        if getattr(response, "status_code", None) == 429:
            return True
        text = str(current).lower()
        if "429 too many requests" in text or "rate limit" in text:
            return True
        cause = getattr(current, "__cause__", None)
        context = getattr(current, "__context__", None)
        if isinstance(cause, BaseException):
            stack.append(cause)
        if isinstance(context, BaseException):
            stack.append(context)
    return False


def exported_task_hf_path(manifest: dict[str, Any], task_name: str) -> str | None:
    tasks = manifest.get("tasks")
    if not isinstance(tasks, dict):
        return None
    entry = tasks.get(task_name)
    if not isinstance(entry, dict):
        return None
    path = entry.get("hf_path")
    return str(path) if path else None


def uploaded_local_rollout_task_names(
    *,
    root: Path,
    active_task_names: set[str],
    manifest: dict[str, Any] | None = None,
) -> list[str]:
    current_manifest = manifest or load_export_manifest(root)
    local_task_names = local_rollout_task_names(root)
    active_names = set(active_task_names)
    return [
        task_name
        for task_name in sorted(local_task_names - active_names)
        if exported_task_hf_path(current_manifest, task_name)
    ]


def clear_uploaded_rollout_tasks(
    *,
    root: Path,
    active_task_names: set[str],
    max_dirs: int,
) -> int:
    if max_dirs <= 0:
        return 0
    task_names = uploaded_local_rollout_task_names(
        root=root,
        active_task_names=active_task_names,
    )
    count = 0
    for task_name in task_names[:max_dirs]:
        task_dir = root / "tasks" / task_name
        if not task_dir.is_dir():
            continue
        shutil.rmtree(task_dir)
        count += 1
    return count


def mark_task_rollouts_exported(
    root: Path,
    *,
    task_name: str,
    path_in_repo: str,
    exported_at: datetime | None = None,
) -> None:
    manifest = load_export_manifest(root)
    tasks = dict(manifest.get("tasks") or {})
    tasks[task_name] = {
        "task_name": task_name,
        "hf_path": path_in_repo,
        "exported_at": (exported_at or datetime.now(tz=UTC)).isoformat(),
    }
    write_export_manifest(root, {**manifest, "tasks": tasks})


def local_rollout_task_names(root: Path) -> set[str]:
    tasks_dir = root / "tasks"
    if not tasks_dir.exists():
        return set()
    return {path.name for path in tasks_dir.iterdir() if path.is_dir()}


def export_task_rollouts_to_hf(
    *,
    config: Any,
    task_name: str,
    upload_file: Any | None = None,
) -> str | None:
    if not rollout_export_enabled(config):
        return None
    root = getattr(config, "resolved_rollout_root")()
    manifest = load_export_manifest(root)
    already_exported = exported_task_hf_path(manifest, task_name)
    if already_exported:
        return already_exported

    rows = [public_rollout(row) for row in load_task_rollouts(root, task_name)]
    if not rows:
        return None

    token_env = getattr(config, "rollout_hf_token_env", None) or "HF_TOKEN"
    token = os.environ.get(token_env)
    dataset_id = getattr(config, "rollout_hf_dataset", None)
    if not token or not dataset_id:
        return None

    path_in_repo = task_rollout_hf_path(task_name)
    with tempfile.NamedTemporaryFile(suffix=".jsonl.gz", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        write_gzip_jsonl(tmp_path, rows)
        uploader = upload_file or _upload_file
        uploader(
            dataset_id=dataset_id,
            token=token,
            path_in_repo=path_in_repo,
            local_path=tmp_path,
            task_name=task_name,
        )
        mark_task_rollouts_exported(root, task_name=task_name, path_in_repo=path_in_repo)
        return path_in_repo
    finally:
        tmp_path.unlink(missing_ok=True)


def _upload_file(
    *,
    dataset_id: str,
    token: str,
    path_in_repo: str,
    local_path: Path,
    task_name: str,
) -> Any:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    return api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=path_in_repo,
        repo_id=dataset_id,
        repo_type="dataset",
        commit_message=f"Publish retired tau rollouts for {task_name}",
    )


def _upload_folder(
    *,
    dataset_id: str,
    token: str,
    folder_path: Path,
    task_count: int,
) -> Any:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    return api.upload_folder(
        folder_path=str(folder_path),
        path_in_repo="",
        repo_id=dataset_id,
        repo_type="dataset",
        commit_message=f"Publish retired tau rollouts for {task_count} task(s)",
    )


def _rollout_hf_batch_size() -> int:
    raw = os.environ.get("TAU_ROLLOUT_HF_BATCH_SIZE")
    if not raw:
        return DEFAULT_HF_BATCH_SIZE
    try:
        value = int(raw)
    except ValueError:
        log.warning("Ignoring invalid TAU_ROLLOUT_HF_BATCH_SIZE=%r", raw)
        return DEFAULT_HF_BATCH_SIZE
    return max(1, value)


def _export_retired_rollouts_to_hf_individual(
    *,
    config: Any,
    active_task_names: set[str],
    upload_file: Any,
) -> int:
    root = getattr(config, "resolved_rollout_root")()
    cooldown_until = _rollout_export_cooldown_until(root)
    if cooldown_until is not None:
        log.info("Skipping rollout HF export until %s due to prior rate limit", cooldown_until.isoformat())
        return 0
    manifest = load_export_manifest(root)
    count = 0
    pending_task_names = sorted(local_rollout_task_names(root) - set(active_task_names))
    for task_name in pending_task_names:
        if exported_task_hf_path(manifest, task_name):
            continue
        try:
            path_in_repo = export_task_rollouts_to_hf(
                config=config,
                task_name=task_name,
                upload_file=upload_file,
            )
        except Exception as exc:
            if _exception_is_hf_rate_limit(exc):
                cooldown_until = _set_rollout_export_cooldown(root, reason=str(exc))
                log.warning("HF rollout export rate-limited; pausing uploads until %s", cooldown_until.isoformat())
                break
            log.exception("Rollout export failed for %s; continuing with remaining tasks", task_name)
            continue
        if not path_in_repo:
            continue
        log.info("Exported rollout task bundle %s to %s", task_name, path_in_repo)
        count += 1
        manifest = load_export_manifest(root)
    return count


def export_retired_rollouts_to_hf(
    *,
    config: Any,
    active_task_names: set[str],
    upload_file: Any | None = None,
    upload_folder: Any | None = None,
) -> int:
    if not rollout_export_enabled(config):
        return 0
    # Tests and a few internal callers inject upload_file to inspect individual
    # gzip payloads. Keep that legacy behavior, but batch production uploads.
    if upload_file is not None and upload_folder is None:
        return _export_retired_rollouts_to_hf_individual(
            config=config,
            active_task_names=active_task_names,
            upload_file=upload_file,
        )

    root = getattr(config, "resolved_rollout_root")()
    cooldown_until = _rollout_export_cooldown_until(root)
    if cooldown_until is not None:
        log.info("Skipping rollout HF export until %s due to prior rate limit", cooldown_until.isoformat())
        return 0
    manifest = load_export_manifest(root)
    token_env = getattr(config, "rollout_hf_token_env", None) or "HF_TOKEN"
    token = os.environ.get(token_env)
    dataset_id = getattr(config, "rollout_hf_dataset", None)
    if not token or not dataset_id:
        return 0

    count = 0
    pending_task_names = sorted(local_rollout_task_names(root) - set(active_task_names))
    batch_size = _rollout_hf_batch_size()
    batch: list[tuple[str, str, list[dict[str, Any]]]] = []

    def flush_batch() -> int:
        if not batch:
            return 0
        uploader = upload_folder or _upload_folder
        with tempfile.TemporaryDirectory() as tmp:
            tmp_root = Path(tmp)
            for task_name, path_in_repo, rows in batch:
                local_path = tmp_root / path_in_repo
                local_path.parent.mkdir(parents=True, exist_ok=True)
                write_gzip_jsonl(local_path, rows)
            uploader(
                dataset_id=dataset_id,
                token=token,
                folder_path=tmp_root,
                task_count=len(batch),
            )
        exported_at = datetime.now(tz=UTC)
        for task_name, path_in_repo, _rows in batch:
            mark_task_rollouts_exported(
                root,
                task_name=task_name,
                path_in_repo=path_in_repo,
                exported_at=exported_at,
            )
            log.info("Exported rollout task bundle %s to %s", task_name, path_in_repo)
        exported = len(batch)
        batch.clear()
        return exported

    for task_name in pending_task_names:
        if exported_task_hf_path(manifest, task_name):
            continue
        try:
            rows = [public_rollout(row) for row in load_task_rollouts(root, task_name)]
        except Exception:
            log.exception("Rollout export staging failed for %s; continuing with remaining tasks", task_name)
            continue
        if not rows:
            continue
        batch.append((task_name, task_rollout_hf_path(task_name), rows))
        if len(batch) < batch_size:
            continue
        try:
            count += flush_batch()
            manifest = load_export_manifest(root)
        except Exception as exc:
            if _exception_is_hf_rate_limit(exc):
                cooldown_until = _set_rollout_export_cooldown(root, reason=str(exc))
                log.warning("HF rollout export rate-limited; pausing uploads until %s", cooldown_until.isoformat())
                batch.clear()
                break
            log.exception(
                "Rollout batch export failed for %d task(s); continuing with remaining tasks",
                len(batch),
            )
            batch.clear()

    try:
        count += flush_batch()
    except Exception as exc:
        if _exception_is_hf_rate_limit(exc):
            cooldown_until = _set_rollout_export_cooldown(root, reason=str(exc))
            log.warning("HF rollout export rate-limited; pausing uploads until %s", cooldown_until.isoformat())
            return count
        log.exception("Rollout batch export failed for %d task(s)", len(batch))
    return count
