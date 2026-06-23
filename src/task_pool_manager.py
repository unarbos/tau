from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import shutil
import tempfile
import threading
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import validate as v
from config import RunConfig
from pipeline import generate_task_run, solve_task_run
from tau.rollouts.export_hf import (
    clear_uploaded_rollout_tasks,
    export_retired_rollouts_to_hf,
    export_task_rollouts_to_hf,
    rollout_export_enabled,
)
from workspace import build_solution_paths, resolve_task_paths, write_json

log = logging.getLogger("swe-eval.task-pool-manager")

_TASK_ARCHIVE_LOCK = threading.Lock()
_TASK_ARCHIVE_UPLOAD_LOCK = threading.Lock()
_SAVED_TASK_FILL_LOCK = threading.Lock()
_SAVED_TASK_FILL_IN_FLIGHT: set[str] = set()
_SAVED_TASK_FILL_IN_FLIGHT_FINGERPRINTS: dict[str, str] = {}
_POOL_FILL_ADD_LOCK = threading.Lock()
_POOL_FILLER_WORKER_OVERSUBSCRIBE = 1
# Throttle concurrent GitHub-sourced task generation independently of solve
# concurrency. Pool-filler workers each call generate_task_run() (GitHub commit
# sampling) before solving; without a cap, every worker can hammer the GitHub
# API simultaneously and trip secondary rate limits. Solve concurrency can be
# scaled high while generation stays bounded here. Tunable via env; reversible.
_POOL_GENERATION_CONCURRENCY = max(
    1, int(os.environ.get("TAU_POOL_GENERATION_CONCURRENCY", "8") or "8")
)
_POOL_GENERATION_SEMAPHORE = threading.BoundedSemaphore(_POOL_GENERATION_CONCURRENCY)
_ARCHIVE_UPLOAD_COMPLETE_STATUSES = {"uploaded_delete_pending", "uploaded_deleted"}
_ARCHIVE_UPLOAD_RETRY_STATUSES = {"pool_inserted", "upload_failed"}
_ARCHIVE_QUOTA_USED_STATUSES = {
    "pool_inserted",
    "uploaded_delete_pending",
    "uploaded_deleted",
}
_ARCHIVE_UPLOAD_BATCH_SIZE = 25


@dataclass(slots=True)
class PoolManagerPaths:
    root: Path
    state_path: Path
    pool_dir: Path
    retest_pool_dir: Path


def prepare_pool_manager_paths(config: RunConfig) -> PoolManagerPaths:
    paths = v._prepare_validate_paths(config.validate_root)
    return PoolManagerPaths(
        root=paths.root,
        state_path=paths.state_path,
        pool_dir=paths.pool_dir,
        retest_pool_dir=paths.retest_pool_dir,
    )


def require_task_archive_config(config: RunConfig) -> None:
    if not config.validate_task_archive_enabled:
        return
    if not config.validate_task_archive_hf_dataset:
        raise RuntimeError(
            "task archive is enabled but no Hugging Face dataset is configured; "
            "set --task-archive-hf-dataset or VALIDATE_TASK_ARCHIVE_HF_DATASET"
        )
    token_env = config.validate_task_archive_hf_token_env or "HF_TOKEN"
    if not os.environ.get(token_env):
        raise RuntimeError(
            f"task archive is enabled but ${token_env} is not set; "
            "set the Hugging Face token env var before starting pool-manager"
        )


def _task_archive_enabled(config: RunConfig) -> bool:
    if not config.validate_task_archive_enabled:
        return False
    if not config.validate_task_archive_hf_dataset:
        return False
    return bool(os.environ.get(config.validate_task_archive_hf_token_env or "HF_TOKEN"))


def task_archive_ledger_path(config: RunConfig) -> Path:
    return config.validate_root / "archived-tasks.json"


def load_task_archive_ledger(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return {"tasks": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), dict):
        return {"tasks": {}}
    return {"tasks": {str(key): value for key, value in payload["tasks"].items() if isinstance(value, dict)}}


def write_task_archive_ledger(path: Path, ledger: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {"tasks": dict(ledger.get("tasks") or {})})


def archived_task_names(config: RunConfig) -> set[str]:
    return set(load_task_archive_ledger(task_archive_ledger_path(config)).get("tasks") or {})


def task_content_fingerprint(task_root: Path) -> str | None:
    """Return a stable identity for the mined task content, independent of task name."""
    commit = _read_json_file(task_root / "task" / "commit.json")
    if not isinstance(commit, dict):
        return None
    repo = str(commit.get("repo_full_name") or "").strip().lower()
    commit_sha = str(commit.get("commit_sha") or commit.get("sha") or "").strip().lower()
    parent_sha = str(commit.get("parent_sha") or "").strip().lower()
    patch_path = task_root / "task" / "reference.patch"
    try:
        patch_sha = hashlib.sha256(patch_path.read_bytes()).hexdigest()
    except Exception:
        patch_sha = ""
    if not any((repo, commit_sha, parent_sha, patch_sha)):
        return None
    payload = {
        "commit_sha": commit_sha,
        "parent_sha": parent_sha,
        "patch_sha256": patch_sha,
        "repo_full_name": repo,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _task_root_fingerprints(task_roots: Iterable[Path]) -> set[str]:
    return {
        fingerprint
        for fingerprint in (task_content_fingerprint(task_root) for task_root in task_roots)
        if fingerprint
    }


def _pool_task_roots(pool: v.TaskPool) -> list[Path]:
    return [Path(task.task_root) for task in pool.list_tasks()]


def _pool_task_roots_from_disk(config: RunConfig) -> list[Path]:
    roots: list[Path] = []
    for path in config.validate_root.glob("task-pool*/*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        task_root = payload.get("task_root")
        if isinstance(task_root, str) and task_root:
            roots.append(Path(task_root))
    return roots


def archived_task_fingerprints(config: RunConfig) -> set[str]:
    tasks = load_task_archive_ledger(task_archive_ledger_path(config)).get("tasks") or {}
    return {
        str(entry.get("content_fingerprint"))
        for entry in tasks.values()
        if isinstance(entry, dict) and entry.get("content_fingerprint")
    }


def removable_archived_pool_names(config: RunConfig) -> set[str]:
    leased_names = active_duel_task_names(config)
    tasks = load_task_archive_ledger(task_archive_ledger_path(config)).get("tasks") or {}
    return {
        str(task_name)
        for task_name, entry in tasks.items()
        if (
            isinstance(entry, dict)
            and entry.get("status") == "uploaded_deleted"
            and str(task_name) not in leased_names
        )
    }


def archive_hour(dt: datetime | None = None) -> str:
    current = dt or datetime.now(tz=UTC)
    return current.astimezone(UTC).strftime("%Y-%m-%d-%H")


def safe_pool_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip() or "pool")


def task_archive_jsonl_path(pool_label: str, hour: str) -> str:
    return f"tasks/{safe_pool_label(pool_label)}/{hour}.jsonl"


def archive_quota_used(ledger: dict[str, Any], *, hour: str) -> int:
    return sum(
        1
        for item in (ledger.get("tasks") or {}).values()
        if (
            isinstance(item, dict)
            and item.get("archive_hour") == hour
            and item.get("status") in _ARCHIVE_QUOTA_USED_STATUSES
            and item.get("archive_reason") != "king_transition"
        )
    )


def archive_quota_remaining(config: RunConfig, *, pool_label: str | None = None, hour: str | None = None) -> int:
    per_hour = max(0, int(config.validate_task_archive_per_hour))
    if per_hour <= 0:
        return 0
    ledger = load_task_archive_ledger(task_archive_ledger_path(config))
    return max(0, per_hour - archive_quota_used(ledger, hour=hour or archive_hour()))


def reserve_archive_quota(
    *,
    config: RunConfig,
    task_name: str,
    pool_label: str,
    hour: str | None = None,
) -> dict[str, Any] | None:
    per_hour = max(0, int(config.validate_task_archive_per_hour))
    if per_hour <= 0:
        return None
    ledger_path = task_archive_ledger_path(config)
    hour_value = hour or archive_hour()
    with _TASK_ARCHIVE_LOCK:
        ledger = load_task_archive_ledger(ledger_path)
        if archive_quota_used(ledger, hour=hour_value) >= per_hour:
            return None
        tasks = dict(ledger.get("tasks") or {})
        updated = {
            **dict(tasks.get(task_name) or {}),
            "task_name": task_name,
            "pool_label": pool_label,
            "archive_hour": hour_value,
            "status": "archive_reserved",
            "hf_path": task_archive_jsonl_path(pool_label, hour_value),
            "updated_at": v._timestamp(),
        }
        updated.setdefault("created_at", updated["updated_at"])
        updated.pop("error", None)
        tasks[task_name] = updated
        ledger["tasks"] = tasks
        write_task_archive_ledger(ledger_path, ledger)
        return updated


def release_archive_reservation(*, config: RunConfig, task_name: str | None) -> None:
    if not task_name:
        return
    ledger_path = task_archive_ledger_path(config)
    with _TASK_ARCHIVE_LOCK:
        ledger = load_task_archive_ledger(ledger_path)
        tasks = dict(ledger.get("tasks") or {})
        entry = tasks.get(task_name)
        if isinstance(entry, dict) and entry.get("status") == "archive_reserved":
            updated = {**entry, "status": "archive_generation_skipped", "updated_at": v._timestamp()}
            tasks[task_name] = updated
            ledger["tasks"] = tasks
            write_task_archive_ledger(ledger_path, ledger)


def pending_archive_task_names(config: RunConfig) -> set[str]:
    tasks = load_task_archive_ledger(task_archive_ledger_path(config)).get("tasks") or {}
    return {
        str(task_name)
        for task_name, entry in tasks.items()
        if isinstance(entry, dict) and entry.get("status") in _ARCHIVE_UPLOAD_RETRY_STATUSES
    }


def select_rotation_archive_task(
    tasks: Sequence[v.PoolTask],
    *,
    candidate_name: str,
    leased_task_names: set[str],
    excluded_task_names: set[str] | None = None,
) -> v.PoolTask | None:
    """Pick an existing pool task to archive after a replacement is ready."""
    excluded = set(excluded_task_names or ())
    eligible = [
        task
        for task in tasks
        if task.task_name != candidate_name and task.task_name not in leased_task_names
        and task.task_name not in excluded
    ]
    if not eligible:
        return None
    return sorted(eligible, key=lambda task: (task.creation_block, task.task_name))[0]


def pool_should_prepare_task(
    *,
    config: RunConfig,
    pool: v.TaskPool,
    king: v.ValidatorSubmission | None,
    pool_label: str,
) -> tuple[bool, str, bool]:
    needs_fill, reason = v._pool_needs_fill_for_king(
        config=config,
        pool=pool,
        king=king,
        pool_label=pool_label,
    )
    if needs_fill:
        return True, reason, False
    if not _task_archive_enabled(config):
        return False, reason, False
    remaining = archive_quota_remaining(config, pool_label=pool_label)
    if remaining <= 0:
        return False, f"{reason}; hourly archive quota exhausted", False
    return True, f"{reason}; hourly archive generation quota remaining={remaining}", True


def record_task_archive_status(
    *,
    config: RunConfig,
    task_name: str,
    pool_label: str,
    status: str,
    archive_hour_value: str | None = None,
    hf_path: str | None = None,
    error: str | None = None,
    archive_reason: str | None = None,
    content_fingerprint: str | None = None,
) -> dict[str, Any]:
    ledger_path = task_archive_ledger_path(config)
    with _TASK_ARCHIVE_LOCK:
        ledger = load_task_archive_ledger(ledger_path)
        tasks = dict(ledger.get("tasks") or {})
        existing = dict(tasks.get(task_name) or {})
        hour = archive_hour_value or str(existing.get("archive_hour") or archive_hour())
        updated = {
            **existing,
            "task_name": task_name,
            "pool_label": pool_label,
            "archive_hour": hour,
            "status": status,
            "updated_at": v._timestamp(),
        }
        if archive_reason is not None:
            updated["archive_reason"] = archive_reason
        if content_fingerprint is not None:
            updated["content_fingerprint"] = content_fingerprint
        updated.setdefault("created_at", updated["updated_at"])
        if hf_path is not None:
            updated["hf_path"] = hf_path
        if error is None and status != "upload_failed":
            updated.pop("error", None)
        elif error is not None:
            updated["error"] = error
        tasks[task_name] = updated
        ledger["tasks"] = tasks
        write_task_archive_ledger(ledger_path, ledger)
        return updated


def archive_entry_upload_is_complete(entry: Any) -> bool:
    return isinstance(entry, dict) and entry.get("status") in _ARCHIVE_UPLOAD_COMPLETE_STATUSES


def archive_entry_hour(entry: Any) -> str:
    return str(entry.get("archive_hour")) if isinstance(entry, dict) and entry.get("archive_hour") else archive_hour()


def _file_artifact_record(path: Path, root: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    common = {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    try:
        return {**common, "encoding": "utf-8", "content": raw.decode("utf-8")}
    except UnicodeDecodeError:
        return {**common, "encoding": "base64", "content_base64": base64.b64encode(raw).decode("ascii")}


def task_artifact_records(task_root: Path) -> list[dict[str, Any]]:
    return [_file_artifact_record(path, task_root) for path in sorted(task_root.rglob("*")) if path.is_file()]


def _read_json_file(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def task_archive_jsonl_row(
    *,
    task: v.PoolTask,
    pool_label: str,
    archive_hour_value: str,
    king: v.ValidatorSubmission | None,
    archive_reason: str = "rotation",
) -> dict[str, Any]:
    task_root = Path(task.task_root)
    artifacts = task_artifact_records(task_root)
    artifact_hash = hashlib.sha256(
        json.dumps(artifacts, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "archived_at": v._timestamp(),
        "archive_hour": archive_hour_value,
        "pool_label": pool_label,
        "archive_reason": archive_reason,
        "task_name": task.task_name,
        "task_root_name": task_root.name,
        "pool_task": task.to_dict(),
        "content_fingerprint": task_content_fingerprint(task_root),
        "king": king.to_dict() if king is not None else None,
        "task_metadata": _read_json_file(task_root / "task" / "task.json"),
        "commit_metadata": _read_json_file(task_root / "task" / "commit.json"),
        "artifact_count": len(artifacts),
        "artifact_bundle_sha256": artifact_hash,
        "artifacts": artifacts,
    }


def hf_download_error_is_missing(exc: BaseException) -> bool:
    from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

    return isinstance(exc, EntryNotFoundError) and not isinstance(exc, LocalEntryNotFoundError)


def append_hf_dataset_jsonl(
    *,
    dataset_id: str,
    token: str,
    path_in_repo: str,
    row: dict[str, Any],
) -> Any:
    return append_hf_dataset_jsonl_rows(
        dataset_id=dataset_id,
        token=token,
        path_in_repo=path_in_repo,
        rows=[row],
        commit_message=f"Archive validator task {row['task_name']}",
    )


def append_hf_dataset_jsonl_rows(
    *,
    dataset_id: str,
    token: str,
    path_in_repo: str,
    rows: Sequence[dict[str, Any]],
    commit_message: str | None = None,
) -> Any:
    from huggingface_hub import HfApi, hf_hub_download

    rows = list(rows)
    if not rows:
        return None
    api = HfApi(token=token)
    existing = ""
    try:
        existing_path = hf_hub_download(
            repo_id=dataset_id,
            filename=path_in_repo,
            repo_type="dataset",
            token=token,
        )
        existing = Path(existing_path).read_text()
    except Exception as exc:
        if not hf_download_error_is_missing(exc):
            raise
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".jsonl", delete=False) as tmp:
        tmp.write(existing)
        if existing and not existing.endswith("\n"):
            tmp.write("\n")
        for row in rows:
            tmp.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        tmp_path = Path(tmp.name)
    try:
        return api.upload_file(
            path_or_fileobj=str(tmp_path),
            path_in_repo=path_in_repo,
            repo_id=dataset_id,
            repo_type="dataset",
            commit_message=commit_message or f"Archive {len(rows)} validator task(s)",
        )
    finally:
        tmp_path.unlink(missing_ok=True)


def archive_pool_task_to_hf_jsonl(
    *,
    config: RunConfig,
    pool: v.TaskPool,
    task: v.PoolTask,
    pool_label: str,
    king: v.ValidatorSubmission | None,
    leased_task_names: set[str],
    upload_jsonl: Any | None = None,
    archive_reason: str = "rotation",
    upload_now: bool = True,
) -> None:
    if not _task_archive_enabled(config):
        return
    token_env = config.validate_task_archive_hf_token_env or "HF_TOKEN"
    token = os.environ.get(token_env)
    dataset_id = config.validate_task_archive_hf_dataset
    if not token or not dataset_id:
        return
    upload_jsonl = upload_jsonl or append_hf_dataset_jsonl
    with _TASK_ARCHIVE_UPLOAD_LOCK:
        existing = (load_task_archive_ledger(task_archive_ledger_path(config)).get("tasks") or {}).get(task.task_name)
        if archive_entry_upload_is_complete(existing):
            pool.remove(task.task_name)
            return
        hour = archive_entry_hour(existing)
        hf_path = task_archive_jsonl_path(pool_label, hour)
        content_fingerprint = task_content_fingerprint(Path(task.task_root))
        record_task_archive_status(
            config=config,
            task_name=task.task_name,
            pool_label=pool_label,
            status="pool_inserted",
            archive_hour_value=hour,
            hf_path=hf_path,
            archive_reason=archive_reason,
            content_fingerprint=content_fingerprint,
        )
        if not upload_now:
            return
        try:
            row = task_archive_jsonl_row(
                task=task,
                pool_label=pool_label,
                archive_hour_value=hour,
                king=king,
                archive_reason=archive_reason,
            )
            upload_result = upload_jsonl(dataset_id=dataset_id, token=token, path_in_repo=hf_path, row=row)
        except Exception as exc:
            record_task_archive_status(
                config=config,
                task_name=task.task_name,
                pool_label=pool_label,
                status="upload_failed",
                archive_hour_value=hour,
                hf_path=hf_path,
                error=str(exc),
                archive_reason=archive_reason,
                content_fingerprint=content_fingerprint,
            )
            log.exception("Task archive[%s]: HF upload failed for %s", pool_label, task.task_name)
            return

        upload_url = getattr(upload_result, "commit_url", None) or str(upload_result or "")
        pool.remove(task.task_name)
        record_task_archive_status(
            config=config,
            task_name=task.task_name,
            pool_label=pool_label,
            status="uploaded_delete_pending",
            archive_hour_value=hour,
            hf_path=hf_path,
            archive_reason=archive_reason,
            content_fingerprint=content_fingerprint,
        )
        deleted = retry_pending_archived_task_deletes(config, (pool,))
        lease_note = "; active lease snapshot existed" if task.task_name in leased_task_names else ""
        log.info(
            "Task archive[%s]: uploaded %s to %s (%s); removed from pool and deleted %d archived local task(s)%s",
            pool_label,
            task.task_name,
            hf_path,
            upload_url,
            deleted,
            lease_note,
        )


def _archive_batch_error_message(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


def _upload_pending_archive_batch(
    *,
    config: RunConfig,
    pool: v.TaskPool,
    pool_label: str,
    hf_path: str,
    entries: Sequence[tuple[str, dict[str, Any]]],
    king: v.ValidatorSubmission | None,
    leased_names: set[str],
    upload_jsonl_rows: Any,
) -> int:
    token_env = config.validate_task_archive_hf_token_env or "HF_TOKEN"
    token = os.environ.get(token_env)
    dataset_id = config.validate_task_archive_hf_dataset
    if not token or not dataset_id:
        return 0

    prepared: list[tuple[str, v.PoolTask, dict[str, Any], dict[str, Any]]] = []
    for task_name, entry in entries[:_ARCHIVE_UPLOAD_BATCH_SIZE]:
        latest = load_task_archive_ledger(task_archive_ledger_path(config)).get("tasks", {}).get(task_name, {})
        if archive_entry_upload_is_complete(latest):
            pool.remove(task_name)
            continue
        if isinstance(latest, dict) and latest.get("status") not in _ARCHIVE_UPLOAD_RETRY_STATUSES:
            continue
        task = pool_task_by_name(pool, task_name)
        if task is None:
            latest = load_task_archive_ledger(task_archive_ledger_path(config)).get("tasks", {}).get(task_name, {})
            if archive_entry_upload_is_complete(latest):
                pool.remove(task_name)
                continue
            record_task_archive_status(
                config=config,
                task_name=task_name,
                pool_label=pool_label,
                status="upload_failed",
                error="cannot retry upload; task is no longer present in the pool",
            )
            continue
        archive_reason = str(entry.get("archive_reason") or "rotation")
        hour = archive_entry_hour(entry)
        row = task_archive_jsonl_row(
            task=task,
            pool_label=pool_label,
            archive_hour_value=hour,
            king=king,
            archive_reason=archive_reason,
        )
        prepared.append((task_name, task, entry, row))

    if not prepared:
        return 0

    rows = [row for _, _, _, row in prepared]
    task_names = [task_name for task_name, _, _, _ in prepared]
    try:
        upload_result = upload_jsonl_rows(
            dataset_id=dataset_id,
            token=token,
            path_in_repo=hf_path,
            rows=rows,
            commit_message=f"Archive {len(rows)} validator task(s)",
        )
    except Exception as exc:
        error = _archive_batch_error_message(exc)
        for task_name, _, entry, _ in prepared:
            record_task_archive_status(
                config=config,
                task_name=task_name,
                pool_label=pool_label,
                status="upload_failed",
                archive_hour_value=archive_entry_hour(entry),
                hf_path=hf_path,
                error=error,
                archive_reason=str(entry.get("archive_reason") or "rotation"),
            )
        log.exception("Task archive[%s]: HF batch upload failed for %d task(s)", pool_label, len(prepared))
        return 0

    upload_url = getattr(upload_result, "commit_url", None) or str(upload_result or "")
    completed = 0
    for task_name, _, entry, _ in prepared:
        pool.remove(task_name)
        record_task_archive_status(
            config=config,
            task_name=task_name,
            pool_label=pool_label,
            status="uploaded_delete_pending",
            archive_hour_value=archive_entry_hour(entry),
            hf_path=hf_path,
            archive_reason=str(entry.get("archive_reason") or "rotation"),
        )
        completed += 1
    deleted = retry_pending_archived_task_deletes(config, (pool,))
    leased_count = len([name for name in task_names if name in leased_names])
    lease_note = f"; {leased_count} active lease snapshot(s) existed" if leased_count else ""
    log.info(
        "Task archive[%s]: uploaded %d task(s) to %s (%s); removed from pool and deleted %d archived local task(s)%s",
        pool_label,
        completed,
        hf_path,
        upload_url,
        deleted,
        lease_note,
    )
    return completed


def active_duel_task_names(config: RunConfig) -> set[str]:
    try:
        state = v._load_state(config.validate_root / "state.json")
    except Exception:
        return set()
    return v._active_duel_task_names(state)


def pending_archive_delete_task_names(config: RunConfig) -> set[str]:
    tasks = load_task_archive_ledger(task_archive_ledger_path(config)).get("tasks") or {}
    return {
        str(task_name)
        for task_name, entry in tasks.items()
        if isinstance(entry, dict) and entry.get("status") == "uploaded_delete_pending"
    }


def _parse_archive_updated_at(entry: dict[str, Any]) -> datetime | None:
    raw = entry.get("updated_at")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def archived_task_delete_ready(entry: dict[str, Any], *, now: datetime, grace_seconds: int) -> bool:
    updated_at = _parse_archive_updated_at(entry)
    if updated_at is None or grace_seconds <= 0:
        return True
    return (now - updated_at).total_seconds() >= grace_seconds


def pool_task_by_name(pool: v.TaskPool, task_name: str) -> v.PoolTask | None:
    return next((task for task in pool.list_tasks() if task.task_name == task_name), None)


def pool_for_archive_label(pools_by_label: dict[str, v.TaskPool], pool_label: str) -> v.TaskPool | None:
    pool = pools_by_label.get(pool_label)
    if pool is not None:
        return pool
    if pool_label.startswith("king-transition-"):
        return pools_by_label.get(pool_label.removeprefix("king-transition-"))
    return None


def retry_failed_task_uploads(
    *,
    config: RunConfig,
    pools_by_label: dict[str, v.TaskPool],
    king: v.ValidatorSubmission | None,
    upload_jsonl: Any | None = None,
    upload_jsonl_rows: Any = append_hf_dataset_jsonl_rows,
) -> int:
    if not _task_archive_enabled(config):
        return 0
    ledger = load_task_archive_ledger(task_archive_ledger_path(config))
    entries = [
        (str(task_name), entry)
        for task_name, entry in (ledger.get("tasks") or {}).items()
        if isinstance(entry, dict) and entry.get("status") in _ARCHIVE_UPLOAD_RETRY_STATUSES
    ]
    if upload_jsonl is None:
        groups: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
        for task_name, entry in sorted(entries):
            pool_label = str(entry.get("pool_label") or "")
            hf_path = str(entry.get("hf_path") or task_archive_jsonl_path(pool_label, archive_entry_hour(entry)))
            groups.setdefault((pool_label, hf_path), []).append((task_name, entry))
        leased_names = active_duel_task_names(config)
        completed = 0
        with _TASK_ARCHIVE_UPLOAD_LOCK:
            for (pool_label, hf_path), grouped_entries in sorted(groups.items()):
                pool = pool_for_archive_label(pools_by_label, pool_label)
                if pool is None:
                    for task_name, _ in grouped_entries:
                        record_task_archive_status(
                            config=config,
                            task_name=task_name,
                            pool_label=pool_label or "unknown",
                            status="upload_failed",
                            error=f"cannot retry upload; no pool named {pool_label!r}",
                        )
                    continue
                completed += _upload_pending_archive_batch(
                    config=config,
                    pool=pool,
                    pool_label=pool_label,
                    hf_path=hf_path,
                    entries=grouped_entries,
                    king=king,
                    leased_names=leased_names,
                    upload_jsonl_rows=upload_jsonl_rows,
                )
        return completed

    # Backward-compatible single-row retry path used by older tests and any
    # callers that inject the original upload_jsonl hook.
    retried = 0
    leased_names = active_duel_task_names(config)
    for task_name, entry in sorted(entries):
        latest = load_task_archive_ledger(task_archive_ledger_path(config)).get("tasks", {}).get(task_name, {})
        if archive_entry_upload_is_complete(latest):
            pool_label = str(latest.get("pool_label") or entry.get("pool_label") or "") if isinstance(latest, dict) else ""
            pool = pool_for_archive_label(pools_by_label, pool_label)
            if pool is not None:
                pool.remove(task_name)
            continue
        if isinstance(latest, dict) and latest.get("status") not in _ARCHIVE_UPLOAD_RETRY_STATUSES:
            continue
        pool_label = str(
            (latest.get("pool_label") if isinstance(latest, dict) else None)
            or entry.get("pool_label")
            or ""
        )
        archive_reason = str(
            (latest.get("archive_reason") if isinstance(latest, dict) else None)
            or entry.get("archive_reason")
            or "rotation"
        )
        pool = pool_for_archive_label(pools_by_label, pool_label)
        if pool is None:
            record_task_archive_status(
                config=config,
                task_name=task_name,
                pool_label=pool_label or "unknown",
                status="upload_failed",
                error=f"cannot retry upload; no pool named {pool_label!r}",
            )
            continue
        task = pool_task_by_name(pool, task_name)
        if task is None:
            latest = load_task_archive_ledger(task_archive_ledger_path(config)).get("tasks", {}).get(task_name, {})
            if archive_entry_upload_is_complete(latest):
                pool.remove(task_name)
                continue
            record_task_archive_status(
                config=config,
                task_name=task_name,
                pool_label=pool_label,
                status="upload_failed",
                error="cannot retry upload; task is no longer present in the pool",
            )
            continue
        archive_pool_task_to_hf_jsonl(
            config=config,
            pool=pool,
            task=task,
            pool_label=pool_label,
            king=king,
            leased_task_names=leased_names,
            upload_jsonl=upload_jsonl,
            archive_reason=archive_reason,
        )
        latest = load_task_archive_ledger(task_archive_ledger_path(config)).get("tasks", {}).get(task_name, {})
        if isinstance(latest, dict) and latest.get("status") != "upload_failed":
            retried += 1
    return retried


def retry_pending_archived_task_deletes(config: RunConfig, pools: Sequence[v.TaskPool]) -> int:
    leased_names = active_duel_task_names(config)
    now = datetime.now(tz=UTC)
    grace_seconds = 0
    ledger_path = task_archive_ledger_path(config)
    removed = 0
    with _TASK_ARCHIVE_LOCK:
        ledger = load_task_archive_ledger(ledger_path)
        tasks = dict(ledger.get("tasks") or {})
        for task_name, entry in list(tasks.items()):
            if not isinstance(entry, dict) or entry.get("status") != "uploaded_delete_pending":
                continue
            if not archived_task_delete_ready(entry, now=now, grace_seconds=grace_seconds):
                continue
            if task_name in leased_names:
                continue
            for pool in pools:
                pool.remove(task_name)
            if rollout_export_enabled(config):
                try:
                    rollout_hf_path = export_task_rollouts_to_hf(config=config, task_name=task_name)
                    if rollout_hf_path:
                        entry["rollout_hf_path"] = rollout_hf_path
                except Exception as exc:
                    entry["updated_at"] = v._timestamp()
                    entry["error"] = f"rollout export failed: {exc}"
                    tasks[task_name] = entry
                    continue
            task_root = config.tasks_root / task_name
            try:
                if task_root.exists():
                    shutil.rmtree(task_root)
            except Exception as exc:
                entry["updated_at"] = v._timestamp()
                entry["error"] = str(exc)
                tasks[task_name] = entry
                continue
            entry["status"] = "uploaded_deleted"
            entry["updated_at"] = v._timestamp()
            entry.pop("error", None)
            tasks[task_name] = entry
            removed += 1
        ledger["tasks"] = tasks
        write_task_archive_ledger(ledger_path, ledger)
    return removed


def _saved_task_fill_cursor_path(config: RunConfig, pool_label: str) -> Path:
    return config.validate_root / f"saved-task-fill-cursor-{safe_pool_label(pool_label)}.json"


def is_complete_saved_task_dir(task_dir: Path) -> bool:
    task_subdir = task_dir / "task"
    return (
        task_dir.is_dir()
        and task_dir.name.startswith("validate-")
        and (task_subdir / "task.json").is_file()
        and (task_subdir / "task.txt").is_file()
        and (task_subdir / "commit.json").is_file()
        and (task_subdir / "reference.patch").is_file()
        and (task_subdir / "original").is_dir()
        and (task_subdir / "reference").is_dir()
    )


def saved_task_can_fill_pool(task_dir: Path) -> bool:
    # A structurally complete task (prompt + reference patch + original/reference
    # checkouts) is fillable. The king is (re)solved at fill time, so no baseline
    # pre-solve artifact is required.
    return is_complete_saved_task_dir(task_dir)


def pool_task_names_from_disk(validate_root: Path) -> set[str]:
    names: set[str] = set()
    for path in validate_root.glob("task-pool*/*.json"):
        try:
            payload = json.loads(path.read_text())
            task_name = str(payload.get("task_name") or path.stem) if isinstance(payload, dict) else path.stem
            if task_name:
                names.add(task_name)
        except Exception:
            names.add(path.stem)
    return names


def claim_saved_task_for_pool(
    config: RunConfig,
    pool: v.TaskPool,
    pool_label: str,
    extra_exclude: set[str] | None = None,
) -> Path | None:
    if not config.tasks_root.exists():
        return None
    with _SAVED_TASK_FILL_LOCK:
        existing = (
            pool.names()
            | pool_task_names_from_disk(config.validate_root)
            | archived_task_names(config)
            | (extra_exclude or set())
        )
        existing_fingerprints = (
            _task_root_fingerprints(_pool_task_roots(pool))
            | _task_root_fingerprints(_pool_task_roots_from_disk(config))
            | _task_root_fingerprints(config.tasks_root / name for name in (extra_exclude or set()))
            | set(_SAVED_TASK_FILL_IN_FLIGHT_FINGERPRINTS.values())
            | archived_task_fingerprints(config)
        )
        candidates: list[tuple[Path, str | None]] = []
        for task_dir in sorted(config.tasks_root.glob("validate-*"), key=lambda p: p.name):
            fingerprint = task_content_fingerprint(task_dir)
            if (
                saved_task_can_fill_pool(task_dir)
                and task_dir.name not in existing
                and task_dir.name not in _SAVED_TASK_FILL_IN_FLIGHT
                and fingerprint not in existing_fingerprints
            ):
                candidates.append((task_dir, fingerprint))
        if not candidates:
            return None
        cursor_path = _saved_task_fill_cursor_path(config, pool_label)
        last_name = ""
        try:
            payload = json.loads(cursor_path.read_text())
            if isinstance(payload, dict):
                last_name = str(payload.get("last_task_name") or "")
        except Exception:
            pass
        start = next((idx for idx, candidate in enumerate(candidates) if candidate[0].name > last_name), 0)
        chosen, chosen_fingerprint = candidates[start]
        _SAVED_TASK_FILL_IN_FLIGHT.add(chosen.name)
        if chosen_fingerprint:
            _SAVED_TASK_FILL_IN_FLIGHT_FINGERPRINTS[chosen.name] = chosen_fingerprint
        try:
            cursor_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(cursor_path, {"last_task_name": chosen.name, "updated_at": v._timestamp()})
        except Exception:
            log.exception("Pool manager[%s]: failed to persist saved-task cursor", pool_label)
        return chosen


def release_saved_task_claim(task_name: str | None) -> None:
    if not task_name:
        return
    with _SAVED_TASK_FILL_LOCK:
        _SAVED_TASK_FILL_IN_FLIGHT.discard(task_name)
        _SAVED_TASK_FILL_IN_FLIGHT_FINGERPRINTS.pop(task_name, None)


def pool_filler_worker_count(config: RunConfig) -> int:
    return max(1, int(config.validate_pool_filler_concurrency)) * _POOL_FILLER_WORKER_OVERSUBSCRIBE


def pool_filler_executor_workers(config: RunConfig) -> int:
    return pool_filler_worker_count(config) * 2


def pool_solve_slot(pool_solve_semaphore: threading.Semaphore | None):
    return pool_solve_semaphore if pool_solve_semaphore is not None else nullcontext()


def _load_manager_state(config: RunConfig) -> v.ValidatorState:
    try:
        return v._load_state(config.validate_root / "state.json")
    except Exception:
        return v.ValidatorState()


def _pool_filler_paused_for_active_duel(config: RunConfig) -> bool:
    """Yield Docker/OpenRouter capacity to the validator while a duel is in flight."""
    return _load_manager_state(config).active_duel is not None


def _allocate_and_save_task_name(config: RunConfig, state_lock: threading.Lock) -> str:
    with state_lock:
        state = _load_manager_state(config)
        task_name = v._allocate_task_name(state)
        v._save_state(config.validate_root / "state.json", state)
        return task_name


def _remove_solution_artifacts(*, task_name: str, solution_name: str, config: RunConfig) -> None:
    try:
        task_paths = resolve_task_paths(config.tasks_root, task_name)
    except FileNotFoundError:
        return
    shutil.rmtree(build_solution_paths(task_paths, solution_name).root, ignore_errors=True)


def reset_solution_artifacts(*, task_name: str, solution_name: str, config: RunConfig) -> None:
    """Remove a solution workspace and fail if a partial directory remains."""
    try:
        task_paths = resolve_task_paths(config.tasks_root, task_name)
    except FileNotFoundError:
        return
    solution_root = build_solution_paths(task_paths, solution_name).root
    if not solution_root.exists():
        return
    shutil.rmtree(solution_root)
    if solution_root.exists():
        raise RuntimeError(f"failed to remove stale solution workspace {solution_root}")


def _prepare_one_task_for_pool(
    *,
    config: RunConfig,
    pool: v.TaskPool,
    pool_label: str,
    state_lock: threading.Lock,
    pool_solve_semaphore: threading.Semaphore | None = None,
) -> bool:
    state = _load_manager_state(config)
    king = state.current_king
    if king is None or config.validate_task_pool_target <= 0:
        return False

    should_prepare, reason, archive_rotation = pool_should_prepare_task(
        config=config,
        pool=pool,
        king=king,
        pool_label=pool_label,
    )
    if not should_prepare:
        return False
    log.debug("Pool manager[%s]: preparing task (%s)", pool_label, reason)

    generated_task_root: Path | None = None
    saved_task_name: str | None = None
    archive_reservation_name: str | None = None
    archive_reservation_hour: str | None = None
    reserved_fingerprint_name: str | None = None
    added_to_pool = False
    try:
        if config.validate_task_pool_fill_from_saved:
            saved_task_root = claim_saved_task_for_pool(
                config,
                pool,
                pool_label,
                extra_exclude=v._active_duel_task_names(state),
            )
            if saved_task_root is None:
                return False
            task_name = saved_task_root.name
            saved_task_name = task_name
            task_root = str(saved_task_root)
            log.info("Pool manager[%s]: reusing saved task %s", pool_label, task_name)
        else:
            task_name = _allocate_and_save_task_name(config, state_lock)
            if archive_rotation:
                reservation = reserve_archive_quota(config=config, task_name=task_name, pool_label=pool_label)
                if reservation is None:
                    log.info("Pool manager[%s]: hourly archive generation quota exhausted before generating %s", pool_label, task_name)
                    return False
                archive_reservation_name = task_name
                archive_reservation_hour = str(reservation["archive_hour"])
            with _POOL_GENERATION_SEMAPHORE:
                generate_result = generate_task_run(task_name=task_name, config=config)
            task_root = generate_result.task_root
            generated_task_root = Path(task_root)
            log.info("Pool manager[%s]: generated task %s", pool_label, task_name)

        if archive_rotation and archive_reservation_name is None:
            reservation = reserve_archive_quota(config=config, task_name=task_name, pool_label=pool_label)
            if reservation is None:
                log.info("Pool manager[%s]: hourly archive generation quota exhausted before preparing %s", pool_label, task_name)
                return False
            archive_reservation_name = task_name
            archive_reservation_hour = str(reservation["archive_hour"])

        if v._count_patch_lines(Path(task_root) / "task" / "reference.patch") < v._MIN_PATCH_LINES:
            log.info("Pool manager[%s]: skipping %s (patch too small)", pool_label, task_name)
            return False

        # Reject duplicate task content BEFORE spending any baseline/king solve
        # compute. The insert-time check below remains the final guard; this
        # avoids burning solves on commits already pooled/archived or currently
        # being solved by another worker, and reserves this fingerprint so
        # concurrent workers skip the same freshly generated commit.
        early_fingerprint = task_content_fingerprint(Path(task_root))
        if early_fingerprint:
            with _SAVED_TASK_FILL_LOCK:
                known_fingerprints = (
                    _task_root_fingerprints(_pool_task_roots(pool))
                    | _task_root_fingerprints(_pool_task_roots_from_disk(config))
                    | set(_SAVED_TASK_FILL_IN_FLIGHT_FINGERPRINTS.values())
                    | archived_task_fingerprints(config)
                )
                if early_fingerprint in known_fingerprints:
                    log.info("Pool manager[%s]: skipping %s (duplicate task content, pre-solve)", pool_label, task_name)
                    return False
                _SAVED_TASK_FILL_IN_FLIGHT_FINGERPRINTS[task_name] = early_fingerprint
                reserved_fingerprint_name = task_name

        current_state = _load_manager_state(config)
        current_king = current_state.current_king
        if current_king is None or current_king.hotkey != king.hotkey or current_king.commit_sha != king.commit_sha:
            log.info("Pool manager[%s]: discarding %s (king changed before solve)", pool_label, task_name)
            return False

        # No baseline pre-solve: the king solve below is the sole solve. Its
        # timeout (and the stored per-task duel timeout) come from a static
        # qualification budget instead of timing a baseline cursor run.
        agent_timeout = v._POOL_KING_QUALIFY_TIMEOUT_SECONDS
        reset_solution_artifacts(task_name=task_name, solution_name="king", config=config)
        king_cfg = replace(v._build_agent_config(config, current_king), agent_timeout=agent_timeout)
        try:
            with pool_solve_slot(pool_solve_semaphore):
                king_result = solve_task_run(task_name=task_name, solution_name="king", config=king_cfg)
        except Exception as exc:
            log.info("Pool manager[%s]: king solve failed for %s; using empty patch: %s", pool_label, task_name, exc)
            reset_solution_artifacts(task_name=task_name, solution_name="king", config=config)
            v._ensure_empty_solution(task_name=task_name, solution_name="king", config=king_cfg, reason=str(exc))
            king_result = None
        if king_result is not None and king_result.exit_reason == "time_limit_exceeded":
            log.info("Pool manager[%s]: king timed out on %s (agent_timeout=%ss)", pool_label, task_name, agent_timeout)

        current_state = _load_manager_state(config)
        current_king = current_state.current_king
        if current_king is None or current_king.hotkey != king.hotkey or current_king.commit_sha != king.commit_sha:
            log.info("Pool manager[%s]: discarding %s (king changed during solve)", pool_label, task_name)
            return False

        qualifies, skip_reason = v._king_solve_qualifies_for_pool(task_name=task_name, config=config)
        if not qualifies:
            log.info("Pool manager[%s]: skipping %s (%s)", pool_label, task_name, skip_reason)
            return False

        try:
            with v._open_subtensor(config) as sub:
                # SubtensorApi() can reset non-bittensor loggers; keep pool-manager progress visible.
                v._setup_logging(debug=config.debug)
                creation_block = sub.block
        except Exception:
            creation_block = 0

        candidate = v.PoolTask(
            task_name=task_name,
            task_root=task_root,
            creation_block=creation_block,
            cursor_elapsed=0.0,
            king_lines=0,
            king_similarity=0.0,
            baseline_lines=0,
            agent_timeout_seconds=agent_timeout,
            king_hotkey=current_king.hotkey,
            king_commit_sha=current_king.commit_sha,
        )
        healthy, reason = v._pool_task_has_healthy_king_cache(config=config, task=candidate)
        if not healthy:
            log.info("Pool manager[%s]: skipping %s (%s)", pool_label, task_name, reason)
            return False

        archive_task: v.PoolTask | None = None
        with _POOL_FILL_ADD_LOCK:
            leased_task_names = v._active_duel_task_names(current_state)
            pending_archive_names = pending_archive_task_names(config)
            candidate_fingerprint = task_content_fingerprint(Path(candidate.task_root))
            existing_fingerprints = (
                _task_root_fingerprints(_pool_task_roots(pool))
                | _task_root_fingerprints(_pool_task_roots_from_disk(config))
                | _task_root_fingerprints(config.tasks_root / name for name in leased_task_names)
                | _task_root_fingerprints(config.tasks_root / name for name in pending_archive_names)
                | {
                    fp
                    for name, fp in _SAVED_TASK_FILL_IN_FLIGHT_FINGERPRINTS.items()
                    if name != task_name
                }
                | archived_task_fingerprints(config)
            )
            if candidate_fingerprint and candidate_fingerprint in existing_fingerprints:
                log.info("Pool manager[%s]: skipping %s (duplicate task content)", pool_label, task_name)
                return False
            should_archive = archive_rotation and archive_reservation_name is not None
            if should_archive:
                if archive_quota_remaining(
                    config,
                    pool_label=pool_label,
                    hour=archive_reservation_hour,
                ) <= 0:
                    log.info(
                        "Pool manager[%s]: hourly archive quota exhausted before inserting %s",
                        pool_label,
                        task_name,
                    )
                    release_archive_reservation(config=config, task_name=archive_reservation_name)
                    archive_reservation_name = None
                    return False
                archive_task = select_rotation_archive_task(
                    pool.list_tasks(),
                    candidate_name=candidate.task_name,
                    leased_task_names=leased_task_names,
                    excluded_task_names=pending_archive_names,
                )
                if archive_task is None:
                    log.info(
                        "Pool manager[%s]: no unleased existing task is available to archive before inserting %s",
                        pool_label,
                        task_name,
                    )
                    release_archive_reservation(config=config, task_name=archive_reservation_name)
                    archive_reservation_name = None
                    return False
            prune_first = v._static_pool_replacement_prune_names(config=config, pool=pool, king=current_king)
            pruned = pool.add(
                candidate,
                keep=config.validate_task_pool_target + (1 if archive_task is not None else 0),
                prune_first=prune_first,
                preserve=leased_task_names | pending_archive_names,
            )
            if archive_task is not None:
                hour = archive_reservation_hour or archive_hour()
                record_task_archive_status(
                    config=config,
                    task_name=archive_task.task_name,
                    pool_label=pool_label,
                    status="pool_inserted",
                    archive_hour_value=hour,
                    hf_path=task_archive_jsonl_path(pool_label, hour),
                    content_fingerprint=task_content_fingerprint(Path(archive_task.task_root)),
                )
                release_archive_reservation(config=config, task_name=archive_reservation_name)
                archive_reservation_name = None
        added_to_pool = True
        log.info("Pool manager[%s]: added %s (pool size=%d, pruned=%d)", pool_label, task_name, pool.size(), pruned)
        if archive_task is not None:
            archive_pool_task_to_hf_jsonl(
                config=config,
                pool=pool,
                task=archive_task,
                pool_label=pool_label,
                king=current_king,
                leased_task_names=v._active_duel_task_names(_load_manager_state(config)),
                upload_now=False,
            )
        return True
    except Exception as exc:
        if v._is_github_rate_limit_error(exc):
            v._note_pool_generation_rate_limit(f"Pool manager[{pool_label}]")
        log.exception("Pool manager[%s]: error preparing task", pool_label)
        return False
    finally:
        release_saved_task_claim(saved_task_name)
        if reserved_fingerprint_name is not None and reserved_fingerprint_name != saved_task_name:
            release_saved_task_claim(reserved_fingerprint_name)
        release_archive_reservation(config=config, task_name=archive_reservation_name)
        if not added_to_pool and generated_task_root is not None and generated_task_root.exists():
            shutil.rmtree(generated_task_root, ignore_errors=True)


def cleanup_old_task_workspaces(config: RunConfig, pools: Sequence[v.TaskPool]) -> None:
    keep_names = active_duel_task_names(config) | pending_archive_delete_task_names(config)
    for pool in pools:
        keep_names |= pool.names()
    with _SAVED_TASK_FILL_LOCK:
        keep_names |= set(_SAVED_TASK_FILL_IN_FLIGHT)
    v._cleanup_old_tasks(
        config.tasks_root,
        keep_names=keep_names,
        min_age_seconds=config.validate_task_cleanup_min_age_seconds,
    )
    v._cleanup_tasks_until_disk_headroom(
        tasks_root=config.tasks_root,
        min_free_bytes=config.validate_min_free_disk_bytes,
        keep_names=keep_names,
        max_dirs_per_pass=config.validate_disk_cleanup_max_dirs_per_pass,
    )


def _pool_worker_loop(
    *,
    config: RunConfig,
    pool: v.TaskPool,
    pool_label: str,
    stop_event: threading.Event,
    state_lock: threading.Lock,
    pool_solve_semaphore: threading.Semaphore,
) -> None:
    while not stop_event.is_set():
        backoff_remaining = v._pool_generation_backoff_remaining()
        if backoff_remaining > 0:
            stop_event.wait(min(backoff_remaining, 30.0))
            continue
        if _pool_filler_paused_for_active_duel(config):
            stop_event.wait(5)
            continue
        did_work = _prepare_one_task_for_pool(
            config=config,
            pool=pool,
            pool_label=pool_label,
            state_lock=state_lock,
            pool_solve_semaphore=pool_solve_semaphore,
        )
        stop_event.wait(1 if did_work else 5)


def run_pool_manager(config: RunConfig) -> None:
    v._setup_logging(debug=config.debug)
    require_task_archive_config(config)
    paths = prepare_pool_manager_paths(config)
    pool = v.TaskPool(paths.pool_dir)
    retest_pool = v.TaskPool(paths.retest_pool_dir)
    stop_event = threading.Event()
    state_lock = threading.Lock()

    def _request_stop(_signum: int, _frame: Any) -> None:
        stop_event.set()

    try:
        import signal

        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
    except ValueError:
        pass

    solve_slots = max(1, int(config.validate_pool_filler_concurrency))
    workers = pool_filler_worker_count(config)
    pool_solve_semaphore = threading.BoundedSemaphore(solve_slots)
    log.info("Starting pool manager with %d worker(s) per pool and %d solve slot(s) at %s", workers, solve_slots, paths.root)
    with ThreadPoolExecutor(max_workers=pool_filler_executor_workers(config)) as executor:
        for _ in range(workers):
            executor.submit(
                _pool_worker_loop,
                config=config,
                pool=pool,
                pool_label="primary",
                stop_event=stop_event,
                state_lock=state_lock,
                pool_solve_semaphore=pool_solve_semaphore,
            )
            executor.submit(
                _pool_worker_loop,
                config=config,
                pool=retest_pool,
                pool_label="retest",
                stop_event=stop_event,
                state_lock=state_lock,
                pool_solve_semaphore=pool_solve_semaphore,
            )
        while not stop_event.is_set():
            state = _load_manager_state(config)
            retried_uploads = retry_failed_task_uploads(
                config=config,
                pools_by_label={"primary": pool, "retest": retest_pool},
                king=state.current_king,
            )
            if retried_uploads:
                log.info("Retried and completed %d failed task archive upload(s)", retried_uploads)
            archived_names = removable_archived_pool_names(config)
            if archived_names:
                pool.remove_many(archived_names)
                retest_pool.remove_many(archived_names)
            removed = retry_pending_archived_task_deletes(config, (pool, retest_pool))
            if removed:
                log.info("Completed local deletion for %d archived task(s)", removed)
            active_rollout_tasks = pool.names() | retest_pool.names() | active_duel_task_names(config)
            try:
                exported_rollouts = export_retired_rollouts_to_hf(
                    config=config,
                    active_task_names=active_rollout_tasks,
                )
            except Exception:
                log.exception("Retired rollout export pass failed; pool manager loop will continue")
                exported_rollouts = 0
            if exported_rollouts:
                log.info("Exported %d retired rollout task bundle(s) to Hugging Face", exported_rollouts)
            if config.clear_uploaded_rollouts:
                try:
                    cleared_rollouts = clear_uploaded_rollout_tasks(
                        root=config.resolved_rollout_root(),
                        active_task_names=active_rollout_tasks,
                        max_dirs=config.validate_disk_cleanup_max_dirs_per_pass,
                    )
                except Exception:
                    log.exception("Uploaded rollout cleanup pass failed; pool manager loop will continue")
                    cleared_rollouts = 0
                if cleared_rollouts:
                    log.info("Cleared %d uploaded rollout task bundle(s) from local disk", cleared_rollouts)
            cleanup_old_task_workspaces(config, (pool, retest_pool))
            stop_event.wait(max(1, int(config.validate_poll_interval_seconds)))
