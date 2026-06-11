from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tau.rollouts.redaction import public_rollout
from tau.rollouts.store import load_task_rollouts, write_gzip_jsonl, write_json_atomic

log = logging.getLogger("swe-eval.rollouts.export-hf")


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


_MANIFEST_ENTRY_RE = re.compile(
    r'"(?P<name>[^"]+)"\s*:\s*\{\s*'
    r'"exported_at"\s*:\s*"(?P<exported_at>[^"]+)"\s*,\s*'
    r'"hf_path"\s*:\s*"(?P<hf_path>[^"]+)"\s*,\s*'
    r'"task_name"\s*:\s*"(?P<task_name>[^"]+)"\s*\}'
)


def _salvage_export_manifest(path: Path, raw: str) -> dict[str, Any]:
    """Recover entries from a corrupt manifest instead of dropping history.

    A truncated manifest (e.g. ENOSPC mid-write) must not read as empty:
    the next mark_task_rollouts_exported() would persist that empty view,
    the exporter would re-upload every local bundle, and the uploaded-dir
    pruner would stop recognizing anything as uploaded.
    """
    tasks: dict[str, Any] = {}
    for match in _MANIFEST_ENTRY_RE.finditer(raw):
        tasks[match.group("name")] = {
            "task_name": match.group("task_name"),
            "hf_path": match.group("hf_path"),
            "exported_at": match.group("exported_at"),
        }
    backup = path.with_suffix(path.suffix + ".corrupt.bak")
    try:
        shutil.copyfile(path, backup)
    except OSError:
        log.exception("Could not back up corrupt rollout export manifest %s", path)
    manifest = {"tasks": tasks}
    try:
        write_json_atomic(path, manifest)
    except OSError:
        log.exception("Could not rewrite salvaged rollout export manifest %s", path)
    log.warning(
        "Rollout export manifest %s was corrupt; salvaged %d entries (backup at %s)",
        path,
        len(tasks),
        backup,
    )
    return manifest


def load_export_manifest(root: Path) -> dict[str, Any]:
    path = rollout_export_manifest_path(root)
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return {"tasks": {}}
    try:
        payload = json.loads(raw)
    except Exception:
        return _salvage_export_manifest(path, raw)
    if not isinstance(payload, dict) or not isinstance(payload.get("tasks"), dict):
        return _salvage_export_manifest(path, raw)
    return {"tasks": dict(payload["tasks"])}


def write_export_manifest(root: Path, manifest: dict[str, Any]) -> None:
    path = rollout_export_manifest_path(root)
    write_json_atomic(path, manifest)


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


def export_retired_rollouts_to_hf(
    *,
    config: Any,
    active_task_names: set[str],
    upload_file: Any | None = None,
) -> int:
    if not rollout_export_enabled(config):
        return 0
    root = getattr(config, "resolved_rollout_root")()
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
        except Exception:
            log.exception("Rollout export failed for %s; continuing with remaining tasks", task_name)
            continue
        if not path_in_repo:
            continue
        log.info("Exported rollout task bundle %s to %s", task_name, path_in_repo)
        count += 1
        manifest = load_export_manifest(root)
    return count
