from __future__ import annotations

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def default_rollout_root(workspace_root: Path) -> Path:
    return workspace_root / "workspace" / "rollouts"


def task_rollout_dir(root: Path, task_name: str) -> Path:
    return root / "tasks" / task_name


def rollout_jsonl_path(root: Path, task_name: str) -> Path:
    return task_rollout_dir(root, task_name) / "rollouts.jsonl"


def rollout_record_dir(root: Path, task_name: str) -> Path:
    return task_rollout_dir(root, task_name) / "records"


def rollout_record_path(root: Path, task_name: str, rollout_id: str) -> Path:
    return rollout_record_dir(root, task_name) / f"{rollout_id}.json"


def write_json_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as tmp:
        json.dump(payload, tmp, sort_keys=True, separators=(",", ":"))
        tmp.write("\n")
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, path)
    return path


def append_rollout(root: Path, record: dict[str, Any]) -> Path:
    task_name = str(record["task_name"])
    rollout_id = str(record["rollout_id"])
    record_path = write_json_atomic(rollout_record_path(root, task_name, rollout_id), record)
    index_path = rollout_jsonl_path(root, task_name)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
    return record_path


def update_rollout(root: Path, task_name: str, rollout_id: str | None, updates: dict[str, Any]) -> bool:
    if not rollout_id:
        return False
    path = rollout_record_path(root, task_name, rollout_id)
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(record, dict):
        return False
    write_json_atomic(path, {**record, **updates})
    return True


def load_task_rollouts(root: Path, task_name: str) -> list[dict[str, Any]]:
    records_dir = rollout_record_dir(root, task_name)
    if records_dir.exists():
        rows: list[dict[str, Any]] = []
        for path in sorted(records_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if isinstance(payload, dict):
                rows.append(payload)
        return rows
    return load_task_rollouts_jsonl(root, task_name)


def load_task_rollouts_jsonl(root: Path, task_name: str) -> list[dict[str, Any]]:
    path = rollout_jsonl_path(root, task_name)
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def write_gzip_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
