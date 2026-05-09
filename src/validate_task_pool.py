from __future__ import annotations

import json
import logging
import re
import secrets
import threading
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

import httpx

from config import RunConfig
from workspace import build_solution_paths, resolve_task_paths, write_json

log = logging.getLogger("swe-eval.validate")

_POOL_SOLVE_TIMEOUT_SECONDS = 300
_POOL_FILLER_RATE_LIMIT_BACKOFF_SECONDS = 300.0
_POOL_GENERATION_BACKOFF_LOCK = threading.Lock()
_SAVED_TASK_FILL_LOCK = threading.Lock()
_SAVED_TASK_FILL_IN_FLIGHT: set[str] = set()
_pool_generation_backoff_until = 0.0


def _timestamp() -> str:
    return datetime.now(tz=UTC).isoformat()


@dataclass(slots=True)
class PoolTask:
    task_name: str
    task_root: str
    creation_block: int
    cursor_elapsed: float
    king_lines: int
    king_similarity: float
    baseline_lines: int = 0
    agent_timeout_seconds: int = 0
    king_hotkey: str = ""
    king_commit_sha: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PoolTask:
        cursor_elapsed = float(d["cursor_elapsed"])
        return cls(
            task_name=str(d["task_name"]),
            task_root=str(d["task_root"]),
            creation_block=int(d["creation_block"]),
            cursor_elapsed=cursor_elapsed,
            king_lines=int(d["king_lines"]),
            king_similarity=float(d["king_similarity"]),
            baseline_lines=int(d.get("baseline_lines", 0)),
            agent_timeout_seconds=int(d.get("agent_timeout_seconds") or _POOL_SOLVE_TIMEOUT_SECONDS),
            king_hotkey=str(d.get("king_hotkey") or ""),
            king_commit_sha=str(d.get("king_commit_sha") or ""),
        )


def _duel_agent_timeout(task: PoolTask) -> int:
    if task.agent_timeout_seconds > 0:
        return task.agent_timeout_seconds
    return _POOL_SOLVE_TIMEOUT_SECONDS


def _order_duel_tasks_for_submission(tasks: list[PoolTask]) -> list[PoolTask]:
    """Spread short and long timeout tasks across the submission order."""
    if len(tasks) <= 2:
        return list(tasks)

    ordered = sorted(tasks, key=lambda task: (_duel_agent_timeout(task), task.cursor_elapsed, task.task_name))
    bucket_count = min(5, len(ordered))
    bucket_size = (len(ordered) + bucket_count - 1) // bucket_count
    buckets = [ordered[i : i + bucket_size] for i in range(0, len(ordered), bucket_size)]

    balanced: list[PoolTask] = []
    for idx in range(bucket_size):
        for bucket in buckets:
            if idx < len(bucket):
                balanced.append(bucket[idx])
    return balanced


class TaskPool:
    """Thread-safe pool of pre-solved tasks shared across all duels.

    Tasks are NOT removed on read so every active duel can reuse the same
    king work. Each duel tracks which tasks it has already used and passes an
    ``exclude`` set to skip them.
    """

    def __init__(self, pool_dir: Path, tasks_root: Path | None = None) -> None:
        self._pool_dir = pool_dir
        self._tasks_root = tasks_root
        self._pool_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def size(self) -> int:
        with self._lock:
            count = 0
            for p in self._pool_dir.glob("*.json"):
                if self._load_task_file(p) is not None:
                    count += 1
            return count

    def names(self) -> set[str]:
        with self._lock:
            names: set[str] = set()
            for p in self._pool_dir.glob("*.json"):
                task = self._load_task_file(p)
                if task is not None and task.task_name:
                    names.add(task.task_name)
            return names

    def add(self, task: PoolTask) -> None:
        path = self._pool_dir / f"{task.task_name}.json"
        with self._lock:
            write_json(path, task.to_dict())

    def list_tasks(self) -> list[PoolTask]:
        with self._lock:
            tasks: list[PoolTask] = []
            for p in sorted(self._pool_dir.glob("*.json")):
                task = self._load_task_file(p)
                if task is not None:
                    tasks.append(task)
            return tasks

    def newest(self, limit: int, exclude: set[str] | None = None) -> list[PoolTask]:
        if limit <= 0:
            return []
        excluded = exclude or set()
        with self._lock:
            candidates: list[PoolTask] = []
            for p in sorted(self._pool_dir.glob("*.json")):
                task = self._load_task_file(p)
                if task is None or task.task_name in excluded:
                    continue
                candidates.append(task)
            candidates.sort(key=lambda task: (task.creation_block, task.task_name), reverse=True)
            return candidates[:limit]

    def remove(self, task_name: str) -> bool:
        path = self._pool_dir / f"{task_name}.json"
        with self._lock:
            existed = path.exists()
            path.unlink(missing_ok=True)
            return existed

    def take(self, min_block: int, exclude: set[str] | None = None) -> PoolTask | None:
        """Return a pool task without removing it.

        Skips tasks whose name is in *exclude* (already used by this duel).
        ``min_block`` is kept for call-site compatibility but no longer filters
        cached tasks; a restart should be able to use the persisted pool.
        """
        del min_block
        excluded = exclude or set()
        with self._lock:
            candidates: list[PoolTask] = []
            for p in sorted(self._pool_dir.glob("*.json")):
                task = self._load_task_file(p)
                if task is None or task.task_name in excluded:
                    continue
                candidates.append(task)
            if candidates:
                candidates.sort(key=lambda task: task.task_name)
                return candidates[secrets.randbelow(len(candidates))]
            return None

    # Keep pop() for backward compat (used by nothing now, but safe to have)
    def pop(self, min_block: int) -> PoolTask | None:
        del min_block
        with self._lock:
            for p in sorted(self._pool_dir.glob("*.json")):
                task = self._load_task_file(p)
                p.unlink(missing_ok=True)
                if task is not None:
                    return task
            return None

    def prune(self, keep: int) -> int:
        """Remove the oldest pool tasks if pool exceeds *keep* entries."""
        with self._lock:
            files = [p for p in sorted(self._pool_dir.glob("*.json")) if self._load_task_file(p) is not None]
            if len(files) <= keep:
                return 0
            removed = 0
            for p in files[:-keep]:
                p.unlink(missing_ok=True)
                removed += 1
            return removed

    def flush(self) -> int:
        with self._lock:
            count = 0
            for p in self._pool_dir.glob("*.json"):
                p.unlink(missing_ok=True)
                count += 1
            return count

    def _load_task_file(self, path: Path) -> PoolTask | None:
        try:
            task = PoolTask.from_dict(json.loads(path.read_text()))
        except Exception:
            path.unlink(missing_ok=True)
            return None
        if not self._is_usable_task(task):
            log.warning(
                "Dropping stale pool entry %s: task workspace is missing or incomplete",
                task.task_name,
            )
            path.unlink(missing_ok=True)
            return None
        return task

    def _is_usable_task(self, task: PoolTask) -> bool:
        if self._tasks_root is None:
            return True
        task_root = self._tasks_root / task.task_name
        task_subdir = task_root / "task"
        required = (
            task_subdir / "task.json",
            task_subdir / "task.txt",
            task_subdir / "commit.json",
            task_subdir / "reference.patch",
            task_subdir / "original",
            task_root / "solutions" / "king" / "solve.json",
            task_root / "solutions" / "king" / "solution.diff",
            task_root / "solutions" / "king" / "repo",
        )
        return all(path.exists() for path in required)


class TaskPoolRefreshBudget:
    """Shared permit counter for periodic full-pool additions.

    Pool filler threads normally sleep when the pool is already at target size,
    but this budget lets a bounded number of those threads add fresh tasks
    either once per time interval (standalone generator) or once per validator
    epoch (live validator). Successful refresh tasks are kept; the pool is not
    pruned back to the target.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._next_refresh_at: float | None = None
        self._epoch_mode = False
        self._last_epoch_index: int | None = None
        self._started_reported = False
        self._active = False
        self._completed = 0
        self._in_flight = 0

    def trigger_epoch(self, *, current_block: int, config: RunConfig) -> tuple[bool, int]:
        count = max(0, int(config.validate_task_pool_refresh_count))
        epoch_blocks = max(1, int(config.validate_weight_interval_blocks))
        epoch_index = max(0, int(current_block)) // epoch_blocks
        if count <= 0:
            return False, epoch_index

        with self._lock:
            self._epoch_mode = True
            if self._last_epoch_index == epoch_index:
                return False, epoch_index
            self._last_epoch_index = epoch_index
            if self._active:
                return False, epoch_index
            self._active = True
            self._completed = 0
            self._in_flight = 0
            self._started_reported = False
            return True, epoch_index

    def claim(self, *, config: RunConfig) -> tuple[bool, bool]:
        count = max(0, int(config.validate_task_pool_refresh_count))
        interval = max(0, int(config.validate_task_pool_refresh_interval_seconds))
        if count <= 0:
            return False, False

        with self._lock:
            if self._epoch_mode:
                if not self._active:
                    return False, False
                started = not self._started_reported
                self._started_reported = True
                if self._completed + self._in_flight >= count:
                    return False, started
                self._in_flight += 1
                return True, started

            if interval <= 0:
                return False, False
            now = time.monotonic()
            if self._next_refresh_at is None:
                self._next_refresh_at = now + interval
                return False, False

            started = False
            if not self._active:
                if now < self._next_refresh_at:
                    return False, False
                self._active = True
                self._completed = 0
                self._in_flight = 0
                started = True

            if self._completed + self._in_flight >= count:
                return False, started

            self._in_flight += 1
            return True, started

    def finish(self, *, config: RunConfig, success: bool) -> bool:
        count = max(0, int(config.validate_task_pool_refresh_count))
        interval = max(0, int(config.validate_task_pool_refresh_interval_seconds))
        if count <= 0:
            return False

        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            if success:
                self._completed += 1
            if self._active and self._completed >= count:
                self._active = False
                self._completed = 0
                self._started_reported = False
                if not self._epoch_mode and interval > 0:
                    self._next_refresh_at = time.monotonic() + interval
                return True
        return False


def _is_github_rate_limit_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    githubish = "github" in text or "api.github.com" in text or "gh:" in text
    rate_limited = (
        "rate limit" in text
        or "too many requests" in text
        or "http 403" in text
        or "http 429" in text
        or "403 forbidden" in text
        or "429 too many requests" in text
    )
    return githubish and rate_limited


def _pool_generation_backoff_remaining() -> float:
    with _POOL_GENERATION_BACKOFF_LOCK:
        return max(0.0, _pool_generation_backoff_until - time.monotonic())


def _note_github_api_rate_limit(context: str) -> None:
    global _pool_generation_backoff_until
    now = time.monotonic()
    next_until = now + _POOL_FILLER_RATE_LIMIT_BACKOFF_SECONDS
    with _POOL_GENERATION_BACKOFF_LOCK:
        extended = next_until > _pool_generation_backoff_until + 1.0
        _pool_generation_backoff_until = max(_pool_generation_backoff_until, next_until)
    if extended:
        log.warning(
            "%s: GitHub rate limit detected; pausing GitHub API work for %.0fs",
            context,
            _POOL_FILLER_RATE_LIMIT_BACKOFF_SECONDS,
        )


def _note_pool_generation_rate_limit(pool_label: str) -> None:
    _note_github_api_rate_limit(f"Pool filler[{pool_label}]")


def _github_response_is_rate_limited(resp: httpx.Response) -> bool:
    if resp.status_code == 429:
        return True
    if resp.status_code != 403:
        return False
    remaining = resp.headers.get("x-ratelimit-remaining")
    if remaining == "0":
        return True
    text = resp.text[:500].lower()
    return "rate limit" in text or "too many requests" in text


def _missing_runtime_secrets(config: RunConfig) -> list[str]:
    missing: list[str] = []
    if not config.openrouter_api_key:
        missing.append("OPENROUTER_API_KEY")
    return missing


def _zero_scored_duel_reason(duel_id: int, rounds: Sequence[Any]) -> str:
    errors = [str(getattr(r, "error", "")) for r in rounds if getattr(r, "error", None)]
    sample = "; ".join(errors[:3])
    if sample:
        return f"duel {duel_id} produced zero scored rounds; retrying instead of recording a defense; sample errors: {sample}"
    return f"duel {duel_id} produced zero scored rounds; retrying instead of recording a defense"


def _saved_task_fill_cursor_path(config: RunConfig, pool_label: str) -> Path:
    safe_label = validate_saved_task_cursor_label(pool_label)
    return config.validate_root / f"saved-task-fill-cursor-{safe_label}.json"


def validate_saved_task_cursor_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", label.strip() or "pool")


def _is_complete_saved_task_dir(task_dir: Path) -> bool:
    task_subdir = task_dir / "task"
    return (
        task_dir.is_dir()
        and task_dir.name.startswith("validate-")
        and (task_subdir / "task.json").is_file()
        and (task_subdir / "task.txt").is_file()
        and (task_subdir / "commit.json").is_file()
        and (task_subdir / "reference.patch").is_file()
    )


def _pool_task_names_from_disk(validate_root: Path) -> set[str]:
    names: set[str] = set()
    for path in (validate_root / "task-pool").glob("*.json"):
        try:
            payload = json.loads(path.read_text())
            task_name = str(payload.get("task_name") or path.stem) if isinstance(payload, dict) else path.stem
            if task_name:
                names.add(task_name)
        except Exception:
            names.add(path.stem)
    return names


def _claim_saved_task_for_pool(
    config: RunConfig,
    pool: TaskPool,
    pool_label: str,
    extra_exclude: set[str] | None = None,
) -> Path | None:
    """Pick the next saved task workspace for a pool fill attempt."""
    if not config.tasks_root.exists():
        return None
    with _SAVED_TASK_FILL_LOCK:
        existing = pool.names() | _pool_task_names_from_disk(config.validate_root) | (extra_exclude or set())
        candidates = [
            task_dir
            for task_dir in sorted(config.tasks_root.glob("validate-*"), key=lambda p: p.name)
            if (
                _is_complete_saved_task_dir(task_dir)
                and task_dir.name not in existing
                and task_dir.name not in _SAVED_TASK_FILL_IN_FLIGHT
            )
        ]
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

        start = 0
        if last_name:
            for idx, candidate in enumerate(candidates):
                if candidate.name > last_name:
                    start = idx
                    break
            else:
                start = 0
        chosen = candidates[start]
        _SAVED_TASK_FILL_IN_FLIGHT.add(chosen.name)
        try:
            cursor_path.parent.mkdir(parents=True, exist_ok=True)
            write_json(cursor_path, {"last_task_name": chosen.name, "updated_at": _timestamp()})
        except Exception:
            log.exception("Pool filler[%s]: failed to persist saved-task cursor", pool_label)
        return chosen


def _release_saved_task_claim(task_name: str | None) -> None:
    if not task_name:
        return
    with _SAVED_TASK_FILL_LOCK:
        _SAVED_TASK_FILL_IN_FLIGHT.discard(task_name)


def _cached_solution_summary(
    *,
    task_name: str,
    solution_name: str,
    config: RunConfig,
) -> tuple[str, float] | None:
    try:
        task_paths = resolve_task_paths(config.tasks_root, task_name)
        solution_paths = build_solution_paths(task_paths, solution_name)
        if not solution_paths.solve_json_path.is_file() or not solution_paths.solution_diff_path.is_file():
            return None
        payload = json.loads(solution_paths.solve_json_path.read_text())
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            return None
        exit_reason = str(result.get("exit_reason") or "")
        elapsed = float(result.get("elapsed_seconds") or _POOL_SOLVE_TIMEOUT_SECONDS)
        return exit_reason, elapsed
    except Exception:
        return None


def _task_summary_fields(*, task_name: str, config: RunConfig) -> dict[str, str]:
    try:
        task_paths = resolve_task_paths(config.tasks_root, task_name)
        payload = json.loads(task_paths.task_json_path.read_text())
    except Exception:
        return {}
    if not isinstance(payload, dict):
        return {}

    task_payload = payload.get("task")
    if not isinstance(task_payload, dict):
        task_payload = payload

    fields: dict[str, str] = {}
    title = str(task_payload.get("title") or "").strip()
    description = str(task_payload.get("description") or "").strip()
    if not description:
        description = str(task_payload.get("prompt_text") or "").strip()
    summary = _compact_task_summary(description)
    if title:
        fields["task_title"] = title
    if summary:
        fields["task_summary"] = summary
    return fields


def _task_summaries_for_names(task_names: Sequence[Any] | None, config: RunConfig) -> list[dict[str, str]]:
    summaries: list[dict[str, str]] = []
    for raw_name in task_names or []:
        task_name = str(raw_name)
        item = {"task_name": task_name}
        item.update(_task_summary_fields(task_name=task_name, config=config))
        summaries.append(item)
    return summaries


def _compact_task_summary(text: str, max_chars: int = 700) -> str:
    summary = re.sub(r"\s+", " ", text).strip()
    if len(summary) <= max_chars:
        return summary
    return summary[: max_chars - 3].rstrip() + "..."
