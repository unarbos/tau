from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, TimeoutError as _FuturesTimeoutError, wait as _futures_wait
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import bittensor as bt
import httpx

from config import RunConfig, SolverAgentSource
from pipeline import _setup_logging, compare_task_run, generate_task_run, solve_task_run
from r2 import (
    duel_to_summary,
    fetch_chain_data,
    publish_dashboard_data,
    publish_duel_data,
    publish_duel_index,
    publish_round_data,
    publish_training_data,
)
from workspace import write_json

log = logging.getLogger("swe-eval.validate")
_DEFAULT_GITHUB_AGENT_SUBDIR = "agent"
_GITHUB_COMMIT_RE = re.compile(
    r"^(?P<repo>[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)@(?P<sha>[0-9a-fA-F]{7,64})$"
)
_BASELINE_MODEL = "gemini-3-flash"
_MIN_PATCH_LINES = 100
_MIN_DECISIVE_ROUNDS = 10
_PARALLEL_DUEL_PER_ROUND_TIMEOUT = 900.0
_PARALLEL_DUEL_HARD_TIMEOUT = 1800.0


def _challenger_wins(wins: int, losses: int, margin: int) -> bool:
    """Check if the challenger has won the duel based on decisive rounds only.

    Ties are fully excluded. The challenger must win more than half of
    decisive rounds plus a margin, and there must be at least
    ``_MIN_DECISIVE_ROUNDS`` decisive rounds to avoid fluky outcomes.
    """
    decisive = wins + losses
    if decisive < _MIN_DECISIVE_ROUNDS:
        return False
    return wins > decisive // 2 + margin


# ---------------------------------------------------------------------------
# Discord new-king notification
# ---------------------------------------------------------------------------

def _notify_new_king(
    new_king: "ValidatorSubmission",
    old_king: "ValidatorSubmission | None",
    duel_result: "DuelResult",
) -> None:
    """Post a gold embed to Discord when a new king is crowned."""
    token = os.environ.get("DISCORD_BOT_TOKEN")
    channel_id = os.environ.get("DISCORD_CHANNEL_ID")
    if not token or not channel_id:
        log.debug("Discord notification skipped (DISCORD_BOT_TOKEN or DISCORD_CHANNEL_ID not set)")
        return

    repo = new_king.repo_full_name
    uid = new_king.uid
    desc_lines = [f"**UID {uid}** is the new king with `{repo}`"]
    if old_king:
        desc_lines.append(
            f"Dethroned **UID {old_king.uid}** (`{old_king.repo_full_name}`)"
        )
    desc_lines.append(
        f"Score: **{duel_result.wins}W / {duel_result.losses}L / {duel_result.ties}T**"
    )

    embed = {
        "title": "New King Crowned",
        "description": "\n".join(desc_lines),
        "color": 0xFFD700,
        "url": f"https://github.com/{repo}",
        "footer": {"text": f"Duel #{duel_result.duel_id}"},
    }

    try:
        resp = httpx.post(
            f"https://discord.com/api/v10/channels/{channel_id}/messages",
            headers={"Authorization": f"Bot {token}", "Content-Type": "application/json"},
            json={"embeds": [embed]},
            timeout=10,
        )
        if resp.status_code >= 400:
            log.warning("Discord notification failed (%d): %s", resp.status_code, resp.text[:200])
        else:
            log.info("Discord new-king notification sent for UID %s", uid)
    except Exception:
        log.exception("Discord notification failed (non-fatal)")


# ---------------------------------------------------------------------------
# Data structures (unchanged for dashboard compatibility)
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ValidatorSubmission:
    hotkey: str
    uid: int
    repo_full_name: str
    repo_url: str
    commit_sha: str
    commitment: str
    commitment_block: int

    @property
    def agent_ref(self) -> str:
        return f"{self.repo_full_name}@{self.commit_sha}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ValidatorSubmission:
        return cls(
            hotkey=str(payload["hotkey"]), uid=int(payload["uid"]),
            repo_full_name=str(payload["repo_full_name"]),
            repo_url=str(payload["repo_url"]),
            commit_sha=str(payload["commit_sha"]),
            commitment=str(payload["commitment"]),
            commitment_block=int(payload["commitment_block"]),
        )


@dataclass(slots=True)
class ValidationRoundResult:
    task_name: str
    winner: str
    king_lines: int
    challenger_lines: int
    king_similarity_ratio: float
    challenger_similarity_ratio: float
    king_challenger_similarity: float
    task_root: str
    king_compare_root: str
    challenger_compare_root: str
    baseline_lines: int = 0
    error: str | None = None

    @property
    def scored(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class DuelResult:
    duel_id: int
    started_at: str
    finished_at: str
    king_before: ValidatorSubmission
    challenger: ValidatorSubmission
    rounds: list[ValidationRoundResult]
    wins: int
    losses: int
    ties: int
    king_after: ValidatorSubmission
    king_replaced: bool
    disqualification_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "duel_id": self.duel_id, "started_at": self.started_at,
            "finished_at": self.finished_at,
            "king_before": self.king_before.to_dict(),
            "challenger": self.challenger.to_dict(),
            "rounds": [r.to_dict() for r in self.rounds],
            "wins": self.wins, "losses": self.losses, "ties": self.ties,
            "king_after": self.king_after.to_dict(),
            "king_replaced": self.king_replaced,
            "disqualification_reason": self.disqualification_reason,
        }


@dataclass(slots=True)
class ValidatorState:
    current_king: ValidatorSubmission | None = None
    queue: list[ValidatorSubmission] = field(default_factory=list)
    seen_hotkeys: list[str] = field(default_factory=list)
    retired_hotkeys: list[str] = field(default_factory=list)
    disqualified_hotkeys: list[str] = field(default_factory=list)
    locked_commitments: dict[str, str] = field(default_factory=dict)
    last_weight_block: int | None = None
    next_task_index: int = 1
    next_duel_index: int = 1
    king_since: str | None = None
    king_duels_defended: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "current_king": self.current_king.to_dict() if self.current_king else None,
            "queue": [s.to_dict() for s in self.queue],
            "seen_hotkeys": self.seen_hotkeys,
            "retired_hotkeys": self.retired_hotkeys,
            "disqualified_hotkeys": self.disqualified_hotkeys,
            "locked_commitments": self.locked_commitments,
            "last_weight_block": self.last_weight_block,
            "next_task_index": self.next_task_index,
            "next_duel_index": self.next_duel_index,
            "king_since": self.king_since,
            "king_duels_defended": self.king_duels_defended,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ValidatorState:
        ck = payload.get("current_king")
        raw_locked = payload.get("locked_commitments", {})
        return cls(
            current_king=ValidatorSubmission.from_dict(ck) if isinstance(ck, dict) else None,
            queue=[ValidatorSubmission.from_dict(i) for i in payload.get("queue", []) if isinstance(i, dict)],
            seen_hotkeys=[str(i) for i in payload.get("seen_hotkeys", [])],
            retired_hotkeys=[str(i) for i in payload.get("retired_hotkeys", [])],
            disqualified_hotkeys=[str(i) for i in payload.get("disqualified_hotkeys", [])],
            locked_commitments={str(k): str(v) for k, v in raw_locked.items()} if isinstance(raw_locked, dict) else {},
            last_weight_block=int(payload["last_weight_block"]) if payload.get("last_weight_block") is not None else None,
            next_task_index=int(payload.get("next_task_index", 1)),
            next_duel_index=int(payload.get("next_duel_index", 1)),
            king_since=payload.get("king_since"),
            king_duels_defended=int(payload.get("king_duels_defended", 0)),
        )


@dataclass(slots=True)
class ValidatePaths:
    root: Path
    state_path: Path
    duels_dir: Path
    pool_dir: Path


@dataclass(slots=True)
class ValidateStageResult:
    validate_root: str
    king_uid: int
    king_hotkey: str
    king_repo: str
    duel_count: int


@dataclass(slots=True)
class PoolTask:
    task_name: str
    task_root: str
    creation_block: int
    cursor_elapsed: float
    king_lines: int
    king_similarity: float
    baseline_lines: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PoolTask:
        return cls(
            task_name=str(d["task_name"]), task_root=str(d["task_root"]),
            creation_block=int(d["creation_block"]),
            cursor_elapsed=float(d["cursor_elapsed"]),
            king_lines=int(d["king_lines"]),
            king_similarity=float(d["king_similarity"]),
            baseline_lines=int(d.get("baseline_lines", 0)),
        )


# ---------------------------------------------------------------------------
# Task pool
# ---------------------------------------------------------------------------

class TaskPool:
    """Thread-safe pool of pre-solved tasks shared across all duels.

    Tasks are NOT removed on read so every active duel can reuse the same
    baseline+king work.  Each duel tracks which tasks it has already used
    and passes an ``exclude`` set to skip them.
    """

    def __init__(self, pool_dir: Path) -> None:
        self._pool_dir = pool_dir
        self._pool_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def size(self) -> int:
        with self._lock:
            return len(list(self._pool_dir.glob("*.json")))

    def add(self, task: PoolTask) -> None:
        path = self._pool_dir / f"{task.task_name}.json"
        write_json(path, task.to_dict())

    def take(self, min_block: int, exclude: set[str] | None = None) -> PoolTask | None:
        """Return a pool task without removing it.

        Skips tasks whose name is in *exclude* (already used by this duel).
        A task with ``creation_block == 0`` (chain lookup failed during pool
        fill) is treated as universally eligible.
        """
        excluded = exclude or set()
        with self._lock:
            for p in sorted(self._pool_dir.glob("*.json")):
                try:
                    d = json.loads(p.read_text())
                    task_name = str(d.get("task_name", ""))
                    if task_name in excluded:
                        continue
                    creation_block = int(d.get("creation_block", 0))
                    if creation_block == 0 or creation_block > min_block:
                        return PoolTask.from_dict(d)
                except Exception:
                    p.unlink(missing_ok=True)
            return None

    # Keep pop() for backward compat (used by nothing now, but safe to have)
    def pop(self, min_block: int) -> PoolTask | None:
        with self._lock:
            for p in sorted(self._pool_dir.glob("*.json")):
                try:
                    d = json.loads(p.read_text())
                    if int(d.get("creation_block", 0)) > min_block:
                        path = p
                        path.unlink(missing_ok=True)
                        return PoolTask.from_dict(d)
                except Exception:
                    p.unlink(missing_ok=True)
            return None

    def prune(self, keep: int) -> int:
        """Remove the oldest pool tasks if pool exceeds *keep* entries."""
        with self._lock:
            files = sorted(self._pool_dir.glob("*.json"))
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


# ---------------------------------------------------------------------------
# Pool filler (background thread)
# ---------------------------------------------------------------------------

def _pool_filler_loop(
    config: RunConfig,
    state: ValidatorState,
    pool: TaskPool,
    stop_event: threading.Event,
    state_lock: threading.Lock,
    pool_starved: threading.Event | None = None,
) -> None:
    while not stop_event.is_set():
        try:
            if state.current_king is None:
                stop_event.wait(5)
                continue

            starved = pool_starved is not None and pool_starved.is_set()
            if pool.size() >= config.validate_task_pool_target and not starved:
                stop_event.wait(2)
                continue

            with state_lock:
                task_name = _allocate_task_name(state)
            log.info("Pool filler: generating task %s", task_name)

            generate_result = generate_task_run(task_name=task_name, config=config)
            task_root = generate_result.task_root

            ref_patch_path = Path(task_root) / "task" / "reference.patch"
            if _count_patch_lines(ref_patch_path) < _MIN_PATCH_LINES:
                log.info("Pool filler: skipping %s (patch too small)", task_name)
                continue

            king = state.current_king
            if king is None:
                continue
            king_hotkey_before = king.hotkey

            baseline_cfg = _build_baseline_config(config)
            king_cfg = replace(_build_agent_config(config, king), agent_timeout=300)

            baseline_start = time.monotonic()
            with ThreadPoolExecutor(max_workers=2) as solve_exec:
                baseline_fut = solve_exec.submit(
                    solve_task_run, task_name=task_name,
                    solution_name="baseline", config=baseline_cfg,
                )
                king_fut = solve_exec.submit(
                    solve_task_run, task_name=task_name,
                    solution_name="king", config=king_cfg,
                )
                baseline_fut.result()
                king_fut.result()
            baseline_elapsed = time.monotonic() - baseline_start

            current_king = state.current_king
            if current_king is None or current_king.hotkey != king_hotkey_before:
                log.info("Pool filler: discarding %s (king changed during solve)", task_name)
                continue

            king_compare = compare_task_run(task_name=task_name, solution_names=["baseline", "king"], config=config)

            try:
                with _open_subtensor(config) as sub:
                    creation_block = sub.block
            except Exception:
                creation_block = 0

            if state.current_king is None or state.current_king.hotkey != king_hotkey_before:
                log.info("Pool filler: discarding %s (king changed during compare)", task_name)
                continue

            pool.add(PoolTask(
                task_name=task_name,
                task_root=task_root,
                creation_block=creation_block,
                cursor_elapsed=baseline_elapsed,
                king_lines=king_compare.matched_changed_lines,
                king_similarity=king_compare.similarity_ratio,
                baseline_lines=king_compare.total_changed_lines_a,
            ))
            pruned = pool.prune(keep=config.validate_task_pool_target)
            if pool_starved is not None:
                pool_starved.clear()
            log.info("Pool filler: added %s (pool size: %d, pruned: %d)", task_name, pool.size(), pruned)

        except Exception:
            log.exception("Pool filler: error generating task (retrying)")
            stop_event.wait(5)


# ---------------------------------------------------------------------------
# Duel runner (runs independently per challenger)
# ---------------------------------------------------------------------------

def _run_duel(
    *,
    config: RunConfig,
    state: ValidatorState,
    king: ValidatorSubmission,
    challenger: ValidatorSubmission,
    duel_id: int,
    pool: TaskPool,
    cancel_event: threading.Event,
    on_round_complete: Any = None,
    pool_starved: threading.Event | None = None,
) -> DuelResult:
    margin = config.validate_win_margin
    started_at = _timestamp()
    rounds: list[ValidationRoundResult] = []
    wins = losses = ties = scored = 0
    used_tasks: set[str] = set()
    duel_start_mono = time.monotonic()
    max_total_rounds = config.validate_duel_rounds * 3
    _POOL_WAIT_TIMEOUT = 300
    _MAX_POOL_TIMEOUTS = 3
    consecutive_pool_timeouts = 0

    log.info("Duel %d: king uid=%s vs challenger uid=%s (%s), %d rounds, "
             "need >50%%+%d of decisive rounds (min %d decisive, ties ignored)",
             duel_id, king.uid, challenger.uid, challenger.repo_full_name,
             config.validate_duel_rounds, margin, _MIN_DECISIVE_ROUNDS)

    while scored < config.validate_duel_rounds and not cancel_event.is_set():
        duel_elapsed = time.monotonic() - duel_start_mono
        if duel_elapsed >= config.validate_duel_timeout_seconds:
            log.warning("Duel %d: wall-clock timeout after %.0fs (scored %d/%d)",
                        duel_id, duel_elapsed, scored, config.validate_duel_rounds)
            break

        if len(rounds) >= max_total_rounds:
            log.warning("Duel %d: exceeded max total rounds %d (scored %d/%d, errors %d)",
                        duel_id, max_total_rounds, scored, config.validate_duel_rounds,
                        len(rounds) - scored)
            break

        task = pool.take(min_block=challenger.commitment_block, exclude=used_tasks)
        if task is None:
            if pool_starved is not None:
                pool_starved.set()
            pool_wait_start = time.monotonic()
            while task is None and not cancel_event.is_set():
                waited = time.monotonic() - pool_wait_start
                if waited >= _POOL_WAIT_TIMEOUT:
                    log.warning("Duel %d: pool wait timeout after %.0fs (pool empty for task with min_block=%d)",
                                duel_id, waited, challenger.commitment_block)
                    break
                if time.monotonic() - duel_start_mono >= config.validate_duel_timeout_seconds:
                    break
                cancel_event.wait(3)
                task = pool.take(min_block=challenger.commitment_block, exclude=used_tasks)
            if task is None:
                consecutive_pool_timeouts += 1
                if consecutive_pool_timeouts >= _MAX_POOL_TIMEOUTS:
                    log.warning("Duel %d: aborting after %d consecutive pool timeouts (scored %d/%d)",
                                duel_id, consecutive_pool_timeouts, scored, config.validate_duel_rounds)
                    break
                continue
        consecutive_pool_timeouts = 0
        used_tasks.add(task.task_name)

        try:
            agent_timeout = min(int(task.cursor_elapsed * 2) + 1, 300)
            solution_label = f"challenger-{challenger.uid}-d{duel_id}"

            challenger_cfg = replace(_build_agent_config(config, challenger), agent_timeout=agent_timeout)
            solve_result = solve_task_run(task_name=task.task_name, solution_name=solution_label, config=challenger_cfg)
            chall_timed_out = solve_result.exit_reason == "time_limit_exceeded"

            if chall_timed_out:
                log.info("Duel %d: challenger uid=%s timed out on %s", duel_id, challenger.uid, task.task_name)

            with ThreadPoolExecutor(max_workers=2) as cmp_exec:
                chall_fut = cmp_exec.submit(
                    compare_task_run, task_name=task.task_name,
                    solution_names=["baseline", solution_label], config=config,
                )
                kc_fut = cmp_exec.submit(
                    compare_task_run, task_name=task.task_name,
                    solution_names=["king", solution_label], config=config,
                )
                chall_compare = chall_fut.result()
                kc_compare = kc_fut.result()

            c_lines = 0 if chall_timed_out else chall_compare.matched_changed_lines
            k_lines = task.king_lines

            if c_lines > k_lines:
                winner = "challenger"
            elif c_lines < k_lines:
                winner = "king"
            else:
                winner = "tie"

            result = ValidationRoundResult(
                task_name=task.task_name, winner=winner,
                king_lines=k_lines, challenger_lines=c_lines,
                king_similarity_ratio=task.king_similarity,
                challenger_similarity_ratio=0.0 if chall_timed_out else chall_compare.similarity_ratio,
                king_challenger_similarity=kc_compare.similarity_ratio,
                task_root=task.task_root,
                king_compare_root="", challenger_compare_root=chall_compare.comparison_root,
                baseline_lines=task.baseline_lines,
            )

            try:
                publish_round_data(
                    duel_id=duel_id, task_name=task.task_name,
                    tasks_root=config.tasks_root,
                    solution_labels={"baseline": "baseline", "king": "king", "challenger": solution_label},
                )
            except Exception:
                log.exception("R2 round publish failed (non-fatal)")

        except Exception as exc:
            result = ValidationRoundResult(
                task_name=task.task_name, winner="error",
                king_lines=0, challenger_lines=0,
                king_similarity_ratio=0.0, challenger_similarity_ratio=0.0,
                king_challenger_similarity=0.0,
                task_root=task.task_root, king_compare_root="", challenger_compare_root="",
                error=f"duel {duel_id} task {task.task_name} failed: {exc}",
            )

        rounds.append(result)
        if result.scored:
            if result.winner == "challenger":
                wins += 1
                scored += 1
            elif result.winner == "king":
                losses += 1
                scored += 1
            else:
                ties += 1

            decisive = wins + losses
            log.info("Duel %d round=%d: %s (W=%d L=%d T=%d, decisive=%d)",
                     duel_id, len(rounds), result.winner, wins, losses, ties, decisive)

            if _challenger_wins(wins, losses, margin):
                log.info("Duel %d: challenger uid=%s WINS (%d/%d decisive)", duel_id, challenger.uid, wins, decisive)
                break
            remaining = config.validate_duel_rounds - scored
            if remaining <= 0 and not _challenger_wins(wins, losses, margin):
                log.info("Duel %d: all rounds exhausted, challenger uid=%s did not win (%d/%d decisive)",
                         duel_id, challenger.uid, wins, decisive)
                break

        if on_round_complete is not None:
            try:
                decisive = wins + losses
                dyn_threshold = decisive // 2 + margin + 1 if decisive >= _MIN_DECISIVE_ROUNDS else _MIN_DECISIVE_ROUNDS
                on_round_complete(duel_id=duel_id, wins=wins, losses=losses, ties=ties,
                                  scored=scored, threshold=dyn_threshold, rounds=rounds)
            except Exception:
                log.exception("on_round_complete callback failed (non-fatal)")

    # Determine outcome
    king_replaced = False
    dq_reason: str | None = None
    king_after = king
    decisive = wins + losses

    log.info("Duel %d final: W=%d L=%d T=%d (decisive=%d, challenger_wins=%s)",
             duel_id, wins, losses, ties, decisive, _challenger_wins(wins, losses, margin))

    if _challenger_wins(wins, losses, margin):
        scored_sim = [r for r in rounds if r.scored and r.king_challenger_similarity > 0]
        mean_sim = sum(r.king_challenger_similarity for r in scored_sim) / len(scored_sim) if scored_sim else 0.0
        _COPY_THRESHOLD = 0.90
        if mean_sim >= _COPY_THRESHOLD:
            dq_reason = f"copy detected (similarity {mean_sim:.3f} >= {_COPY_THRESHOLD})"
            log.warning("Duel %d: %s", duel_id, dq_reason)
        else:
            king_replaced = True

    return DuelResult(
        duel_id=duel_id, started_at=started_at, finished_at=_timestamp(),
        king_before=king, challenger=challenger, rounds=rounds,
        wins=wins, losses=losses, ties=ties,
        king_after=king_after, king_replaced=king_replaced,
        disqualification_reason=dq_reason,
    )


# ---------------------------------------------------------------------------
# Parallel duel runner (all rounds run concurrently)
# ---------------------------------------------------------------------------

def _gather_pool_tasks(
    pool: TaskPool,
    n: int,
    min_block: int,
    timeout: float = 600,
    pool_starved: threading.Event | None = None,
) -> list[PoolTask]:
    """Collect up to *n* distinct tasks from the pool, waiting if needed."""
    tasks: list[PoolTask] = []
    seen: set[str] = set()
    deadline = time.monotonic() + timeout
    while len(tasks) < n:
        remaining_time = deadline - time.monotonic()
        if remaining_time <= 0:
            break
        task = pool.take(min_block=min_block, exclude=seen)
        if task is not None:
            tasks.append(task)
            seen.add(task.task_name)
        else:
            if pool_starved is not None:
                pool_starved.set()
            time.sleep(min(3, remaining_time))
    if pool_starved is not None:
        pool_starved.clear()
    return tasks


def _solve_and_compare_round(
    *,
    task: PoolTask,
    challenger: ValidatorSubmission,
    config: RunConfig,
    duel_id: int,
) -> ValidationRoundResult:
    """Run a single round: solve challenger, then compare. Thread-safe."""
    solution_label = f"challenger-{challenger.uid}-d{duel_id}"
    try:
        agent_timeout = min(int(task.cursor_elapsed * 2) + 1, 300)
        challenger_cfg = replace(
            _build_agent_config(config, challenger), agent_timeout=agent_timeout,
        )
        solve_result = solve_task_run(
            task_name=task.task_name, solution_name=solution_label,
            config=challenger_cfg,
        )
        chall_timed_out = solve_result.exit_reason == "time_limit_exceeded"
        if chall_timed_out:
            log.info("Duel %d: challenger uid=%s timed out on %s",
                     duel_id, challenger.uid, task.task_name)

        with ThreadPoolExecutor(max_workers=2) as cmp_exec:
            chall_fut = cmp_exec.submit(
                compare_task_run, task_name=task.task_name,
                solution_names=["baseline", solution_label], config=config,
            )
            kc_fut = cmp_exec.submit(
                compare_task_run, task_name=task.task_name,
                solution_names=["king", solution_label], config=config,
            )
            # Bound compare time so a wedged comparator can't pin a round forever.
            chall_compare = chall_fut.result(timeout=600)
            kc_compare = kc_fut.result(timeout=600)

        c_lines = 0 if chall_timed_out else chall_compare.matched_changed_lines
        k_lines = task.king_lines

        if c_lines > k_lines:
            winner = "challenger"
        elif c_lines < k_lines:
            winner = "king"
        else:
            winner = "tie"

        result = ValidationRoundResult(
            task_name=task.task_name, winner=winner,
            king_lines=k_lines, challenger_lines=c_lines,
            king_similarity_ratio=task.king_similarity,
            challenger_similarity_ratio=0.0 if chall_timed_out else chall_compare.similarity_ratio,
            king_challenger_similarity=kc_compare.similarity_ratio,
            task_root=task.task_root,
            king_compare_root="",
            challenger_compare_root=chall_compare.comparison_root,
            baseline_lines=task.baseline_lines,
        )

        try:
            publish_round_data(
                duel_id=duel_id, task_name=task.task_name,
                tasks_root=config.tasks_root,
                solution_labels={
                    "baseline": "baseline", "king": "king",
                    "challenger": solution_label,
                },
            )
        except Exception:
            log.exception("R2 round publish failed (non-fatal)")

        return result

    except Exception as exc:
        return ValidationRoundResult(
            task_name=task.task_name, winner="error",
            king_lines=0, challenger_lines=0,
            king_similarity_ratio=0.0, challenger_similarity_ratio=0.0,
            king_challenger_similarity=0.0,
            task_root=task.task_root, king_compare_root="",
            challenger_compare_root="",
            error=f"duel {duel_id} task {task.task_name} failed: {exc}",
        )


def _run_parallel_duel(
    *,
    config: RunConfig,
    state: ValidatorState,
    king: ValidatorSubmission,
    challenger: ValidatorSubmission,
    duel_id: int,
    pool: TaskPool,
    pool_starved: threading.Event | None = None,
    on_round_complete: Any = None,
) -> DuelResult:
    """Run a duel with all rounds executing in parallel.

    Instead of running rounds sequentially, this gathers N tasks from the
    pool up front and then launches all challenger solves + comparisons
    concurrently.  Wall-clock time is roughly that of a single round.
    """
    n_rounds = config.validate_duel_rounds
    concurrency = config.validate_round_concurrency
    margin = config.validate_win_margin
    started_at = _timestamp()

    log.info(
        "Parallel duel %d: king uid=%s vs challenger uid=%s (%s), "
        "%d rounds at concurrency %d, need >50%%+%d of decisive rounds "
        "(min %d decisive, ties ignored)",
        duel_id, king.uid, challenger.uid, challenger.repo_full_name,
        n_rounds, concurrency, margin, _MIN_DECISIVE_ROUNDS,
    )

    # Phase 1: gather tasks from pool
    log.info("Duel %d phase 1: gathering %d tasks from pool (pool size=%d)",
             duel_id, n_rounds, pool.size())
    tasks = _gather_pool_tasks(
        pool, n_rounds, min_block=challenger.commitment_block,
        timeout=config.validate_duel_timeout_seconds,
        pool_starved=pool_starved,
    )
    log.info("Duel %d: gathered %d/%d tasks", duel_id, len(tasks), n_rounds)
    if not tasks:
        log.warning("Duel %d: no tasks available, aborting", duel_id)
        return DuelResult(
            duel_id=duel_id, started_at=started_at, finished_at=_timestamp(),
            king_before=king, challenger=challenger, rounds=[],
            wins=0, losses=0, ties=0,
            king_after=king, king_replaced=False,
        )

    # Phase 2+3: solve and compare all rounds in parallel
    log.info("Duel %d phase 2: launching %d parallel solves + compares",
             duel_id, len(tasks))
    solve_start = time.monotonic()

    rounds: list[ValidationRoundResult] = []
    duel_deadline = time.monotonic() + _PARALLEL_DUEL_HARD_TIMEOUT
    last_progress_at = time.monotonic()
    last_heartbeat_at = time.monotonic()
    # Wake up frequently so we can (a) honour the hard deadline even when
    # rounds keep dribbling in slowly and (b) emit a dashboard heartbeat so
    # the public dashboard's updated_at doesn't appear frozen during long
    # duels where individual rounds take many minutes.
    _DASHBOARD_HEARTBEAT_INTERVAL = 15.0
    _WAIT_SLICE = 5.0
    # Manage the executor manually so we can force-shutdown on timeout
    # without blocking on hung worker threads. The `with` block's __exit__
    # calls shutdown(wait=True) which would deadlock the validator if a
    # solver/comparator thread is permanently stuck (e.g. a wedged docker
    # exec). We use shutdown(wait=False, cancel_futures=True) instead and
    # let any genuinely-hung threads be reaped when the process exits.
    executor = ThreadPoolExecutor(max_workers=concurrency)
    timed_out_clean_shutdown = True

    def _emit_progress() -> None:
        if on_round_complete is None:
            return
        wins = sum(1 for r in rounds if r.scored and r.winner == "challenger")
        losses = sum(1 for r in rounds if r.scored and r.winner == "king")
        ties = sum(1 for r in rounds if r.scored and r.winner == "tie")
        scored = wins + losses
        decisive = wins + losses
        dyn_threshold = decisive // 2 + margin + 1 if decisive >= _MIN_DECISIVE_ROUNDS else _MIN_DECISIVE_ROUNDS
        try:
            on_round_complete(
                duel_id=duel_id, wins=wins, losses=losses, ties=ties,
                scored=scored, threshold=dyn_threshold, rounds=rounds,
            )
        except Exception:
            log.exception("on_round_complete callback failed (non-fatal)")

    try:
        futures = {
            executor.submit(
                _solve_and_compare_round,
                task=task, challenger=challenger, config=config,
                duel_id=duel_id,
            ): task
            for task in tasks
        }
        pending = set(futures.keys())
        while pending:
            now = time.monotonic()
            slack = max(duel_deadline - now, 0.0)
            stale = now - last_progress_at
            per_round_slack = max(_PARALLEL_DUEL_PER_ROUND_TIMEOUT - stale, 0.0)
            # Cap each wait() at _WAIT_SLICE so we always come back to
            # check the hard deadline + emit a heartbeat, even when rounds
            # are slowly trickling in.
            wait_timeout = min(_WAIT_SLICE, per_round_slack, slack) if slack > 0 else 0.0
            done, pending = _futures_wait(pending, timeout=wait_timeout, return_when=FIRST_COMPLETED)
            now = time.monotonic()

            # Hard-deadline / stuck-progress check fires regardless of whether
            # any future completed in this slice. Previously this was nested
            # under `if not done`, so a duel where rounds slowly dribbled in
            # could run forever past the deadline.
            hard_timed_out = now >= duel_deadline
            stuck = (now - last_progress_at) >= _PARALLEL_DUEL_PER_ROUND_TIMEOUT
            if hard_timed_out or stuck:
                reason = "hard duel deadline" if hard_timed_out else f"no round progress in {_PARALLEL_DUEL_PER_ROUND_TIMEOUT:.0f}s"
                log.error(
                    "Duel %d: %s with %d rounds still pending (%d done); cancelling and recording as errors",
                    duel_id, reason, len(pending), len(rounds),
                )
                # Drain anything that completed in the final slice before
                # cancelling so we don't lose work.
                for future in done:
                    try:
                        result = future.result()
                    except Exception as exc:
                        task = futures[future]
                        result = ValidationRoundResult(
                            task_name=task.task_name, winner="error",
                            king_lines=0, challenger_lines=0,
                            king_similarity_ratio=0.0,
                            challenger_similarity_ratio=0.0,
                            king_challenger_similarity=0.0,
                            task_root=task.task_root, king_compare_root="",
                            challenger_compare_root="",
                            error=f"duel {duel_id} task {task.task_name} crashed: {exc}",
                        )
                    rounds.append(result)
                for fut in list(pending):
                    fut.cancel()
                    task = futures[fut]
                    rounds.append(
                        ValidationRoundResult(
                            task_name=task.task_name, winner="error",
                            king_lines=0, challenger_lines=0,
                            king_similarity_ratio=0.0,
                            challenger_similarity_ratio=0.0,
                            king_challenger_similarity=0.0,
                            task_root=task.task_root, king_compare_root="",
                            challenger_compare_root="",
                            error=f"duel {duel_id} task {task.task_name} timed out ({reason})",
                        )
                    )
                pending = set()
                timed_out_clean_shutdown = False
                try:
                    _kill_stale_containers()
                except Exception:
                    log.exception("docker cleanup after duel timeout failed (non-fatal)")
                break

            if not done:
                # No completion this slice; emit a heartbeat publish so the
                # public dashboard stays fresh even when rounds are slow.
                if (now - last_heartbeat_at) >= _DASHBOARD_HEARTBEAT_INTERVAL:
                    _emit_progress()
                    last_heartbeat_at = now
                continue

            last_progress_at = now
            for future in done:
                try:
                    result = future.result()
                except Exception as exc:
                    task = futures[future]
                    log.exception("Duel %d: round %s raised", duel_id, task.task_name)
                    result = ValidationRoundResult(
                        task_name=task.task_name, winner="error",
                        king_lines=0, challenger_lines=0,
                        king_similarity_ratio=0.0,
                        challenger_similarity_ratio=0.0,
                        king_challenger_similarity=0.0,
                        task_root=task.task_root, king_compare_root="",
                        challenger_compare_root="",
                        error=f"duel {duel_id} task {task.task_name} crashed: {exc}",
                    )
                rounds.append(result)
                _emit_progress()
            last_heartbeat_at = time.monotonic()
    finally:
        # On the happy path all rounds finished, so wait=True is fine and
        # cheap. On timeout, never wait -- hung threads would deadlock
        # the validator forever (this is the bug we were hitting).
        executor.shutdown(wait=timed_out_clean_shutdown, cancel_futures=True)

    solve_elapsed = time.monotonic() - solve_start
    log.info("Duel %d: all %d rounds completed in %.1fs", duel_id, len(rounds), solve_elapsed)

    # Phase 4: score
    wins = sum(1 for r in rounds if r.scored and r.winner == "challenger")
    losses = sum(1 for r in rounds if r.scored and r.winner == "king")
    ties = sum(1 for r in rounds if r.scored and r.winner == "tie")
    decisive = wins + losses

    challenger_won = _challenger_wins(wins, losses, margin)
    log.info("Duel %d result: W=%d L=%d T=%d (decisive=%d, challenger_wins=%s)",
             duel_id, wins, losses, ties, decisive, challenger_won)

    king_replaced = False
    dq_reason: str | None = None
    king_after = king

    if challenger_won:
        scored_sim = [r for r in rounds if r.scored and r.king_challenger_similarity > 0]
        mean_sim = (
            sum(r.king_challenger_similarity for r in scored_sim) / len(scored_sim)
            if scored_sim else 0.0
        )
        _COPY_THRESHOLD = 0.90
        if mean_sim >= _COPY_THRESHOLD:
            dq_reason = f"copy detected (similarity {mean_sim:.3f} >= {_COPY_THRESHOLD})"
            log.warning("Duel %d: %s", duel_id, dq_reason)
        else:
            king_replaced = True
            log.info("Duel %d: challenger uid=%s WINS (%d/%d decisive)",
                     duel_id, challenger.uid, wins, decisive)
    else:
        log.info("Duel %d: king defends (challenger uid=%s got %d/%d decisive, needed >50%%+%d)",
                 duel_id, challenger.uid, wins, decisive, margin)

    return DuelResult(
        duel_id=duel_id, started_at=started_at, finished_at=_timestamp(),
        king_before=king, challenger=challenger, rounds=rounds,
        wins=wins, losses=losses, ties=ties,
        king_after=king_after, king_replaced=king_replaced,
        disqualification_reason=dq_reason,
    )


# ---------------------------------------------------------------------------
# Main validator loop
# ---------------------------------------------------------------------------

def _kill_stale_containers() -> None:
    """Kill and remove all swe-eval-* containers left over from a prior run."""
    try:
        running = subprocess.run(
            ["docker", "ps", "-q", "--filter", "name=swe-eval-"],
            capture_output=True, text=True, timeout=10,
        )
        if running.returncode == 0 and running.stdout.strip():
            ids = running.stdout.strip().splitlines()
            subprocess.run(["docker", "kill", *ids], capture_output=True, timeout=30)
            log.info("Killed %d orphaned swe-eval containers", len(ids))
        stopped = subprocess.run(
            ["docker", "ps", "-aq", "--filter", "name=swe-eval-"],
            capture_output=True, text=True, timeout=10,
        )
        if stopped.returncode == 0 and stopped.stdout.strip():
            ids = stopped.stdout.strip().splitlines()
            subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, timeout=30)
    except Exception:
        log.exception("Startup container cleanup failed (non-fatal)")


def validate_loop_run(config: RunConfig) -> ValidateStageResult:
    _setup_logging(debug=config.debug)
    _kill_stale_containers()
    log.info("Scoring: %d rounds per duel, ties fully ignored, "
             "challenger must win >50%%+%d of decisive rounds (min %d decisive)",
             config.validate_duel_rounds, config.validate_win_margin, _MIN_DECISIVE_ROUNDS)

    if not config.validate_wallet_name or not config.validate_wallet_hotkey:
        raise ValueError("validate requires --wallet-name and --wallet-hotkey")

    paths = _prepare_validate_paths(config.validate_root)
    state = _load_state(paths.state_path)
    dashboard_history = _load_dashboard_history(paths.root / "dashboard_history.json")

    # Recover task index
    if config.tasks_root.exists():
        max_idx = 0
        for td in config.tasks_root.glob("validate-*"):
            parts = td.name.rsplit("-", 1)
            if len(parts) == 2 and parts[1].isdigit():
                max_idx = max(max_idx, int(parts[1]))
        if max_idx >= state.next_task_index:
            state.next_task_index = max_idx + 1

    pool = TaskPool(paths.pool_dir)
    pool_stop = threading.Event()
    pool_starved = threading.Event()
    state_lock = threading.Lock()
    validator_started_at = _timestamp()
    chain_data: dict[str, Any] | None = None
    last_king_check = 0.0

    github_client = _build_github_client(config)
    duel_count = 0

    active_duel_info: dict[str, Any] | None = None
    pool_filler_executor = ThreadPoolExecutor(
        max_workers=config.validate_pool_filler_concurrency,
    )

    try:
        with _open_subtensor(config) as subtensor:
            log.info("Connected to chain for netuid %s", config.validate_netuid)

            # Initial chain poll + king setup (no block cutoff yet so king can be selected)
            chain_data = fetch_chain_data(config.validate_netuid) or chain_data
            chain_submissions = _fetch_chain_submissions(subtensor=subtensor, github_client=github_client, config=config)
            _refresh_queue(chain_submissions=chain_submissions, config=config, state=state)

            _ensure_king(state=state)

            # Set block cutoff AFTER king is established so initial queue isn't filtered
            if config.validate_min_commitment_block == 0:
                config.validate_min_commitment_block = subtensor.block
                log.info("Auto-set min_commitment_block to current block %d",
                         config.validate_min_commitment_block)

            if state.current_king:
                if not state.king_since:
                    state.king_since = _timestamp()

            # Start pool fillers
            for _ in range(config.validate_pool_filler_concurrency):
                pool_filler_executor.submit(_pool_filler_loop, config, state, pool, pool_stop, state_lock, pool_starved)

            while True:
              try:
                current_block = subtensor.block
                log.info("Poll: block=%s king=%s queue=%d pool=%d",
                         current_block,
                         state.current_king.commitment if state.current_king else None,
                         len(state.queue), pool.size())

                # Refresh dashboard heartbeat at the top of every poll so the
                # external watchdog (which keys off dashboard_data.json mtime)
                # doesn't restart us during the multi-second chain RPC + queue
                # refresh below. Without this, a fresh validator process can be
                # killed before it ever reaches the duel-start publish path.
                try:
                    _publish_dashboard(state, dashboard_history, config, validator_started_at,
                                       active_duel_info, chain_data)
                except Exception:
                    log.exception("Pre-poll dashboard publish failed (non-fatal)")

                chain_data = fetch_chain_data(config.validate_netuid) or chain_data
                chain_submissions = _fetch_chain_submissions(subtensor=subtensor, github_client=github_client, config=config)
                _refresh_queue(chain_submissions=chain_submissions, config=config, state=state)

                if state.current_king is None and not state.queue:
                    log.info("No king and empty queue; waiting for new miners to register and commit")

                prev_king = state.current_king.hotkey if state.current_king else None
                _ensure_king(state=state)
                if state.current_king and state.current_king.hotkey != prev_king:
                    state.king_since = _timestamp()
                    state.king_duels_defended = 0

                if state.current_king and len(state.current_king.commit_sha) < 40:
                    full = _resolve_public_commit(github_client, state.current_king.repo_full_name, state.current_king.commit_sha)
                    if full:
                        state.current_king.commit_sha = full

                if state.current_king:
                    if time.monotonic() - last_king_check > 600:
                        try:
                            _maybe_disqualify_king(subtensor=subtensor, github_client=github_client, config=config, state=state)
                        except Exception:
                            log.exception("King disqualification check failed (non-fatal)")
                        last_king_check = time.monotonic()
                    try:
                        _maybe_set_weights(subtensor=subtensor, config=config, state=state, current_block=current_block)
                    except Exception:
                        log.exception("set_weights failed (non-fatal, will retry next interval)")

                # --- Serialized duel: run one challenger at a time ---
                if state.queue and state.current_king:
                    challenger = _pop_next_valid_challenger(subtensor=subtensor, github_client=github_client, config=config, state=state)
                    if challenger is not None:
                        duel_id = state.next_duel_index
                        state.next_duel_index += 1

                        active_duel_info = {
                            "king_uid": state.current_king.uid,
                            "king_repo": state.current_king.repo_full_name,
                            "challenger_uid": challenger.uid,
                            "challenger_repo": challenger.repo_full_name,
                            "threshold": _MIN_DECISIVE_ROUNDS,
                            "win_margin": config.validate_win_margin,
                            "duel_rounds": config.validate_duel_rounds,
                        }

                        def _make_progress_callback(chall_hk: str) -> Any:
                            def cb(*, duel_id: int, wins: int, losses: int, ties: int,
                                   scored: int, threshold: int, rounds: list, **kw: Any) -> None:
                                nonlocal active_duel_info
                                active_duel_info = {
                                    "king_uid": state.current_king.uid if state.current_king else None,
                                    "king_repo": state.current_king.repo_full_name if state.current_king else None,
                                    "challenger_uid": challenger.uid,
                                    "challenger_repo": challenger.repo_full_name,
                                    "threshold": threshold,
                                    "duel_rounds": config.validate_duel_rounds,
                                    "wins": wins, "losses": losses, "ties": ties,
                                    "scored": scored,
                                    "rounds": [{"task_name": r.task_name, "winner": r.winner,
                                                "king_lines": r.king_lines, "challenger_lines": r.challenger_lines,
                                                "king_similarity_ratio": r.king_similarity_ratio,
                                                "challenger_similarity_ratio": r.challenger_similarity_ratio,
                                                "king_challenger_similarity": r.king_challenger_similarity}
                                               for r in rounds if r.scored],
                                }
                                try:
                                    _publish_dashboard(state, dashboard_history, config, validator_started_at,
                                                       active_duel_info, chain_data)
                                except Exception:
                                    log.exception("Dashboard progress publish failed (non-fatal)")
                            return cb

                        log.info("Starting parallel duel %d: uid=%s (%s)",
                                 duel_id, challenger.uid, challenger.repo_full_name)

                        try:
                            duel_result = _run_parallel_duel(
                                config=config, state=state,
                                king=state.current_king, challenger=challenger,
                                duel_id=duel_id, pool=pool,
                                pool_starved=pool_starved,
                                on_round_complete=_make_progress_callback(challenger.hotkey),
                            )
                        except Exception:
                            log.exception("Parallel duel %d raised (treating as defender win)", duel_id)
                            duel_count += 1
                            active_duel_info = None
                            _save_state(paths.state_path, state)
                            time.sleep(config.validate_poll_interval_seconds)
                            continue

                        active_duel_info = None
                        duel_count += 1

                        log.info("Duel %d finished: uid=%s W=%d L=%d T=%d replaced=%s",
                                 duel_result.duel_id, challenger.uid,
                                 duel_result.wins, duel_result.losses, duel_result.ties,
                                 duel_result.king_replaced)

                        if duel_result.king_replaced:
                            replacement = _resolve_promotion_candidate(
                                subtensor=subtensor, github_client=github_client,
                                config=config, state=state, primary_candidate=challenger,
                            )
                            if replacement:
                                old_king = state.current_king
                                _retire_hotkey(state, state.current_king.hotkey)
                                state.current_king = replacement
                                duel_result.king_after = replacement
                                state.king_since = _timestamp()
                                state.king_duels_defended = 0
                                log.info("NEW KING: uid=%s (%s)", replacement.uid, replacement.agent_ref)
                                flushed = pool.flush()
                                log.info("Flushed %d pool tasks (new king)", flushed)
                                try:
                                    _notify_new_king(replacement, old_king, duel_result)
                                except Exception:
                                    log.exception("notify_new_king failed (non-fatal)")
                        elif duel_result.disqualification_reason:
                            _mark_disqualified(state, challenger.hotkey)
                        else:
                            state.king_duels_defended += 1

                        duel_dict = duel_result.to_dict()
                        _write_duel(paths, duel_result)
                        chall_label = f"challenger-{challenger.uid}-d{duel_result.duel_id}"
                        try:
                            publish_duel_data(duel_id=duel_result.duel_id, duel_dict=duel_dict)
                        except Exception:
                            log.exception("R2 duel publish failed (non-fatal)")
                        try:
                            publish_training_data(
                                duel_id=duel_result.duel_id, duel_dict=duel_dict,
                                tasks_root=config.tasks_root,
                                solution_labels={"baseline": "baseline", "king": "king", "challenger": chall_label},
                            )
                        except Exception:
                            log.exception("R2 training data publish failed (non-fatal)")
                        dashboard_history.append(duel_to_summary(duel_dict))
                        try:
                            publish_duel_index(duel_history=dashboard_history, latest_duel_dict=duel_dict)
                        except Exception:
                            log.exception("R2 index publish failed (non-fatal)")

                _save_state(paths.state_path, state)
                _save_dashboard_history(paths.root / "dashboard_history.json", dashboard_history)
                _publish_dashboard(state, dashboard_history, config, validator_started_at,
                                   active_duel_info, chain_data)
                _cleanup_old_tasks(config.tasks_root)
                _cleanup_orphaned_containers()

              except KeyboardInterrupt:
                raise
              except Exception:
                log.exception("Main loop iteration failed; will retry after poll interval")

              time.sleep(config.validate_poll_interval_seconds)

    finally:
        pool_stop.set()
        pool_filler_executor.shutdown(wait=False, cancel_futures=True)
        github_client.close()

    king = state.current_king
    if king is None:
        raise RuntimeError("validate loop exited without a current king")
    return ValidateStageResult(
        validate_root=str(paths.root), king_uid=king.uid,
        king_hotkey=king.hotkey, king_repo=king.agent_ref, duel_count=duel_count,
    )


# ---------------------------------------------------------------------------
# Dashboard publishing
# ---------------------------------------------------------------------------

def _publish_dashboard(
    state: ValidatorState, history: list[dict[str, Any]], config: RunConfig,
    validator_started_at: str,
    active_duel: dict[str, Any] | None = None,
    chain_data: dict[str, Any] | None = None,
) -> None:
    king = state.current_king
    king_dict = {
        "uid": king.uid, "hotkey": king.hotkey,
        "repo_full_name": king.repo_full_name,
        "repo_url": f"https://github.com/{king.repo_full_name}",
        "commit_sha": king.commit_sha,
    } if king else None

    active_duel_info = active_duel

    commitment_map: dict[str, dict[str, Any]] = {}
    for d in history:
        for role in ("king", "challenger"):
            hk = d.get(f"{role}_hotkey")
            if hk and hk not in commitment_map:
                commitment_map[hk] = {"uid": d.get(f"{role}_uid"), "hotkey": hk, "repo": d.get(f"{role}_repo")}

    def _resolve_hk(hk: str) -> dict[str, Any]:
        if hk in commitment_map:
            return commitment_map[hk]
        c = state.locked_commitments.get(hk, "")
        repo = c.split("@")[0] if "@" in c else c
        return {"uid": None, "hotkey": hk, "repo": repo or "unknown"}

    total_rounds = sum(
        1 for d in history for r in d.get("rounds", [])
        if r.get("winner") not in ("tie", None)
    )
    status = {
        "validator_started_at": validator_started_at,
        "netuid": config.validate_netuid,
        "scoring": {
            "method": "race",
            "duel_rounds": config.validate_duel_rounds,
            "win_margin": config.validate_win_margin,
            "min_decisive_rounds": _MIN_DECISIVE_ROUNDS,
            "ties_count": False,
            "description": "Challenger must win >50%+margin of decisive rounds (ties ignored, min decisive rounds required)",
        },
        "queue": [{"uid": s.uid, "repo": s.repo_full_name, "hotkey": s.hotkey, "commitment_block": s.commitment_block} for s in state.queue],
        "active_duel": active_duel_info,
        "disqualified": [_resolve_hk(hk) for hk in state.disqualified_hotkeys],
        "retired": [_resolve_hk(hk) for hk in state.retired_hotkeys],
        "total_rounds": total_rounds,
        "miners_seen": len(state.seen_hotkeys),
        "king_since": state.king_since,
        "king_duels_defended": state.king_duels_defended,
        "chain_data": chain_data,
    }

    payload = {"updated_at": _timestamp(), "current_king": king_dict, "duels": history, "status": status}
    try:
        write_json(config.validate_root / "dashboard_data.json", payload)
    except Exception:
        log.exception("Local dashboard write failed (non-fatal)")
    try:
        publish_dashboard_data(current_king=king_dict, duel_history=history, status=status)
    except Exception:
        log.exception("R2 dashboard publish failed (non-fatal)")


# ---------------------------------------------------------------------------
# Chain + queue management (preserved from original)
# ---------------------------------------------------------------------------

def _build_github_client(config: RunConfig) -> httpx.Client:
    headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "swe-eval-validate"}
    token = None
    if config.github_tokens:
        tokens = [t.strip() for t in config.github_tokens.split(",") if t.strip()]
        if tokens:
            token = tokens[0]
    if not token:
        token = config.github_token
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return httpx.Client(base_url="https://api.github.com", headers=headers, follow_redirects=True, timeout=config.http_timeout)


def _refresh_queue(*, chain_submissions: list[ValidatorSubmission], config: RunConfig, state: ValidatorState) -> None:
    known = set(state.seen_hotkeys)
    if state.current_king:
        known.add(state.current_king.hotkey)
    known.update(s.hotkey for s in state.queue)

    known_agents: set[str] = set()
    if state.current_king:
        known_agents.add(state.current_king.agent_ref)
    known_agents.update(s.agent_ref for s in state.queue)

    for sub in chain_submissions:
        if config.validate_min_commitment_block and sub.commitment_block < config.validate_min_commitment_block:
            continue
        locked = state.locked_commitments.get(sub.hotkey)
        if locked is not None and locked != sub.commitment:
            log.warning("Hotkey %s changed commitment; ignoring (immutable)", sub.hotkey)
            continue
        if sub.hotkey in known:
            continue
        if sub.agent_ref in known_agents:
            log.info("Hotkey %s submits already-queued agent %s; marking seen without duel", sub.hotkey, sub.agent_ref)
            state.locked_commitments[sub.hotkey] = sub.commitment
            state.seen_hotkeys.append(sub.hotkey)
            known.add(sub.hotkey)
            continue
        if config.validate_queue_size is not None and len(state.queue) >= config.validate_queue_size:
            break
        state.locked_commitments[sub.hotkey] = sub.commitment
        state.queue.append(sub)
        state.seen_hotkeys.append(sub.hotkey)
        known.add(sub.hotkey)
        known_agents.add(sub.agent_ref)
    state.queue.sort(key=lambda s: (s.commitment_block, s.uid, s.hotkey))


def _fetch_chain_submissions(*, subtensor, github_client: httpx.Client, config: RunConfig) -> list[ValidatorSubmission]:
    revealed = subtensor.commitments.get_all_revealed_commitments(config.validate_netuid)
    current_commitments = subtensor.commitments.get_all_commitments(config.validate_netuid)
    submissions: list[ValidatorSubmission] = []
    seen: set[str] = set()
    current_block = subtensor.block

    for hotkey, entries in revealed.items():
        normalized = [(int(i[0]), str(i[1])) for i in entries if isinstance(i, tuple) and len(i) == 2] if isinstance(entries, tuple) else []
        if not normalized:
            continue
        block, commitment = min(normalized, key=lambda x: x[0])
        sub = _build_submission(subtensor=subtensor, github_client=github_client, config=config, hotkey=str(hotkey), commitment=str(commitment), commitment_block=block)
        if sub:
            submissions.append(sub)
            seen.add(sub.hotkey)

    for hotkey, commitment in current_commitments.items():
        hotkey = str(hotkey)
        if hotkey in seen:
            continue
        commit_block = current_block
        try:
            meta = subtensor.commitments.get_commitment_metadata(config.validate_netuid, hotkey)
            if isinstance(meta, list):
                blocks = [int(m["block"]) for m in meta if isinstance(m, dict) and "block" in m]
                if blocks:
                    commit_block = min(blocks)
            elif isinstance(meta, dict) and "block" in meta:
                commit_block = int(meta["block"])
        except Exception:
            pass
        sub = _build_submission(subtensor=subtensor, github_client=github_client, config=config, hotkey=hotkey, commitment=str(commitment), commitment_block=commit_block)
        if sub:
            submissions.append(sub)

    submissions.sort(key=lambda s: (s.commitment_block, s.uid, s.hotkey))
    return submissions



def _build_submission(*, subtensor, github_client, config, hotkey, commitment, commitment_block) -> ValidatorSubmission | None:
    parsed = _parse_submission_commitment(commitment)
    if not parsed:
        return None
    uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(hotkey, config.validate_netuid)
    if uid is None:
        return None
    repo, sha = parsed
    full_sha = _resolve_public_commit(github_client, repo, sha)
    if not full_sha:
        return None
    return ValidatorSubmission(hotkey=hotkey, uid=int(uid), repo_full_name=repo, repo_url=f"https://github.com/{repo}.git", commit_sha=full_sha, commitment=commitment, commitment_block=commitment_block)


def _ensure_king(*, state: ValidatorState) -> None:
    if state.current_king or not state.queue:
        return
    state.current_king = state.queue.pop(0)


def _pop_next_valid_challenger(*, subtensor, github_client, config, state) -> ValidatorSubmission | None:
    while state.queue:
        c = state.queue.pop(0)
        if _submission_is_eligible(subtensor=subtensor, github_client=github_client, config=config, submission=c):
            return c
        _mark_disqualified(state, c.hotkey)
    return None


def _submission_is_eligible(*, subtensor, github_client, config, submission) -> bool:
    uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(submission.hotkey, config.validate_netuid)
    if uid is None:
        return False
    if not _is_public_commit(github_client, submission.repo_full_name, submission.commit_sha):
        return False
    submission.uid = int(uid)
    return True


def _maybe_disqualify_king(*, subtensor, github_client, config, state) -> None:
    king = state.current_king
    if not king:
        return
    if _submission_is_eligible(subtensor=subtensor, github_client=github_client, config=config, submission=king):
        return
    _mark_disqualified(state, king.hotkey)
    state.current_king = None
    state.current_king = _pop_next_valid_challenger(subtensor=subtensor, github_client=github_client, config=config, state=state)


def _retire_hotkey(state, hotkey):
    if hotkey not in state.retired_hotkeys:
        state.retired_hotkeys.append(hotkey)

def _mark_disqualified(state, hotkey):
    if hotkey not in state.disqualified_hotkeys:
        state.disqualified_hotkeys.append(hotkey)

def _resolve_promotion_candidate(*, subtensor, github_client, config, state, primary_candidate):
    if _submission_is_eligible(subtensor=subtensor, github_client=github_client, config=config, submission=primary_candidate):
        return primary_candidate
    _mark_disqualified(state, primary_candidate.hotkey)
    return _pop_next_valid_challenger(subtensor=subtensor, github_client=github_client, config=config, state=state)


# ---------------------------------------------------------------------------
# Weight setting
# ---------------------------------------------------------------------------

def _maybe_set_weights(*, subtensor, config, state, current_block):
    king = state.current_king
    if not king:
        return
    if state.last_weight_block is not None and current_block - state.last_weight_block < config.validate_weight_interval_blocks:
        return
    neurons = list(subtensor.neurons.neurons_lite(config.validate_netuid))
    if not neurons:
        log.error("Subnet %s has no neurons; skipping set_weights", config.validate_netuid)
        return
    uid = subtensor.subnets.get_uid_for_hotkey_on_subnet(king.hotkey, config.validate_netuid)
    if uid is None:
        log.error("King %s is no longer registered; skipping set_weights", king.hotkey)
        return
    king.uid = int(uid)
    uids = [int(n.uid) for n in neurons]
    weights = [1.0 if u == king.uid else 0.0 for u in uids]
    wallet = bt.Wallet(name=config.validate_wallet_name, hotkey=config.validate_wallet_hotkey, path=config.validate_wallet_path)
    resp = subtensor.extrinsics.set_weights(wallet=wallet, netuid=config.validate_netuid, uids=uids, weights=weights, wait_for_inclusion=True, wait_for_finalization=False)
    state.last_weight_block = current_block
    log.info("Set weights at block %s to king uid=%s response=%s", current_block, king.uid, resp)


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------

def _build_baseline_config(config: RunConfig) -> RunConfig:
    model = config.baseline_model or _BASELINE_MODEL
    return replace(config, solver_backend="cursor", solve_agent="baseline", solver_agent_source=None, solver_model=model)

def _build_agent_config(config: RunConfig, sub: ValidatorSubmission) -> RunConfig:
    src = SolverAgentSource(raw=sub.agent_ref, kind="github_repo", repo_url=sub.repo_url, agent_subdir=_DEFAULT_GITHUB_AGENT_SUBDIR, commit_sha=sub.commit_sha)
    return replace(config, solver_backend="docker-pi", solve_agent=sub.agent_ref, solver_agent_source=src)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _allocate_task_name(state: ValidatorState) -> str:
    idx = state.next_task_index
    state.next_task_index += 1
    ts = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    return f"validate-{ts}-{idx:06d}"

def _prepare_validate_paths(root: Path) -> ValidatePaths:
    root.mkdir(parents=True, exist_ok=True)
    duels = root / "duels"
    duels.mkdir(parents=True, exist_ok=True)
    pool = root / "task-pool"
    pool.mkdir(parents=True, exist_ok=True)
    return ValidatePaths(root=root, state_path=root / "state.json", duels_dir=duels, pool_dir=pool)

def _load_state(path: Path) -> ValidatorState:
    if not path.exists():
        return ValidatorState()
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid state file: {path}")
    return ValidatorState.from_dict(payload)

def _save_state(path: Path, state: ValidatorState) -> None:
    write_json(path, state.to_dict())

def _write_duel(paths: ValidatePaths, duel: DuelResult) -> None:
    write_json(paths.duels_dir / f"{duel.duel_id:06d}.json", duel.to_dict())

def _load_dashboard_history(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
        return payload if isinstance(payload, list) else []
    except Exception:
        log.exception("Failed to load dashboard history; starting fresh")
        return []

def _save_dashboard_history(path: Path, history: list) -> None:
    write_json(path, history)


# ---------------------------------------------------------------------------
# Commitment parsing + GitHub helpers
# ---------------------------------------------------------------------------

def _parse_submission_commitment(raw: str) -> tuple[str, str] | None:
    cleaned = raw.strip().rstrip("/")
    m = _GITHUB_COMMIT_RE.fullmatch(cleaned)
    if m:
        return m.group("repo"), m.group("sha")
    for prefix in ("https://github.com/", "github.com/"):
        if cleaned.startswith(prefix):
            parts = [p for p in cleaned[len(prefix):].split("/") if p]
            if len(parts) >= 4 and parts[2] == "commit":
                return "/".join(parts[:2]), parts[3]
    return None

_verified_commits: dict[str, str] = {}


class _TransientCommitCheckError(Exception):
    """Raised when GitHub can't be reached / rate-limits us / 5xx's. Caller
    must NOT disqualify the submission on this -- the king/challenger is
    almost certainly still valid; we just couldn't verify right now."""


def _resolve_public_commit(client: httpx.Client, repo: str, sha: str) -> str | None:
    """Returns the full commit sha if the repo+commit is verifiably public,
    or None if it is verifiably NOT public (404 / private). Raises
    _TransientCommitCheckError for any other failure (network, 5xx, 403
    rate-limit, JSON decode error). Callers must treat the exception as
    "skip this check" rather than as a disqualification."""
    cache_key = f"{repo}@{sha}"
    if cache_key in _verified_commits:
        return _verified_commits[cache_key]
    try:
        r = client.get(f"/repos/{repo}")
    except (httpx.HTTPError, OSError) as exc:
        raise _TransientCommitCheckError(f"GET /repos/{repo} failed: {exc}") from exc
    if r.status_code == 404:
        return None  # definitively not public
    if r.status_code != 200:
        # 5xx, 403 rate-limit, 401, etc -- all transient from our POV
        raise _TransientCommitCheckError(f"GET /repos/{repo} -> HTTP {r.status_code}")
    try:
        body = r.json()
    except ValueError as exc:
        raise _TransientCommitCheckError(f"GET /repos/{repo} bad json: {exc}") from exc
    if body.get("private") is True:
        return None  # definitively private
    try:
        r2 = client.get(f"/repos/{repo}/commits/{sha}")
    except (httpx.HTTPError, OSError) as exc:
        raise _TransientCommitCheckError(f"GET /repos/{repo}/commits/{sha} failed: {exc}") from exc
    if r2.status_code == 404 or r2.status_code == 422:
        return None  # commit definitively gone/invalid
    if r2.status_code != 200:
        raise _TransientCommitCheckError(f"GET /repos/{repo}/commits/{sha} -> HTTP {r2.status_code}")
    try:
        full_sha = r2.json().get("sha", sha)
    except ValueError as exc:
        raise _TransientCommitCheckError(f"GET commits bad json: {exc}") from exc
    _verified_commits[cache_key] = full_sha
    return full_sha


def _is_public_commit(client: httpx.Client, repo: str, sha: str) -> bool:
    """Returns True if verifiably public, False if verifiably not. On
    transient errors, returns True (fail-open) so we don't disqualify
    miners due to GitHub flakiness. The transient-aware variant
    _check_public_commit below is preferred for new code."""
    try:
        return _resolve_public_commit(client, repo, sha) is not None
    except _TransientCommitCheckError as exc:
        log.warning("Transient GitHub check error for %s@%s, treating as eligible: %s", repo, sha, exc)
        return True


# ---------------------------------------------------------------------------
# Chain connection + market data
# ---------------------------------------------------------------------------

def _open_subtensor(config: RunConfig):
    network = config.validate_subtensor_endpoint or config.validate_network
    if network:
        return bt.SubtensorApi(network=network, websocket_shutdown_timer=0)
    return bt.SubtensorApi(websocket_shutdown_timer=0)


# ---------------------------------------------------------------------------
# Cleanup utilities
# ---------------------------------------------------------------------------

def _cleanup_old_tasks(tasks_root: Path, keep: int = 500) -> None:
    try:
        dirs = sorted(tasks_root.glob("validate-*"), key=lambda p: p.name)
        if len(dirs) <= keep:
            return
        for d in dirs[:-keep]:
            shutil.rmtree(d, ignore_errors=True)
            log.info("Cleaned task dir: %s", d.name)
    except Exception:
        log.exception("Task cleanup failed (non-fatal)")

def _cleanup_orphaned_containers(max_age: int = 3600, max_containers: int = 100) -> None:
    try:
        r = subprocess.run(["docker", "ps", "-q", "--filter", "name=swe-eval-"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0 or not r.stdout.strip():
            return
        container_ids = r.stdout.strip().splitlines()
        if len(container_ids) > max_containers:
            log.warning("High container count: %d swe-eval containers running (limit %d)",
                        len(container_ids), max_containers)
        for cid in container_ids:
            ir = subprocess.run(["docker", "inspect", "--format", "{{.State.StartedAt}}", cid], capture_output=True, text=True, timeout=10)
            if ir.returncode != 0:
                continue
            started = datetime.fromisoformat(ir.stdout.strip().replace("Z", "+00:00"))
            age = (datetime.now(tz=UTC) - started).total_seconds()
            if age > max_age:
                subprocess.run(["docker", "kill", cid], capture_output=True, timeout=10)
                subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=10)
                log.info("Killed orphaned container %s (age %.0fs)", cid[:12], age)
    except Exception:
        log.exception("Container cleanup failed (non-fatal)")

def _count_patch_lines(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text().splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---")))

def _timestamp() -> str:
    return datetime.now(tz=UTC).isoformat()
