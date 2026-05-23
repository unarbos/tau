#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from config import RunConfig  # noqa: E402
from validate import (  # noqa: E402
    TaskPool,
    ValidatorSubmission,
    _MIN_PATCH_LINES,
    _count_patch_lines,
    _load_state,
    _prepare_validate_paths,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate saved tasks and/or validator pool entries against the current on-disk contract."
    )
    parser.add_argument("--workspace-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--netuid", type=int, default=66)
    parser.add_argument(
        "--scope",
        choices=("saved", "main", "retest", "all"),
        default="all",
        help="Which task set to validate.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Optional limit on checked entries per scope.",
    )
    parser.add_argument(
        "--sample-errors",
        type=int,
        default=20,
        help="Number of failing examples to print.",
    )
    return parser.parse_args()


def _task_has_minimum_files(task_root: Path) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if not (task_root / "task" / "task.json").is_file():
        problems.append("missing task/task.json")
    if not (task_root / "task" / "task.txt").is_file():
        problems.append("missing task/task.txt")
    if not (task_root / "task" / "commit.json").is_file():
        problems.append("missing task/commit.json")
    reference_patch = task_root / "task" / "reference.patch"
    if not reference_patch.is_file():
        problems.append("missing task/reference.patch")
    elif _count_patch_lines(reference_patch) < _MIN_PATCH_LINES:
        problems.append("reference patch too small")
    if not (task_root / "task" / "original").exists():
        problems.append("missing task/original repo")
    if not (task_root / "task" / "reference").exists():
        problems.append("missing task/reference repo")
    return not problems, problems


def _king_ready_for_current(task_root: Path, king: ValidatorSubmission | None) -> tuple[bool, list[str]]:
    if king is None:
        return False, ["no current king"]
    problems: list[str] = []
    solve_json = task_root / "solutions" / "king" / "solve.json"
    solution_diff = task_root / "solutions" / "king" / "solution.diff"
    if not solve_json.is_file():
        problems.append("missing king solve.json")
    if not solution_diff.is_file():
        problems.append("missing king solution.diff")
    if problems:
        return False, problems
    try:
        payload = json.loads(solve_json.read_text())
    except Exception:
        return False, ["invalid king solve.json"]
    agent = str(payload.get("agent") or "")
    if agent != king.agent_ref:
        problems.append(f"king agent mismatch ({agent} != {king.agent_ref})")
    result = payload.get("result")
    exit_reason = str(result.get("exit_reason") or "") if isinstance(result, dict) else ""
    if exit_reason not in {"completed", "time_limit_exceeded"}:
        problems.append(f"bad king exit_reason {exit_reason or '<missing>'}")
    return not problems, problems


def _current_king(config: RunConfig) -> ValidatorSubmission | None:
    state = _load_state(_prepare_validate_paths(config.validate_root).state_path)
    if state.current_king is not None:
        return state.current_king
    if state.active_duel is not None:
        return state.active_duel.king
    return None


def _iter_saved_tasks(tasks_root: Path, limit: int | None) -> list[Path]:
    tasks = sorted(p for p in tasks_root.glob("validate-*") if p.is_dir())
    if limit is not None:
        tasks = tasks[: max(0, limit)]
    return tasks


def _iter_pool_entries(pool: TaskPool, limit: int | None) -> list[tuple[str, Path]]:
    tasks = pool.list_tasks()
    if limit is not None:
        tasks = tasks[: max(0, limit)]
    return [(task.task_name, Path(task.task_root)) for task in tasks]


def _print_summary(scope: str, counts: Counter[str], failures: list[dict[str, object]]) -> None:
    print(f"\n[{scope}]")
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")
    if failures:
        print("  sample_failures:")
        for item in failures:
            print(f"    - {item['name']}: {', '.join(item['problems'])}")


def main() -> int:
    args = _parse_args()
    config = RunConfig(workspace_root=args.workspace_root.resolve(), validate_netuid=args.netuid)
    validate_paths = _prepare_validate_paths(config.validate_root)
    current_king = _current_king(config)

    scopes = [args.scope] if args.scope != "all" else ["saved", "main", "retest"]
    overall_failures = 0

    for scope in scopes:
        counts: Counter[str] = Counter()
        failures: list[dict[str, object]] = []

        if scope == "saved":
            entries = [(task.name, task) for task in _iter_saved_tasks(config.tasks_root, args.limit)]
            for name, task_root in entries:
                counts["checked"] += 1
                ok_task, task_problems = _task_has_minimum_files(task_root)
                if ok_task:
                    counts["valid"] += 1
                    continue
                counts["invalid"] += 1
                overall_failures += 1
                if len(failures) < args.sample_errors:
                    failures.append({"name": name, "problems": task_problems})

        else:
            pool = TaskPool(validate_paths.pool_dir if scope == "main" else validate_paths.retest_pool_dir)
            entries = _iter_pool_entries(pool, args.limit)
            for name, task_root in entries:
                counts["checked"] += 1
                ok_task, task_problems = _task_has_minimum_files(task_root)
                ok_king, king_problems = _king_ready_for_current(task_root, current_king)
                if ok_task and ok_king:
                    counts["valid"] += 1
                    continue
                counts["invalid"] += 1
                overall_failures += 1
                if len(failures) < args.sample_errors:
                    failures.append({"name": name, "problems": task_problems + king_problems})

        _print_summary(scope, counts, failures)

    return 1 if overall_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
