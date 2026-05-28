#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from config import RunConfig
from pipeline import compare_task_run, solve_task_run
from validate import (
    TaskPool,
    ValidatorState,
    _build_agent_config,
    _duel_agent_timeout,
    _pool_task_matches_king,
    _prepare_validate_paths,
)
from workspace import build_compare_paths, build_solution_paths, derive_compare_name, resolve_task_paths, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute current king solve cache with fp8 while preserving existing validator pool tasks.",
    )
    parser.add_argument("--workspace-root", type=Path, default=ROOT)
    parser.add_argument("--netuid", type=int, default=66)
    parser.add_argument("--solver-model", default="minimax/minimax-m2.7")
    parser.add_argument("--agent-timeout", type=int, default=1800)
    parser.add_argument("--provider", default="minimax/fp8")
    parser.add_argument("--pool", choices=("all", "primary", "retest"), default="all")
    parser.add_argument("--task", action="append", default=[], help="Specific task_name to recalc; repeatable.")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N unique tasks; 0 means all.")
    parser.add_argument("--concurrency", type=int, default=25, help="Parallel king solves to run.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    return RunConfig(
        workspace_root=args.workspace_root.resolve(),
        validate_netuid=args.netuid,
        solver_model=args.solver_model,
        agent_timeout=args.agent_timeout,
        solver_provider_only=args.provider,
        solver_provider_allow_fallbacks=False,
        docker_solver_max_output_bytes=100_000_000,
    )


def load_state(path: Path) -> ValidatorState:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ValidatorState.from_dict(payload)


def selected_pool_tasks(*, config: RunConfig, pool_name: str) -> list[Any]:
    paths = _prepare_validate_paths(config.validate_root)
    pools = []
    if pool_name in ("all", "primary"):
        pools.append(TaskPool(paths.pool_dir))
    if pool_name in ("all", "retest"):
        pools.append(TaskPool(paths.retest_pool_dir))

    tasks_by_name: dict[str, Any] = {}
    for pool in pools:
        for task in pool.list_tasks():
            tasks_by_name.setdefault(task.task_name, task)
    return [tasks_by_name[name] for name in sorted(tasks_by_name)]


def remove_king_artifacts(*, config: RunConfig, task_name: str) -> None:
    task_paths = resolve_task_paths(config.tasks_root, task_name)
    king_paths = build_solution_paths(task_paths, "king")
    shutil.rmtree(king_paths.root, ignore_errors=True)
    comparisons_dir = task_paths.root / "comparisons"
    if not comparisons_dir.exists():
        return
    for compare_dir in comparisons_dir.iterdir():
        if compare_dir.is_dir() and "king" in compare_dir.name:
            shutil.rmtree(compare_dir, ignore_errors=True)


def rewrite_pool_task(*, config: RunConfig, original_task: Any, refreshed_task: Any) -> None:
    paths = _prepare_validate_paths(config.validate_root)
    for pool_dir in (paths.pool_dir, paths.retest_pool_dir):
        path = pool_dir / f"{original_task.task_name}.json"
        if path.exists():
            write_json(path, refreshed_task.to_dict())


def recompute_task(*, config: RunConfig, king_config: RunConfig, task: Any, king: Any, dry_run: bool) -> dict[str, Any]:
    agent_timeout = _duel_agent_timeout(task)
    if dry_run:
        return {"task_name": task.task_name, "status": "dry_run", "agent_timeout": agent_timeout}

    remove_king_artifacts(config=config, task_name=task.task_name)
    solve_result = solve_task_run(
        task_name=task.task_name,
        solution_name="king",
        config=replace(king_config, agent_timeout=agent_timeout),
    )
    compare_result = compare_task_run(
        task_name=task.task_name,
        solution_names=["king", "baseline"],
        config=config,
    )
    refreshed = replace(
        task,
        king_lines=compare_result.matched_changed_lines,
        king_similarity=compare_result.similarity_ratio,
        baseline_lines=compare_result.total_changed_lines_b,
        agent_timeout_seconds=agent_timeout,
        king_hotkey=king.hotkey,
        king_commit_sha=king.commit_sha,
    )
    rewrite_pool_task(config=config, original_task=task, refreshed_task=refreshed)
    return {
        "task_name": task.task_name,
        "status": solve_result.exit_reason,
        "agent_timeout": agent_timeout,
        "king_lines": refreshed.king_lines,
        "king_similarity": refreshed.king_similarity,
    }


def main() -> int:
    args = parse_args()
    config = build_config(args)
    paths = _prepare_validate_paths(config.validate_root)
    state = load_state(paths.state_path)
    if state.current_king is None:
        raise SystemExit("No current king in validator state.")
    king = state.current_king
    requested_tasks = set(args.task or [])
    tasks = [task for task in selected_pool_tasks(config=config, pool_name=args.pool) if _pool_task_matches_king(task, king)]
    if requested_tasks:
        tasks = [task for task in tasks if task.task_name in requested_tasks]
        missing = requested_tasks - {task.task_name for task in tasks}
        if missing:
            raise SystemExit(f"requested task(s) are not in the selected current-king pool: {sorted(missing)}")
    if args.limit > 0:
        tasks = tasks[: args.limit]
    king_config = _build_agent_config(config, king)

    concurrency = max(1, int(args.concurrency))
    print(f"current king: uid={king.uid} ref={king.agent_ref}")
    print(f"provider: {config.solver_provider_only} fallbacks={config.solver_provider_allow_fallbacks}")
    print(f"tasks: {len(tasks)} pool={args.pool} dry_run={args.dry_run} concurrency={concurrency}")
    results = []
    failures = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {
            executor.submit(
                recompute_task,
                config=config,
                king_config=king_config,
                task=task,
                king=king,
                dry_run=args.dry_run,
            ): task
            for task in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # noqa: BLE001
                result = {"task_name": task.task_name, "status": "error", "error": str(exc)}
                failures.append(result)
            else:
                if result.get("status") not in {"completed", "dry_run"}:
                    failures.append(result)
            results.append(result)
            print(f"[{completed}/{len(tasks)}] {task.task_name}: {result.get('status')}", flush=True)
    print(json.dumps({"count": len(results), "failures": failures, "results": results}, indent=2, sort_keys=True))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
