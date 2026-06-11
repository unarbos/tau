"""Re-solve the current king on the existing pool tasks with the configured
solver model and rewrite each pool entry's king metrics in place.

Used when the solver model changes (e.g. swap to local DeepSeek V4 Flash):
the cached king solutions and the pool entries' ``king_lines`` /
``king_similarity`` were produced by the previous model and are stale, but the
task definitions (issue + reference.patch) are model-independent and must be
kept. This rebuilds the king solve + king/reference compare for each existing
pool task and updates the pool JSON, mirroring the pool-manager's own
solve -> compare -> PoolTask flow (see task_pool_manager._prepare_one_task_for_pool).

Run with the same upstream routing env the validator uses, e.g.:

  doppler run -p arbos -c dev -- env \
    OPENROUTER_UPSTREAM_BASE_URL=https://<tunnel> \
    OPENROUTER_API_KEY=sk-... SOLVER_PROVIDER_ONLY= OPENROUTER_PROVIDER_ONLY= \
    PYTHONPATH=src python scripts/resolve_king_pool.py --concurrency 12 --write
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import task_pool_manager as tpm
import validate as v
from config import RunConfig
from pipeline import compare_task_run, solve_task_run
from workspace import resolve_task_paths

_PRINT_LOCK = threading.Lock()


def _log(msg: str) -> None:
    with _PRINT_LOCK:
        print(msg, flush=True)


def resolve_one(
    *,
    task: v.PoolTask,
    config: RunConfig,
    king: Any,
    solve_semaphore: threading.Semaphore,
    write: bool,
    pool: v.TaskPool,
    agent_timeout_override: int = 0,
) -> dict[str, Any]:
    name = task.task_name
    try:
        resolve_task_paths(config.tasks_root, name)
    except FileNotFoundError:
        return {"task": name, "status": "missing_workspace"}

    agent_timeout = agent_timeout_override if agent_timeout_override > 0 else v._duel_agent_timeout(task)
    tpm.reset_solution_artifacts(task_name=name, solution_name="king", config=config)
    v._remove_compare_artifacts(
        task_name=name,
        solution_names=v._reference_compare_solution_names("king"),
        config=config,
    )
    king_cfg = replace(v._build_agent_config(config, king), agent_timeout=agent_timeout)

    t0 = time.monotonic()
    exit_reason = None
    try:
        with solve_semaphore:
            king_result = solve_task_run(task_name=name, solution_name="king", config=king_cfg)
        exit_reason = getattr(king_result, "exit_reason", None)
    except Exception as exc:  # noqa: BLE001
        return {"task": name, "status": "solve_failed", "error": str(exc)[:300]}

    try:
        v._remove_compare_artifacts(
            task_name=name,
            solution_names=v._reference_compare_solution_names("king"),
            config=config,
        )
        king_compare = compare_task_run(
            task_name=name,
            solution_names=v._reference_compare_solution_names("king"),
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        return {"task": name, "status": "compare_failed", "error": str(exc)[:300], "exit": exit_reason}

    new_task = v.PoolTask(
        task_name=name,
        task_root=task.task_root,
        creation_block=task.creation_block,
        cursor_elapsed=task.cursor_elapsed,
        king_lines=king_compare.matched_changed_lines,
        king_similarity=king_compare.similarity_ratio,
        baseline_lines=king_compare.total_changed_lines_b,
        agent_timeout_seconds=agent_timeout,
        king_hotkey=king.hotkey,
        king_commit_sha=king.commit_sha,
    )
    healthy, reason = v._pool_task_has_healthy_king_cache(config=config, task=new_task)
    out = {
        "task": name,
        "status": "ok" if healthy else "unhealthy",
        "reason": reason,
        "exit": exit_reason,
        "king_lines": new_task.king_lines,
        "king_similarity": round(float(new_task.king_similarity), 4),
        "baseline_lines": new_task.baseline_lines,
        "elapsed_s": round(time.monotonic() - t0, 1),
    }
    if healthy and write:
        pool.add(new_task)
        out["written"] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace-root", default="/home/const/subnet66/tau")
    ap.add_argument("--netuid", type=int, default=66)
    ap.add_argument("--solver-model", default="deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--pool", choices=["primary", "retest", "both"], default="both")
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="Only process the first N tasks (0 = all).")
    ap.add_argument("--write", action="store_true", help="Write updated pool entries (otherwise dry-run).")
    ap.add_argument("--agent-timeout", type=int, default=0, help="Override per-task king solve timeout (0 = use stored).")
    args = ap.parse_args()

    config = RunConfig(
        workspace_root=Path(args.workspace_root),
        validate_netuid=args.netuid,
        solver_model=args.solver_model,
    )
    king = tpm._load_manager_state(config).current_king
    if king is None:
        raise SystemExit("no current king in manager state")
    _log(f"current king: {king.agent_ref} hotkey={king.hotkey} sha={king.commit_sha[:12]}")
    _log(f"solver_model={config.solver_model} upstream-routed via OPENROUTER_UPSTREAM_BASE_URL env")

    paths = v._prepare_validate_paths(config.validate_root)
    pool_specs = []
    if args.pool in ("primary", "both"):
        pool_specs.append(("primary", v.TaskPool(paths.pool_dir)))
    if args.pool in ("retest", "both"):
        pool_specs.append(("retest", v.TaskPool(paths.retest_pool_dir)))

    solve_semaphore = threading.Semaphore(args.concurrency)
    grand_total = 0
    results_by_pool: dict[str, list[dict[str, Any]]] = {}
    for label, pool in pool_specs:
        tasks = pool.list_tasks()
        if args.limit > 0:
            tasks = tasks[: args.limit]
        if not tasks:
            _log(f"[{label}] pool empty, skipping")
            continue
        _log(f"[{label}] re-solving king on {len(tasks)} task(s) (concurrency={args.concurrency}, write={args.write})")
        results: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            futs = {
                ex.submit(
                    resolve_one,
                    task=t,
                    config=config,
                    king=king,
                    solve_semaphore=solve_semaphore,
                    write=args.write,
                    pool=pool,
                    agent_timeout_override=args.agent_timeout,
                ): t
                for t in tasks
            }
            done = 0
            for f in concurrent.futures.as_completed(futs):
                r = f.result()
                results.append(r)
                done += 1
                _log(
                    f"[{label}] {done}/{len(tasks)} {r['task']} -> {r['status']}"
                    + (f" lines={r.get('king_lines')} sim={r.get('king_similarity')} exit={r.get('exit')}" if r.get("status") in ("ok", "unhealthy") else "")
                    + (f" ({r.get('error')})" if r.get("error") else "")
                )
        results_by_pool[label] = results
        grand_total += len(results)

    _log("\n=== SUMMARY ===")
    for label, results in results_by_pool.items():
        from collections import Counter
        counts = Counter(r["status"] for r in results)
        written = sum(1 for r in results if r.get("written"))
        _log(f"[{label}] {dict(counts)} written={written}")
    Path("/tmp/resolve_king_pool_result.json").write_text(json.dumps(results_by_pool, indent=2, default=str) + "\n")
    _log("details -> /tmp/resolve_king_pool_result.json")


if __name__ == "__main__":
    main()
