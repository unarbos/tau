from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


def build_solve_spend_payload(
    *,
    tasks_root: Path,
    now: float | None = None,
    window_seconds: int = 86_400,
) -> dict[str, Any]:
    current_time = time.time() if now is None else float(now)
    cutoff = current_time - int(window_seconds)
    summaries = [
        summary
        for path in tasks_root.rglob("solve.json")
        for summary in [_solve_cost_summary(path, cutoff_epoch=cutoff)]
        if summary is not None
    ]
    total_cost = sum(item["cost"] for item in summaries)
    return {
        "window_seconds": int(window_seconds),
        "cutoff_epoch": cutoff,
        "solve_count": len(summaries),
        "total_cost_usd": round(total_cost, 6),
        "by_model_usd": _sum_by_key(summaries, "model"),
        "by_solution_prefix_usd": _sum_by_key(summaries, "solution_prefix"),
    }


def _solve_cost_summary(path: Path, *, cutoff_epoch: float) -> dict[str, Any] | None:
    try:
        modified_at = path.stat().st_mtime
    except OSError:
        return None
    if modified_at < cutoff_epoch:
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    result = payload.get("result")
    if not isinstance(result, dict):
        return None
    return {
        "cost": _float_value(result.get("cost")),
        "model": str(result.get("model") or payload.get("solver_model") or payload.get("model") or "unknown"),
        "solution_prefix": _solution_prefix(str(payload.get("solution_name") or path.parent.name)),
    }


def _float_value(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def _solution_prefix(solution_name: str) -> str:
    return solution_name.split("-", 1)[0] if "-" in solution_name else solution_name


def _sum_by_key(items: list[dict[str, Any]], key: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    for item in items:
        group = str(item.get(key) or "unknown")
        totals[group] = totals.get(group, 0.0) + float(item.get("cost") or 0)
    return {
        group: round(total, 6)
        for group, total in sorted(totals.items(), key=lambda entry: (-entry[1], entry[0]))
    }
