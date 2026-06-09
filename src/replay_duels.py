"""Replay the LLM diff judge on rollout snapshot pairs.

Reads king+challenger rollout pairs from a snapshot directory, re-runs the
judge (optionally with a different/cheaper model), and reports agreement with
the original stored outcomes.

Usage:
    python replay_duels.py \\
        --rollouts-dir /path/to/rollouts/2026-06-08-09 \\
        --tasks-dir    /path/to/tasks \\
        --judge-model  gemma-3-4b-it \\
        --limit        10 \\
        --concurrency  4
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Allow running from the repo root or src/ directly.
sys.path.insert(0, str(Path(__file__).parent))

from config import RunConfig
from validate import _judge_diffs_direct

log = logging.getLogger("replay_duels")

_DEFAULT_JUDGE_MODEL = "gemma-3-4b-it"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_rollout_pairs(rollouts_dir: Path) -> list[tuple[dict, dict]]:
    """Return (king_record, challenger_record) pairs from .jsonl.gz files."""
    pairs: list[tuple[dict, dict]] = []
    for path in sorted(rollouts_dir.glob("*.jsonl.gz")):
        try:
            with gzip.open(path, "rt", encoding="utf-8") as fh:
                records = [json.loads(line) for line in fh if line.strip()]
        except Exception as exc:
            log.warning("Skipping %s: %s", path.name, exc)
            continue
        king = next((r for r in records if r.get("role") == "king"), None)
        challenger = next((r for r in records if r.get("role") == "challenger"), None)
        if king and challenger:
            pairs.append((king, challenger))
        else:
            log.debug("No king+challenger pair in %s", path.name)
    return pairs


def _load_task_artifact_index(tasks_dir: Path) -> dict[str, dict[str, str]]:
    """Build {task_name: {artifact_path: content}} from task archive .jsonl files.

    Used to supply reference patches that aren't stored in rollout records.
    """
    index: dict[str, dict[str, str]] = {}
    for tasks_file in sorted(tasks_dir.rglob("*.jsonl")):
        try:
            with open(tasks_file, encoding="utf-8") as fh:
                lines = fh.readlines()
        except Exception as exc:
            log.warning("Skipping task file %s: %s", tasks_file, exc)
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            task_name = row.get("task_name")
            if not task_name:
                continue
            arts: dict[str, str] = {}
            for a in row.get("artifacts", []):
                if isinstance(a, dict) and isinstance(a.get("content"), str):
                    arts[a["path"]] = a["content"]
            index[task_name] = arts
    return index


# ---------------------------------------------------------------------------
# Per-pair replay
# ---------------------------------------------------------------------------

def _replay_pair(
    king: dict,
    challenger: dict,
    task_index: dict[str, dict[str, str]],
    config: RunConfig,
) -> dict:
    task_name: str = king["task_name"]
    challenger_name: str = challenger.get("solution_name") or "challenger"
    king_patch: str = king.get("final_patch") or ""
    challenger_patch: str = challenger.get("final_patch") or ""
    task_prompt: str = king.get("issue") or ""

    arts = task_index.get(task_name, {})
    reference_patch: str = arts.get("task/reference.patch", "")
    if not reference_patch:
        log.debug("No reference patch available for task %s", task_name)

    result = _judge_diffs_direct(
        task_name=task_name,
        challenger_solution_name=challenger_name,
        task_prompt=task_prompt,
        reference_patch=reference_patch,
        king_patch=king_patch,
        challenger_patch=challenger_patch,
        config=config,
    )

    original: dict = king.get("judge") or {}
    original_winner = original.get("winner")
    agreed = (result.winner == original_winner) if original_winner else None

    return {
        "task_name": task_name,
        "duel_id": king.get("duel_id"),
        "original_winner": original_winner,
        "original_model": original.get("model"),
        "replay_winner": result.winner,
        "replay_model": result.model,
        "replay_king_score": result.king_score,
        "replay_challenger_score": result.challenger_score,
        "replay_rationale": result.rationale,
        "replay_error": result.error,
        "agreed": agreed,
        "has_reference_patch": bool(reference_patch),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_config(args: argparse.Namespace) -> RunConfig:
    api_key = args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY") or ""
    return RunConfig(
        openrouter_api_key=api_key or None,
        validate_judge_model=args.judge_model,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--rollouts-dir",
        required=True,
        type=Path,
        help="Directory containing *.jsonl.gz rollout files (one duel per file).",
    )
    p.add_argument(
        "--tasks-dir",
        type=Path,
        default=None,
        help="Optional directory containing task archive *.jsonl files (for reference patches).",
    )
    p.add_argument(
        "--judge-model",
        default=_DEFAULT_JUDGE_MODEL,
        help=f"OpenRouter model to use as judge (default: {_DEFAULT_JUDGE_MODEL}).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of pairs to replay (default: all).",
    )
    p.add_argument(
        "--concurrency",
        type=int,
        default=4,
        help="Number of parallel judge calls (default: 4).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write JSONL results.",
    )
    p.add_argument(
        "--openrouter-api-key",
        default=None,
        help="OpenRouter API key (overrides OPENROUTER_API_KEY env var).",
    )
    p.add_argument("--debug", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config = _build_config(args)
    if not config.openrouter_api_key:
        log.error("OPENROUTER_API_KEY is not set — set it via env or --openrouter-api-key")
        return 1

    rollouts_dir = args.rollouts_dir
    if not rollouts_dir.is_dir():
        log.error("--rollouts-dir %s does not exist", rollouts_dir)
        return 1

    log.info("Loading rollout pairs from %s", rollouts_dir)
    pairs = _load_rollout_pairs(rollouts_dir)
    log.info("Found %d rollout pair(s)", len(pairs))

    task_index: dict[str, dict[str, str]] = {}
    if args.tasks_dir:
        log.info("Indexing task artifacts from %s", args.tasks_dir)
        task_index = _load_task_artifact_index(args.tasks_dir)
        log.info("Indexed %d task(s) with artifacts", len(task_index))

    if args.limit:
        pairs = pairs[: args.limit]
        log.info("Limiting to %d pair(s)", len(pairs))

    log.info("Judge model: %s  concurrency: %d", args.judge_model, args.concurrency)

    results: list[dict] = [{}] * len(pairs)
    lock = threading.Lock()
    done = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        future_to_idx = {
            pool.submit(_replay_pair, king, chall, task_index, config): i
            for i, (king, chall) in enumerate(pairs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                result = future.result()
            except Exception as exc:
                king, _ = pairs[idx]
                result = {
                    "task_name": king.get("task_name"),
                    "duel_id": king.get("duel_id"),
                    "replay_error": str(exc),
                    "agreed": None,
                }
            results[idx] = result
            with lock:
                done += 1
                agreed_sym = (
                    "✓" if result.get("agreed") else
                    "✗" if result.get("agreed") is False else
                    "?"
                )
                err = f"  ERROR: {result.get('replay_error')}" if result.get("replay_error") else ""
                log.info(
                    "[%d/%d] %s task=%s orig=%s replay=%s%s",
                    done, len(pairs),
                    agreed_sym,
                    result.get("task_name", "?"),
                    result.get("original_winner", "?"),
                    result.get("replay_winner", "?"),
                    err,
                )

    # Summary
    with_original = [r for r in results if r.get("original_winner")]
    agreed = [r for r in with_original if r.get("agreed")]
    errors = [r for r in results if r.get("replay_error")]

    print()
    print("=" * 60)
    print(f"Replay summary  judge={args.judge_model}")
    print("=" * 60)
    print(f"  Pairs replayed : {len(results)}")
    print(f"  With original  : {len(with_original)}")
    if with_original:
        pct = 100.0 * len(agreed) / len(with_original)
        print(f"  Agreement      : {len(agreed)}/{len(with_original)} ({pct:.1f}%)")
    print(f"  Errors         : {len(errors)}")

    winner_breakdown: dict[str, dict[str, int]] = {}
    for r in with_original:
        orig = r.get("original_winner", "?")
        rep = r.get("replay_winner", "?")
        winner_breakdown.setdefault(orig, {}).setdefault(rep, 0)
        winner_breakdown[orig][rep] += 1
    if winner_breakdown:
        print()
        print("  Original → Replay winner breakdown:")
        for orig_w, replay_counts in sorted(winner_breakdown.items()):
            for rep_w, cnt in sorted(replay_counts.items()):
                mark = "✓" if orig_w == rep_w else "✗"
                print(f"    {orig_w:12s} → {rep_w:12s}  {cnt:3d}  {mark}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as fh:
            for r in results:
                fh.write(json.dumps(r, default=str) + "\n")
        log.info("Results written to %s", args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
