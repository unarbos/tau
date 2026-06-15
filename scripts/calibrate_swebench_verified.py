#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Any

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cli import _resolve_agent_source  # noqa: E402
from config import RunConfig  # noqa: E402
from docker_solver import _materialize_agent_source  # noqa: E402

DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"


def default_api_base() -> str:
    return (
        os.environ.get("OPENROUTER_UPSTREAM_BASE_URL")
        or os.environ.get("OPENROUTER_BASE_URL")
        or DEFAULT_API_BASE
    )


DEFAULT_MANIFEST = ROOT / "data" / "swebench_verified_sample_50_seed66.json"
_AGENT_MANIFEST = "tau_agent_files.json"


@dataclass(frozen=True, slots=True)
class CalibrationResult:
    instance_id: str
    repo: str
    elapsed_seconds: float
    cost: float | None
    success: bool
    exit_reason: str
    patch_bytes: int
    error: str | None


@dataclass(frozen=True, slots=True)
class LoadedAgent:
    module: ModuleType
    label: str
    agent_root: Path
    agent_file: Path
    commit_sha: str | None
    multi_file: bool
    source: dict[str, Any]


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_task_manifest(args)
    write_json(output_dir / "manifest.json", manifest)

    agent_raw = args.agent
    agent_ref = args.agent_ref
    if args.king_state is not None:
        king = load_current_king(args.king_state.expanduser().resolve())
        agent_raw = str(king.get("repo_full_name") or king.get("repo_url") or "")
        agent_ref = str(king["commit_sha"])

    loaded = resolve_and_load_agent(
        agent_raw=agent_raw,
        agent_ref=agent_ref,
        cache_dir=output_dir / "agent",
        cwd=Path.cwd(),
    )
    write_json(output_dir / "agent.json", loaded_source_payload(loaded))
    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY is required unless --api-key is supplied.")

    instances = load_instances(manifest)
    predictions_path = output_dir / "predictions.jsonl"
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    predictions_path.unlink(missing_ok=True)
    results_path.unlink(missing_ok=True)

    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                run_instance,
                agent=loaded,
                instance=instance,
                output_dir=output_dir,
                model=args.model,
                api_base=args.api_base,
                api_key=api_key,
            )
            for instance in instances
        ]
        for future in as_completed(futures):
            result, prediction = future.result()
            append_jsonl(results_path, asdict(result))
            append_jsonl(predictions_path, prediction)
            print(
                json.dumps(
                    {
                        "instance_id": result.instance_id,
                        "elapsed_seconds": round(result.elapsed_seconds, 3),
                        "cost": result.cost,
                        "success": result.success,
                        "exit_reason": result.exit_reason,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    results = [json.loads(line) for line in results_path.read_text(encoding="utf-8").splitlines()]
    summary = summarize_results(results, total_elapsed_seconds=time.monotonic() - start)
    summary["agent"] = loaded_source_payload(loaded)
    summary["manifest"] = manifest
    if args.score:
        summary["score"] = score_predictions(
            predictions_path=predictions_path,
            manifest=manifest,
            output_dir=output_dir,
            max_workers=args.scoring_workers,
            run_id=f"calibrate-{output_dir.name}",
        )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an agent on a SWE-bench Verified sample without Docker.")
    parser.add_argument(
        "--agent",
        help=(
            "Agent source: local agent.py, directory with agent.py (multi-file ok), "
            "GitHub shorthand like org/repo@commit, or "
            "https://github.com/org/repo/commit/<sha>."
        ),
    )
    parser.add_argument(
        "--king-state",
        type=Path,
        help="Use current_king from validator state.json instead of --agent.",
    )
    parser.add_argument(
        "--agent-ref",
        help="Optional git commit/ref for a GitHub --agent source without an embedded @commit.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="Pinned SWE-bench instance manifest JSON (default: data/swebench_verified_sample_50_seed66.json).",
    )
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for repos and JSONL outputs.")
    parser.add_argument("--count", type=int, default=50, help="Fallback slice size when --manifest is omitted.")
    parser.add_argument("--offset", type=int, default=0, help="Fallback dataset offset when --manifest is omitted.")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent agent workers.")
    parser.add_argument("--split", default="test", help="Fallback dataset split when --manifest is omitted.")
    parser.add_argument("--model", default=os.environ.get("SOLVER_MODEL") or os.environ.get("BASELINE_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--api-base", default=default_api_base())
    parser.add_argument("--api-key", help="API key override. Defaults to OPENROUTER_API_KEY.")
    parser.add_argument("--score", action="store_true", help="Run official SWE-bench harness scoring after patch generation.")
    parser.add_argument("--scoring-workers", type=int, default=8, help="Parallel workers for official scoring.")
    args = parser.parse_args()
    if args.agent is None and args.king_state is None:
        parser.error("one of --agent or --king-state is required")
    return args


def load_current_king(state_path: Path) -> dict[str, Any]:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    king = payload.get("current_king") if isinstance(payload, dict) else None
    if not isinstance(king, dict) or not king.get("commit_sha"):
        raise SystemExit(f"No current_king.commit_sha found in {state_path}")
    return king


def load_task_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.manifest is None:
        return {
            "dataset_name": DATASET_NAME,
            "split": args.split,
            "count": args.count,
            "offset": args.offset,
            "instance_ids": None,
        }
    manifest_path = args.manifest.expanduser().resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    instance_ids = [str(item) for item in payload.get("instance_ids", ())]
    if not instance_ids:
        raise SystemExit(f"manifest has no instance_ids: {manifest_path}")
    return {
        "dataset_name": str(payload.get("dataset_name") or DATASET_NAME),
        "split": str(payload.get("split") or "test"),
        "count": len(instance_ids),
        "seed": payload.get("seed"),
        "manifest_path": str(manifest_path),
        "instance_ids": instance_ids,
    }


def load_instances(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    instance_ids = manifest.get("instance_ids")
    if not instance_ids:
        offset = int(manifest.get("offset") or 0)
        count = int(manifest.get("count") or 50)
        return list(load_dataset(manifest["dataset_name"], split=manifest["split"]).select(range(offset, offset + count)))
    selected = set(instance_ids)
    rows = [
        dict(row)
        for row in load_dataset(manifest["dataset_name"], split=manifest["split"])
        if row["instance_id"] in selected
    ]
    by_id = {str(row["instance_id"]): row for row in rows}
    missing = [item for item in instance_ids if item not in by_id]
    if missing:
        raise RuntimeError(f"SWE-bench dataset missing manifest ids: {', '.join(missing)}")
    return [by_id[item] for item in instance_ids]


def score_predictions(
    *,
    predictions_path: Path,
    manifest: dict[str, Any],
    output_dir: Path,
    max_workers: int,
    run_id: str,
) -> dict[str, Any]:
    instance_ids = manifest.get("instance_ids") or []
    scoring_dir = output_dir / "official_scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        str(manifest["dataset_name"]),
        "--split",
        str(manifest["split"]),
        "--predictions_path",
        str(predictions_path),
        "--max_workers",
        str(max_workers),
        "--run_id",
        run_id,
        "--instance_ids",
        *instance_ids,
    ]
    result = subprocess.run(command, cwd=scoring_dir, capture_output=True, text=True, check=False)
    (scoring_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (scoring_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        return {
            "resolved_count": None,
            "total_count": len(instance_ids),
            "pass_rate": None,
            "error": f"official scoring failed exit={result.returncode}",
        }
    return extract_score_summary(scoring_dir=scoring_dir, total_count=len(instance_ids))


def extract_score_summary(*, scoring_dir: Path, total_count: int) -> dict[str, Any]:
    candidates = sorted(scoring_dir.rglob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        resolved_ids = resolved_ids_from_report(payload)
        if resolved_ids is not None:
            resolved_count = len(resolved_ids)
            return {
                "resolved_count": resolved_count,
                "total_count": total_count,
                "pass_rate": resolved_count / total_count if total_count else None,
                "report_path": str(path),
                "resolved_ids": sorted(resolved_ids),
            }
    return {
        "resolved_count": None,
        "total_count": total_count,
        "pass_rate": None,
        "error": "could not find scoring report",
    }


def resolved_ids_from_report(payload: Any) -> set[str] | None:
    if not isinstance(payload, dict):
        return None
    if "resolved" in payload and isinstance(payload["resolved"], list):
        return {str(item) for item in payload["resolved"]}
    if "resolved_ids" in payload and isinstance(payload["resolved_ids"], list):
        return {str(item) for item in payload["resolved_ids"]}
    submitted = payload.get("submitted")
    if isinstance(submitted, list):
        resolved = {str(row.get("instance_id")) for row in submitted if isinstance(row, dict) and row.get("resolved")}
        if resolved:
            return resolved
    return None


def resolve_and_load_agent(*, agent_raw: str, agent_ref: str | None, cache_dir: Path, cwd: Path) -> LoadedAgent:
    source = _resolve_agent_source(agent_raw, cwd=cwd)
    if agent_ref:
        if source.kind != "github_repo":
            raise SystemExit("--agent-ref requires a GitHub repo --agent source")
        source = replace(source, commit_sha=agent_ref)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    config = RunConfig(solver_agent_source=source)
    agent_root, agent_file = _materialize_agent_source(config=config, target_dir=cache_dir / "materialized")
    commit_sha = resolve_agent_commit_sha(source=source, agent_root=agent_root)
    multi_file = agent_is_multi_file(agent_root)
    module = load_agent_module(agent_root=agent_root, agent_file=agent_file)
    label = agent_label(source=source, commit_sha=commit_sha)
    return LoadedAgent(
        module=module,
        label=label,
        agent_root=agent_root,
        agent_file=agent_file,
        commit_sha=commit_sha,
        multi_file=multi_file,
        source=source.to_dict(),
    )


def resolve_agent_commit_sha(*, source, agent_root: Path) -> str | None:
    if source.commit_sha:
        return str(source.commit_sha)
    git_dir = agent_root / ".git"
    if git_dir.is_dir():
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=agent_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            return result.stdout.strip()
    return None


def agent_is_multi_file(agent_root: Path) -> bool:
    manifest = agent_root / _AGENT_MANIFEST
    if manifest.is_file():
        return True
    py_files = [path for path in agent_root.rglob("*.py") if "__pycache__" not in path.parts]
    return len(py_files) > 1


def agent_label(*, source, commit_sha: str | None) -> str:
    if commit_sha:
        base = source.raw.split("@", 1)[0]
        return f"{base}@{commit_sha[:12]}"
    return source.raw


def loaded_source_payload(loaded: LoadedAgent) -> dict[str, Any]:
    return {
        "label": loaded.label,
        "agent_root": str(loaded.agent_root),
        "agent_file": str(loaded.agent_file),
        "commit_sha": loaded.commit_sha,
        "multi_file": loaded.multi_file,
        "source": loaded.source,
    }


def load_agent_module(*, agent_root: Path, agent_file: Path) -> ModuleType:
    agent_root_str = str(agent_root.resolve())
    if agent_root_str not in sys.path:
        sys.path.insert(0, agent_root_str)
    spec = importlib.util.spec_from_file_location("calibration_agent", str(agent_file.resolve()))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load agent module from {agent_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "solve"):
        raise RuntimeError(f"{agent_file} does not define solve(...)")
    return module


def run_instance(
    *,
    agent: LoadedAgent,
    instance: dict[str, Any],
    output_dir: Path,
    model: str,
    api_base: str,
    api_key: str,
) -> tuple[CalibrationResult, dict[str, str]]:
    instance_id = str(instance["instance_id"])
    repo_name = str(instance["repo"])
    repo_dir = output_dir / "repos" / instance_id
    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    error: str | None = None
    payload: dict[str, Any] = {}

    try:
        prepare_repo(repo_name=repo_name, base_commit=str(instance["base_commit"]), repo_dir=repo_dir)
        payload = agent.module.solve(
            repo_path=str(repo_dir),
            issue=str(instance["problem_statement"]),
            model=model,
            api_base=api_base,
            api_key=api_key,
        )
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)

    elapsed = time.monotonic() - start
    patch = str(payload.get("patch") or "") if isinstance(payload, dict) else ""
    logs = str(payload.get("logs") or "") if isinstance(payload, dict) else ""
    (logs_dir / f"{instance_id}.log").write_text(logs + ("\n" if logs else ""), encoding="utf-8")
    result = CalibrationResult(
        instance_id=instance_id,
        repo=repo_name,
        elapsed_seconds=elapsed,
        cost=payload.get("cost") if isinstance(payload.get("cost"), (int, float)) else None,
        success=bool(payload.get("success")) if isinstance(payload, dict) else False,
        exit_reason="completed" if error is None else "error",
        patch_bytes=len(patch.encode("utf-8")),
        error=error,
    )
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": agent.label,
        "model_patch": patch,
    }
    return result, prediction


def prepare_repo(*, repo_name: str, base_commit: str, repo_dir: Path) -> None:
    if repo_dir.exists():
        return
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = repo_dir.with_name(f"{repo_dir.name}.tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    run(["git", "clone", "--quiet", "--no-tags", f"https://github.com/{repo_name}.git", str(temp_dir)])
    run(["git", "checkout", "--quiet", base_commit], cwd=temp_dir)
    temp_dir.rename(repo_dir)


def run(command: list[str], *, cwd: Path | None = None) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def summarize_results(results: list[dict[str, Any]], *, total_elapsed_seconds: float) -> dict[str, Any]:
    elapsed = sorted(float(row["elapsed_seconds"]) for row in results)
    costs = [float(row["cost"]) for row in results if isinstance(row.get("cost"), (int, float))]
    return {
        "count": len(results),
        "completed": sum(1 for row in results if row.get("exit_reason") == "completed"),
        "errors": sum(1 for row in results if row.get("exit_reason") != "completed"),
        "successes": sum(1 for row in results if row.get("success")),
        "total_elapsed_seconds": total_elapsed_seconds,
        "sum_task_elapsed_seconds": sum(elapsed),
        "min_task_seconds": elapsed[0] if elapsed else None,
        "median_task_seconds": percentile(elapsed, 0.5),
        "p90_task_seconds": percentile(elapsed, 0.9),
        "max_task_seconds": elapsed[-1] if elapsed else None,
        "total_cost": sum(costs) if costs else None,
        "mean_cost": (sum(costs) / len(costs)) if costs else None,
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    index = min(len(values) - 1, max(0, round((len(values) - 1) * fraction)))
    return values[index]


if __name__ == "__main__":
    main()
