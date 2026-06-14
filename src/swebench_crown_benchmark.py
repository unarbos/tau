from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config import RunConfig, SolverAgentSource
from docker_solver import solve_task_in_docker
from openrouter_proxy import SolveUsageSummary
from task_generation import GeneratedTask
from workspace import git_diff

log = logging.getLogger("swe-eval.swebench_crown_benchmark")

DEFAULT_DATASET_NAME = "princeton-nlp/SWE-bench_Verified"
DEFAULT_SPLIT = "test"
DEFAULT_MODEL = "google/gemini-3.1-flash-lite"
DEFAULT_PROVIDER_ONLY = "google-ai-studio"
DEFAULT_PI_REPO_URL = "https://github.com/earendil-works/pi"
DEFAULT_MINI_SWE_AGENT_REPO_URL = "https://github.com/SWE-agent/mini-swe-agent"
DEFAULT_BASELINE = "mini-swe-agent"
DEFAULT_POLL_INTERVAL_SECONDS = 60
DEFAULT_WORKERS = 20


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    dataset_name: str
    split: str
    seed: int
    instance_ids: tuple[str, ...]
    manifest_hash: str

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["instance_ids"] = list(self.instance_ids)
        return payload


@dataclass(frozen=True, slots=True)
class AgentIdentity:
    name: str
    repo_url: str
    commit_sha: str
    agent_path: Path
    source: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["agent_path"] = str(self.agent_path)
        return payload


@dataclass(frozen=True, slots=True)
class SolveRecord:
    agent_name: str
    instance_id: str
    repo: str
    elapsed_seconds: float
    patch_bytes: int
    success: bool
    exit_reason: str
    error: str | None
    usage_summary: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ScoreSummary:
    resolved_count: int | None
    total_count: int
    pass_rate: float | None
    report_path: str | None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def utc_now() -> str:
    return datetime.now(tz=UTC).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def load_manifest(path: Path) -> BenchmarkManifest:
    payload = read_json(path)
    instance_ids = tuple(str(item) for item in payload.get("instance_ids", ()))
    if not instance_ids:
        raise ValueError(f"manifest has no instance_ids: {path}")
    duplicates = sorted({item for item in instance_ids if instance_ids.count(item) > 1})
    if duplicates:
        raise ValueError(f"manifest has duplicate instance_ids: {', '.join(duplicates)}")
    expected_count = payload.get("count")
    if isinstance(expected_count, int) and expected_count != len(instance_ids):
        raise ValueError(f"manifest count={expected_count} but has {len(instance_ids)} ids")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return BenchmarkManifest(
        dataset_name=str(payload.get("dataset_name") or DEFAULT_DATASET_NAME),
        split=str(payload.get("split") or DEFAULT_SPLIT),
        seed=int(payload.get("seed") or 0),
        instance_ids=instance_ids,
        manifest_hash=digest,
    )


def current_king_from_state(state_path: Path) -> dict[str, Any] | None:
    if not state_path.exists():
        return None
    payload = read_json(state_path)
    king = payload.get("current_king") if isinstance(payload, dict) else None
    return king if isinstance(king, dict) and king.get("commit_sha") else None


def latest_completed_commit(benchmark_root: Path) -> str | None:
    latest_path = benchmark_root / "latest.json"
    if not latest_path.exists():
        return None
    payload = read_json(latest_path)
    commit = payload.get("king_commit_sha") if isinstance(payload, dict) else None
    return str(commit) if commit else None


def should_benchmark_king(*, king: dict[str, Any] | None, benchmark_root: Path) -> bool:
    if not king:
        return False
    commit = str(king.get("commit_sha") or "")
    if not commit:
        return False
    job_dir = benchmark_root / commit
    job_path = job_dir / "job.json"
    if not job_path.exists():
        return True
    return not completed_swebench_job_valid(job_dir)


def completed_swebench_job_valid(job_dir: Path) -> bool:
    job = read_json(job_dir / "job.json")
    if not (isinstance(job, dict) and job.get("status") == "completed"):
        return False
    comparison_path = job_dir / "comparison.json"
    if not comparison_path.exists():
        return False
    comparison = read_json(comparison_path)
    baseline_name = str(comparison.get("baseline_name") or "pi") if isinstance(comparison, dict) else "pi"
    return predictions_include_patch(job_dir / baseline_name / "predictions.jsonl")


def queue_latest(current_pending: dict[str, Any] | None, candidate: dict[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return current_pending
    if current_pending is None:
        return candidate
    return candidate if candidate.get("commit_sha") != current_pending.get("commit_sha") else current_pending


def run_daemon(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.manifest)
    benchmark_root = args.validate_root / "benchmarks" / "swebench-verified"
    pending: dict[str, Any] | None = None
    log.info("Starting SWE-bench crown benchmark daemon root=%s", benchmark_root)
    while True:
        try:
            king = current_king_from_state(args.state_path)
            if should_benchmark_king(king=king, benchmark_root=benchmark_root):
                pending = queue_latest(pending, king)
            if pending is not None:
                king_to_run = pending
                pending = None
                run_crown_benchmark(
                    king=king_to_run,
                    manifest=manifest,
                    args=args,
                    benchmark_root=benchmark_root,
                )
        except Exception:
            log.exception("SWE-bench crown benchmark loop failed")
        time.sleep(args.poll_interval_seconds)


def run_once(args: argparse.Namespace) -> dict[str, Any]:
    manifest = load_manifest(args.manifest)
    king = current_king_from_state(args.state_path)
    if king is None:
        raise RuntimeError(f"No current king found in {args.state_path}")
    benchmark_root = args.validate_root / "benchmarks" / "swebench-verified"
    return run_crown_benchmark(king=king, manifest=manifest, args=args, benchmark_root=benchmark_root)


def run_crown_benchmark(
    *,
    king: dict[str, Any],
    manifest: BenchmarkManifest,
    args: argparse.Namespace,
    benchmark_root: Path,
) -> dict[str, Any]:
    start = time.monotonic()
    started_at = utc_now()
    king_commit = str(king["commit_sha"])
    job_dir = benchmark_root / king_commit
    job_dir.mkdir(parents=True, exist_ok=True)
    write_json(job_dir / "manifest.json", manifest.to_dict())
    write_job(job_dir, status="running", king=king, manifest=manifest, started_at=started_at)

    try:
        instances = load_instances(manifest)
        king_agent = resolve_king_agent(king=king, cache_root=job_dir / "agents" / "king")
        baseline_agent = resolve_pinned_baseline_agent(
            baseline=args.baseline,
            benchmark_root=benchmark_root,
            repo_url=baseline_repo_url(args),
            ref=baseline_ref(args),
            cache_root=job_dir / "agents" / baseline_agent_name(args.baseline),
        )
        cache_dir = baseline_cache_dir(
            benchmark_root=benchmark_root,
            baseline_agent=baseline_agent,
            manifest=manifest,
            model=args.model,
            provider_only=args.provider_only,
        )
        baseline_cached = restore_baseline(cache_dir=cache_dir, job_dir=job_dir, baseline_name=baseline_agent.name, skip_scoring=args.skip_scoring)
        if not baseline_cached:
            write_job(
                job_dir,
                status="generating_baseline_predictions",
                king=king,
                manifest=manifest,
                started_at=started_at,
                agents=[baseline_agent.to_dict()],
            )
            generate_predictions(
                agents=(baseline_agent,),
                instances=instances,
                job_dir=job_dir,
                model=args.model,
                provider_only=args.provider_only,
                openrouter_api_key=args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY"),
                api_base=args.api_base,
                workers=args.workers,
            )
            write_job(job_dir, status="scoring_baseline", king=king, manifest=manifest, started_at=started_at)
            score_predictions(agent=baseline_agent, manifest=manifest, job_dir=job_dir, max_workers=args.workers, skip=args.skip_scoring)
            save_baseline(cache_dir=cache_dir, job_dir=job_dir, baseline_name=baseline_agent.name)
        write_job(
            job_dir,
            status="generating_king_predictions",
            king=king,
            manifest=manifest,
            started_at=started_at,
            agents=[king_agent.to_dict(), baseline_agent.to_dict()],
        )
        generate_predictions(
            agents=(king_agent,),
            instances=instances,
            job_dir=job_dir,
            model=args.model,
            provider_only=args.provider_only,
            openrouter_api_key=args.openrouter_api_key or os.environ.get("OPENROUTER_API_KEY"),
            api_base=args.api_base,
            workers=args.workers,
        )
        write_job(job_dir, status="scoring", king=king, manifest=manifest, started_at=started_at)
        king_score = score_predictions(agent=king_agent, manifest=manifest, job_dir=job_dir, max_workers=args.workers, skip=args.skip_scoring)
        baseline_score = read_score_summary(job_dir / baseline_agent.name / "score_summary.json")
        comparison = build_comparison(
            king=king,
            king_agent=king_agent,
            baseline_agent=baseline_agent,
            manifest=manifest,
            job_dir=job_dir,
            scores=(king_score, baseline_score),
            total_elapsed_seconds=time.monotonic() - start,
            model=args.model,
            provider_only=args.provider_only,
            baseline_cached=baseline_cached,
        )
        write_json(job_dir / "comparison.json", comparison)
        write_json(benchmark_root / "latest.json", comparison)
        write_history_entry(benchmark_root / "history.jsonl", comparison)
        merge_benchmark_into_dashboard(args.validate_root / "dashboard_data.json", comparison)
        publish_benchmark_payload(comparison)
        write_job(job_dir, status="completed", king=king, manifest=manifest, started_at=comparison["started_at"], comparison=comparison)
        return comparison
    except Exception as exc:
        error_payload = {"status": "failed", "error": repr(exc), "finished_at": utc_now(), "king": king}
        write_json(job_dir / "job.json", error_payload)
        merge_benchmark_into_dashboard(
            args.validate_root / "dashboard_data.json",
            benchmark_dashboard_payload(error_payload),
        )
        raise


def write_job(
    job_dir: Path,
    *,
    status: str,
    king: dict[str, Any],
    manifest: BenchmarkManifest,
    started_at: str,
    agents: list[dict[str, Any]] | None = None,
    comparison: dict[str, Any] | None = None,
) -> None:
    payload = {
        "status": status,
        "started_at": started_at,
        "updated_at": utc_now(),
        "king": king,
        "manifest": manifest.to_dict(),
    }
    if agents is not None:
        payload["agents"] = agents
    if comparison is not None:
        payload["comparison"] = comparison
    write_json(job_dir / "job.json", payload)


def load_instances(manifest: BenchmarkManifest) -> list[dict[str, Any]]:
    from datasets import load_dataset

    selected = set(manifest.instance_ids)
    rows = [
        dict(row)
        for row in load_dataset(manifest.dataset_name, split=manifest.split)
        if row["instance_id"] in selected
    ]
    by_id = {str(row["instance_id"]): row for row in rows}
    missing = [instance_id for instance_id in manifest.instance_ids if instance_id not in by_id]
    if missing:
        raise RuntimeError(f"SWE-bench dataset missing manifest ids: {', '.join(missing)}")
    return [by_id[instance_id] for instance_id in manifest.instance_ids]


def resolve_king_agent(*, king: dict[str, Any], cache_root: Path) -> AgentIdentity:
    repo_url = str(king.get("repo_url") or f"https://github.com/{king['repo_full_name']}.git")
    commit_sha = str(king["commit_sha"])
    return resolve_repo_agent(
        name="king",
        repo_url=repo_url,
        ref=commit_sha,
        cache_root=cache_root,
        source=king,
    )


def resolve_repo_agent(
    *,
    name: str,
    repo_url: str,
    ref: str,
    cache_root: Path,
    source: dict[str, Any],
) -> AgentIdentity:
    repo_dir, commit_sha = checkout_repo(repo_url=repo_url, ref=ref, cache_root=cache_root)
    agent_path = repo_dir / "agent.py"
    if not agent_path.exists():
        raise FileNotFoundError(f"{name} repo does not contain agent.py: {agent_path}")
    return AgentIdentity(name=name, repo_url=repo_url, commit_sha=commit_sha, agent_path=agent_path, source=source)


def checkout_repo(*, repo_url: str, ref: str, cache_root: Path) -> tuple[Path, str]:
    repo_dir = cache_root / "repo"
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--quiet", "--no-tags", repo_url, str(repo_dir)], cwd=None, timeout=600)
    run(["git", "fetch", "--quiet", "origin", ref], cwd=repo_dir, timeout=300, check=False)
    run(["git", "checkout", "--quiet", ref], cwd=repo_dir, timeout=300)
    commit_sha = run(["git", "rev-parse", "HEAD"], cwd=repo_dir, timeout=60).stdout.strip()
    return repo_dir, commit_sha


def resolve_pi_repo_agent(
    *,
    repo_url: str,
    ref: str,
    cache_root: Path,
    source: dict[str, Any],
) -> AgentIdentity:
    repo_dir, commit_sha = checkout_repo(repo_url=repo_url, ref=ref, cache_root=cache_root)
    package_path = repo_dir / "packages" / "coding-agent" / "package.json"
    if not package_path.exists():
        raise FileNotFoundError(f"pi repo does not contain packages/coding-agent/package.json: {package_path}")
    return AgentIdentity(name="pi", repo_url=repo_url, commit_sha=commit_sha, agent_path=repo_dir, source=source)


def resolve_mini_swe_agent_repo_agent(
    *,
    repo_url: str,
    ref: str,
    cache_root: Path,
    source: dict[str, Any],
) -> AgentIdentity:
    repo_dir, commit_sha = checkout_repo(repo_url=repo_url, ref=ref, cache_root=cache_root)
    if not (repo_dir / "pyproject.toml").exists():
        raise FileNotFoundError(f"mini-swe-agent repo does not contain pyproject.toml: {repo_dir}")
    return AgentIdentity(name="mini-swe-agent", repo_url=repo_url, commit_sha=commit_sha, agent_path=repo_dir, source=source)


def baseline_agent_name(baseline: str) -> str:
    return "mini-swe-agent" if baseline == "mini-swe-agent" else "pi"


def baseline_repo_url(args: argparse.Namespace) -> str:
    return args.mini_swe_agent_repo if args.baseline == "mini-swe-agent" else args.pi_repo


def baseline_ref(args: argparse.Namespace) -> str:
    return args.mini_swe_agent_ref if args.baseline == "mini-swe-agent" else args.pi_ref


def resolve_baseline_repo_agent(
    *,
    baseline: str,
    repo_url: str,
    ref: str,
    cache_root: Path,
    source: dict[str, Any],
) -> AgentIdentity:
    if baseline == "mini-swe-agent":
        return resolve_mini_swe_agent_repo_agent(repo_url=repo_url, ref=ref, cache_root=cache_root, source=source)
    return resolve_pi_repo_agent(repo_url=repo_url, ref=ref, cache_root=cache_root, source=source)


def baseline_pin_path(benchmark_root: Path, baseline: str) -> Path:
    return benchmark_root / "_baselines" / baseline_agent_name(baseline) / "pinned_head.json"


def pi_pin_path(benchmark_root: Path) -> Path:
    return baseline_pin_path(benchmark_root, "pi")


def read_pi_pin(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    if not isinstance(payload, dict) or not payload.get("commit_sha"):
        return None
    return payload


def write_pi_pin(path: Path, *, repo_url: str, ref: str, commit_sha: str) -> dict[str, Any]:
    payload = {
        "repo_url": repo_url,
        "requested_ref": ref,
        "commit_sha": commit_sha,
        "pinned_at": utc_now(),
    }
    write_json(path, payload)
    return payload


def resolve_pinned_baseline_agent(
    *,
    baseline: str,
    benchmark_root: Path,
    repo_url: str,
    ref: str,
    cache_root: Path,
) -> AgentIdentity:
    pin_path = baseline_pin_path(benchmark_root, baseline)
    pin = read_pi_pin(pin_path)
    if pin is not None:
        pinned_repo_url = str(pin.get("repo_url") or repo_url)
        pinned_ref = str(pin["commit_sha"])
        return resolve_baseline_repo_agent(
            baseline=baseline,
            repo_url=pinned_repo_url,
            ref=pinned_ref,
            cache_root=cache_root,
            source={**pin, "pinned": True, "pin_path": str(pin_path)},
        )
    agent = resolve_baseline_repo_agent(
        baseline=baseline,
        repo_url=repo_url,
        ref=ref,
        cache_root=cache_root,
        source={"repo_url": repo_url, "ref": ref, "pinned": False},
    )
    pin = write_pi_pin(pin_path, repo_url=repo_url, ref=ref, commit_sha=agent.commit_sha)
    return AgentIdentity(
        name=agent.name,
        repo_url=agent.repo_url,
        commit_sha=agent.commit_sha,
        agent_path=agent.agent_path,
        source={**pin, "pinned": True, "pin_path": str(pin_path)},
    )


def resolve_pinned_pi_agent(
    *,
    benchmark_root: Path,
    repo_url: str,
    ref: str,
    cache_root: Path,
) -> AgentIdentity:
    return resolve_pinned_baseline_agent(
        baseline="pi",
        benchmark_root=benchmark_root,
        repo_url=repo_url,
        ref=ref,
        cache_root=cache_root,
    )


def baseline_cache_key(
    *,
    baseline_agent: AgentIdentity,
    manifest: BenchmarkManifest,
    model: str,
    provider_only: str,
) -> str:
    payload = {
        "agent": baseline_agent.name,
        "commit_sha": baseline_agent.commit_sha,
        "manifest_hash": manifest.manifest_hash,
        "model": model,
        "provider_only": provider_only,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def baseline_cache_dir(
    *,
    benchmark_root: Path,
    baseline_agent: AgentIdentity,
    manifest: BenchmarkManifest,
    model: str,
    provider_only: str,
) -> Path:
    return benchmark_root / "_baselines" / baseline_agent.name / baseline_cache_key(
        baseline_agent=baseline_agent,
        manifest=manifest,
        model=model,
        provider_only=provider_only,
    )


def pi_baseline_cache_key(
    *,
    pi_agent: AgentIdentity,
    manifest: BenchmarkManifest,
    model: str,
    provider_only: str,
) -> str:
    return baseline_cache_key(
        baseline_agent=pi_agent,
        manifest=manifest,
        model=model,
        provider_only=provider_only,
    )


def pi_baseline_cache_dir(
    *,
    benchmark_root: Path,
    pi_agent: AgentIdentity,
    manifest: BenchmarkManifest,
    model: str,
    provider_only: str,
) -> Path:
    return baseline_cache_dir(
        benchmark_root=benchmark_root,
        baseline_agent=pi_agent,
        manifest=manifest,
        model=model,
        provider_only=provider_only,
    )


def baseline_complete(*, cache_dir: Path, skip_scoring: bool) -> bool:
    required = ("predictions.jsonl", "solve_results.jsonl", "usage_summary.json", "score_summary.json")
    if not all((cache_dir / name).exists() for name in required):
        return False
    if not predictions_include_patch(cache_dir / "predictions.jsonl"):
        return False
    if skip_scoring:
        return True
    return (cache_dir / "official_scoring").exists()


def predictions_include_patch(path: Path) -> bool:
    if not path.exists():
        return False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if str(payload.get("model_patch") or payload.get("patch") or "").strip():
            return True
    return False


def copy_baseline_files(*, source_dir: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in ("predictions.jsonl", "solve_results.jsonl", "usage_summary.json", "score_summary.json"):
        shutil.copy2(source_dir / name, target_dir / name)
    scoring_source = source_dir / "official_scoring"
    if scoring_source.exists():
        shutil.copytree(scoring_source, target_dir / "official_scoring", dirs_exist_ok=True)


def restore_baseline(*, cache_dir: Path, job_dir: Path, baseline_name: str, skip_scoring: bool) -> bool:
    if not baseline_complete(cache_dir=cache_dir, skip_scoring=skip_scoring):
        return False
    copy_baseline_files(source_dir=cache_dir, target_dir=job_dir / baseline_name)
    return True


def restore_pi_baseline(*, cache_dir: Path, job_dir: Path, skip_scoring: bool) -> bool:
    return restore_baseline(cache_dir=cache_dir, job_dir=job_dir, baseline_name="pi", skip_scoring=skip_scoring)


def save_baseline(*, cache_dir: Path, job_dir: Path, baseline_name: str) -> None:
    source_dir = job_dir / baseline_name
    if not (source_dir / "score_summary.json").exists():
        return
    copy_baseline_files(source_dir=source_dir, target_dir=cache_dir)


def save_pi_baseline(*, cache_dir: Path, job_dir: Path) -> None:
    save_baseline(cache_dir=cache_dir, job_dir=job_dir, baseline_name="pi")


def generate_predictions(
    *,
    agents: tuple[AgentIdentity, ...],
    instances: list[dict[str, Any]],
    job_dir: Path,
    model: str,
    provider_only: str,
    openrouter_api_key: str | None,
    api_base: str | None,
    workers: int,
) -> None:
    if not openrouter_api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required for SWE-bench prediction generation")
    for agent in agents:
        if agent.name == "pi":
            ensure_pi_cli(agent)
        if agent.name == "mini-swe-agent":
            ensure_mini_swe_agent_cli(agent)
        reset_prediction_outputs(job_dir / agent.name)
    tasks = [(agent, instance) for agent in agents for instance in instances]
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                solve_agent_instance,
                agent=agent,
                instance=instance,
                job_dir=job_dir,
                model=model,
                provider_only=provider_only,
                openrouter_api_key=openrouter_api_key,
                api_base=api_base,
            )
            for agent, instance in tasks
        ]
        for future in as_completed(futures):
            record, prediction = future.result()
            agent_dir = job_dir / record.agent_name
            append_jsonl(agent_dir / "solve_results.jsonl", asdict(record))
            append_jsonl(agent_dir / "predictions.jsonl", prediction)
            write_usage_summary(agent_dir)
            log.info(
                "SWE-bench %s %s elapsed=%.2fs success=%s",
                record.agent_name,
                record.instance_id,
                record.elapsed_seconds,
                record.success,
            )


def reset_prediction_outputs(agent_dir: Path) -> None:
    for path in (
        agent_dir / "predictions.jsonl",
        agent_dir / "solve_results.jsonl",
        agent_dir / "usage_summary.json",
        agent_dir / "score_summary.json",
    ):
        if path.exists():
            path.unlink()
    for path in (agent_dir / "official_scoring", agent_dir / "logs", agent_dir / "mini_outputs", agent_dir / "repos"):
        if path.exists():
            shutil.rmtree(path)


def solve_agent_instance(
    *,
    agent: AgentIdentity,
    instance: dict[str, Any],
    job_dir: Path,
    model: str,
    provider_only: str,
    openrouter_api_key: str,
    api_base: str | None,
) -> tuple[SolveRecord, dict[str, str]]:
    if agent.name == "pi":
        return solve_pi_instance(
            agent=agent,
            instance=instance,
            job_dir=job_dir,
            model=model,
            openrouter_api_key=openrouter_api_key,
        )
    if agent.name == "mini-swe-agent":
        return solve_mini_swe_agent_instance(
            agent=agent,
            instance=instance,
            job_dir=job_dir,
            model=model,
            openrouter_api_key=openrouter_api_key,
        )
    return solve_instance(
        agent=agent,
        instance=instance,
        job_dir=job_dir,
        model=model,
        provider_only=provider_only,
        openrouter_api_key=openrouter_api_key,
        api_base=api_base,
    )


def solve_instance(
    *,
    agent: AgentIdentity,
    instance: dict[str, Any],
    job_dir: Path,
    model: str,
    provider_only: str,
    openrouter_api_key: str,
    api_base: str | None,
) -> tuple[SolveRecord, dict[str, str]]:
    instance_id = str(instance["instance_id"])
    repo_dir = job_dir / agent.name / "repos" / instance_id
    logs_dir = job_dir / agent.name / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    error: str | None = None
    usage = SolveUsageSummary().to_dict()
    result = None
    try:
        prepare_repo(repo_name=str(instance["repo"]), base_commit=str(instance["base_commit"]), repo_dir=repo_dir)
        task = GeneratedTask(
            title=instance_id,
            description=str(instance["problem_statement"]),
            acceptance_criteria=[],
            raw_output=str(instance["problem_statement"]),
            elapsed_seconds=0.0,
        )
        config = benchmark_run_config(
            agent=agent,
            model=model,
            provider_only=provider_only,
            openrouter_api_key=openrouter_api_key,
        )
        result = solve_task_in_docker(
            repo_dir=repo_dir,
            task=task,
            model=model,
            timeout=config.agent_timeout,
            config=config,
            run_label=f"swebench-{agent.name}-{instance_id}",
            task_name=instance_id,
            solution_name=agent.name,
            repo_full_name=str(instance["repo"]),
            commit_sha=str(instance["base_commit"]),
        )
        usage = result.usage_summary.to_dict() if result.usage_summary else usage
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    elapsed = time.monotonic() - start
    patch = result.solution_diff if result is not None else ""
    logs = result.raw_output if result is not None else ""
    (logs_dir / f"{instance_id}.log").write_text(logs + ("\n" if logs else ""), encoding="utf-8")
    record = SolveRecord(
        agent_name=agent.name,
        instance_id=instance_id,
        repo=str(instance["repo"]),
        elapsed_seconds=elapsed,
        patch_bytes=len(patch.encode("utf-8")),
        success=bool(result.success) if result is not None else False,
        exit_reason=(result.exit_reason if result is not None else "error"),
        error=error,
        usage_summary=usage,
    )
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": agent.name,
        "model_patch": patch,
    }
    return record, prediction


def ensure_pi_cli(agent: AgentIdentity) -> Path:
    repo_dir = agent.agent_path
    cli_path = repo_dir / "packages" / "coding-agent" / "dist" / "cli.js"
    if cli_path.exists():
        return cli_path
    run(["npm", "install", "--ignore-scripts"], cwd=repo_dir, timeout=900)
    run(["npm", "run", "build"], cwd=repo_dir, timeout=900)
    if not cli_path.exists():
        raise FileNotFoundError(f"pi build did not create CLI: {cli_path}")
    return cli_path


def ensure_mini_swe_agent_cli(agent: AgentIdentity) -> Path:
    cli_path = Path(sys.executable).parent / "mini"
    if cli_path.exists() and mini_swe_agent_cli_works(cli_path=cli_path, agent_path=agent.agent_path):
        return cli_path
    run([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "-e", str(agent.agent_path)], cwd=agent.agent_path, timeout=900)
    if cli_path.exists() and mini_swe_agent_cli_works(cli_path=cli_path, agent_path=agent.agent_path):
        return cli_path
    resolved = shutil.which("mini")
    if resolved and mini_swe_agent_cli_works(cli_path=Path(resolved), agent_path=agent.agent_path):
        return Path(resolved)
    raise FileNotFoundError(f"mini-swe-agent install did not create mini near {sys.executable}")


def mini_swe_agent_pythonpath(agent_path: Path) -> str:
    paths = [str(agent_path / "src")]
    current = os.environ.get("PYTHONPATH")
    if current:
        paths.append(current)
    return os.pathsep.join(paths)


def mini_swe_agent_env(*, openrouter_api_key: str, agent_path: Path, model: str | None = None) -> dict[str, str]:
    return {
        **pi_env(openrouter_api_key),
        "PYTHONPATH": mini_swe_agent_pythonpath(agent_path),
        "MSWEA_CONFIGURED": "true",
        "MSWEA_API_KEY": openrouter_api_key,
        "MSWEA_MODEL_NAME": pi_model_arg(model) if model else "",
    }


def mini_swe_agent_cli_works(*, cli_path: Path, agent_path: Path) -> bool:
    result = subprocess.run(
        [str(cli_path), "--help"],
        cwd=agent_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env={**os.environ, "PYTHONPATH": mini_swe_agent_pythonpath(agent_path)},
    )
    return result.returncode == 0


def pi_prompt(instance: dict[str, Any]) -> str:
    return "\n".join(
        [
            "We need modify this repository to fix the following SWE-bench issue.",
            "Edit files directly. Do not ask for clarification. Stop when the fix is implemented.",
            "",
            str(instance["problem_statement"]),
        ],
    )


def pi_env(openrouter_api_key: str) -> dict[str, str]:
    return {
        **os.environ,
        "OPENROUTER_API_KEY": openrouter_api_key,
        "PI_SKIP_VERSION_CHECK": "1",
        "PI_TELEMETRY": "0",
        "MSWEA_COST_TRACKING": "ignore_errors",
        "NO_COLOR": "1",
    }


def pi_model_arg(model: str) -> str:
    return model if model.startswith("openrouter/") else f"openrouter/{model}"


def mini_swe_agent_filter(instance_id: str) -> str:
    return f"^{instance_id}$"


def mini_swe_agent_config(agent: AgentIdentity) -> Path:
    return agent.agent_path / "src" / "minisweagent" / "config" / "mini.yaml"


def mini_swe_agent_prediction_patch(predictions_path: Path, instance_id: str) -> str:
    if not predictions_path.exists():
        return ""
    payload = read_json(predictions_path)
    if isinstance(payload, dict):
        if "model_patch" in payload:
            return str(payload.get("model_patch") or "")
        row = payload.get(instance_id)
        if isinstance(row, dict):
            return str(row.get("model_patch") or row.get("patch") or "")
        if isinstance(row, str):
            return row
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict) and str(row.get("instance_id")) == instance_id:
                return str(row.get("model_patch") or row.get("patch") or "")
    return ""


def solve_pi_instance(
    *,
    agent: AgentIdentity,
    instance: dict[str, Any],
    job_dir: Path,
    model: str,
    openrouter_api_key: str,
) -> tuple[SolveRecord, dict[str, str]]:
    instance_id = str(instance["instance_id"])
    repo_dir = job_dir / agent.name / "repos" / instance_id
    logs_dir = job_dir / agent.name / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    error: str | None = None
    returncode = 1
    output = ""
    patch = ""
    try:
        prepare_repo(repo_name=str(instance["repo"]), base_commit=str(instance["base_commit"]), repo_dir=repo_dir)
        cli_path = ensure_pi_cli(agent)
        result = subprocess.run(
            [
                "node",
                str(cli_path),
                "--print",
                "--provider",
                "openrouter",
                "--model",
                pi_model_arg(model),
                "--no-session",
                pi_prompt(instance),
            ],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env=mini_swe_agent_env(openrouter_api_key=openrouter_api_key, agent_path=agent.agent_path, model=model),
        )
        returncode = result.returncode
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        patch = git_diff(repo_dir)
        if result.returncode != 0:
            error = f"pi exited {result.returncode}"
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    elapsed = time.monotonic() - start
    if repo_dir.exists():
        patch = git_diff(repo_dir)
    (logs_dir / f"{instance_id}.log").write_text(output + ("\n" if output else ""), encoding="utf-8")
    usage = SolveUsageSummary().to_dict()
    record = SolveRecord(
        agent_name=agent.name,
        instance_id=instance_id,
        repo=str(instance["repo"]),
        elapsed_seconds=elapsed,
        patch_bytes=len(patch.encode("utf-8")),
        success=error is None and bool(patch.strip()),
        exit_reason="completed" if error is None else "error",
        error=error,
        usage_summary=usage,
    )
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": agent.name,
        "model_patch": patch,
    }
    return record, prediction


def solve_mini_swe_agent_instance(
    *,
    agent: AgentIdentity,
    instance: dict[str, Any],
    job_dir: Path,
    model: str,
    openrouter_api_key: str,
) -> tuple[SolveRecord, dict[str, str]]:
    instance_id = str(instance["instance_id"])
    agent_dir = job_dir / agent.name
    repo_dir = agent_dir / "repos" / instance_id
    logs_dir = agent_dir / "logs"
    output_dir = agent_dir / "mini_outputs" / instance_id
    logs_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    error: str | None = None
    output = ""
    patch = ""
    try:
        prepare_repo(repo_name=str(instance["repo"]), base_commit=str(instance["base_commit"]), repo_dir=repo_dir)
        cli_path = ensure_mini_swe_agent_cli(agent)
        config_path = mini_swe_agent_config(agent)
        if not config_path.exists():
            raise FileNotFoundError(f"mini-swe-agent config missing: {config_path}")
        result = subprocess.run(
            [
                str(cli_path),
                "--model",
                pi_model_arg(model),
                "--task",
                pi_prompt(instance),
                "--yolo",
                "--exit-immediately",
                "--cost-limit",
                "3",
                "--config",
                str(config_path),
                "--config",
                "agent.step_limit=80",
                "--output",
                str(output_dir / "trajectory.json"),
            ],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            env=mini_swe_agent_env(openrouter_api_key=openrouter_api_key, agent_path=agent.agent_path),
        )
        output = ((result.stdout or "") + (result.stderr or "")).strip()
        patch = git_diff(repo_dir)
        if result.returncode != 0:
            error = f"mini-swe-agent exited {result.returncode}"
    except Exception as exc:  # noqa: BLE001
        error = repr(exc)
    elapsed = time.monotonic() - start
    (logs_dir / f"{instance_id}.log").write_text(output + ("\n" if output else ""), encoding="utf-8")
    usage = SolveUsageSummary().to_dict()
    record = SolveRecord(
        agent_name=agent.name,
        instance_id=instance_id,
        repo=str(instance["repo"]),
        elapsed_seconds=elapsed,
        patch_bytes=len(patch.encode("utf-8")),
        success=error is None and bool(patch.strip()),
        exit_reason="completed" if error is None else "error",
        error=error,
        usage_summary=usage,
    )
    prediction = {
        "instance_id": instance_id,
        "model_name_or_path": agent.name,
        "model_patch": patch,
    }
    return record, prediction


def _benchmark_solver_agent_source(agent: AgentIdentity) -> SolverAgentSource:
    agent_path = agent.agent_path
    repo_dir = agent_path.parent
    manifest = repo_dir / "tau_agent_files.json"
    if agent_path.name == "agent.py" and manifest.is_file():
        return SolverAgentSource(
            raw=str(repo_dir),
            kind="local_path",
            local_path=str(repo_dir),
            agent_file="agent.py",
            commit_sha=agent.commit_sha,
        )
    if agent_path.is_dir():
        return SolverAgentSource(
            raw=str(agent_path),
            kind="local_path",
            local_path=str(agent_path),
            agent_file="agent.py",
            commit_sha=agent.commit_sha,
        )
    return SolverAgentSource(
        raw=str(agent_path),
        kind="local_file",
        local_path=str(agent_path),
        agent_file="agent.py",
        commit_sha=agent.commit_sha,
    )


def benchmark_run_config(
    *,
    agent: AgentIdentity,
    model: str,
    provider_only: str,
    openrouter_api_key: str,
) -> RunConfig:
    return RunConfig(
        openrouter_api_key=openrouter_api_key,
        solver_model=model,
        agent_timeout=600,
        solver_provider_only=provider_only,
        solver_provider_allow_fallbacks=False,
        solver_agent_source=_benchmark_solver_agent_source(agent),
        docker_solver_memory="2g",
        docker_solver_cpus="1",
        docker_solver_pids_limit=256,
        docker_solver_tmp_size="128m",
        docker_solver_workdir_size="2g",
        docker_solver_max_output_bytes=100_000_000,
    )


def prepare_repo(*, repo_name: str, base_commit: str, repo_dir: Path) -> None:
    if repo_dir.exists():
        shutil.rmtree(repo_dir)
    repo_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = repo_dir.with_name(f"{repo_dir.name}.tmp")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    run(["git", "clone", "--quiet", "--no-tags", f"https://github.com/{repo_name}.git", str(temp_dir)], cwd=None, timeout=600)
    run(["git", "checkout", "--quiet", base_commit], cwd=temp_dir, timeout=300)
    temp_dir.rename(repo_dir)


def score_predictions(
    *,
    agent: AgentIdentity,
    manifest: BenchmarkManifest,
    job_dir: Path,
    max_workers: int,
    skip: bool,
) -> ScoreSummary:
    agent_dir = job_dir / agent.name
    scoring_dir = agent_dir / "official_scoring"
    scoring_dir.mkdir(parents=True, exist_ok=True)
    if skip:
        summary = ScoreSummary(resolved_count=None, total_count=len(manifest.instance_ids), pass_rate=None, report_path=None, error="skipped")
        write_json(agent_dir / "score_summary.json", summary.to_dict())
        return summary
    command = [
        sys.executable,
        "-m",
        "swebench.harness.run_evaluation",
        "--dataset_name",
        manifest.dataset_name,
        "--split",
        manifest.split,
        "--predictions_path",
        str(agent_dir / "predictions.jsonl"),
        "--max_workers",
        str(max_workers),
        "--run_id",
        f"tau-{job_dir.name}-{agent.name}",
        "--instance_ids",
        *manifest.instance_ids,
    ]
    result = subprocess.run(command, cwd=scoring_dir, capture_output=True, text=True, check=False)
    (scoring_dir / "stdout.txt").write_text(result.stdout, encoding="utf-8")
    (scoring_dir / "stderr.txt").write_text(result.stderr, encoding="utf-8")
    if result.returncode != 0:
        summary = ScoreSummary(
            resolved_count=None,
            total_count=len(manifest.instance_ids),
            pass_rate=None,
            report_path=None,
            error=f"official scoring failed exit={result.returncode}",
        )
        write_json(agent_dir / "score_summary.json", summary.to_dict())
        return summary
    summary = extract_score_summary(scoring_dir=scoring_dir, total_count=len(manifest.instance_ids))
    write_json(agent_dir / "score_summary.json", summary.to_dict())
    return summary


def read_score_summary(path: Path) -> ScoreSummary:
    payload = read_json(path)
    return ScoreSummary(
        resolved_count=payload.get("resolved_count"),
        total_count=int(payload["total_count"]),
        pass_rate=payload.get("pass_rate"),
        report_path=payload.get("report_path"),
        error=payload.get("error"),
    )


def extract_score_summary(*, scoring_dir: Path, total_count: int) -> ScoreSummary:
    candidates = sorted(scoring_dir.rglob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        try:
            payload = read_json(path)
        except Exception:
            continue
        resolved_ids = resolved_ids_from_report(payload)
        if resolved_ids is not None:
            resolved_count = len(resolved_ids)
            return ScoreSummary(
                resolved_count=resolved_count,
                total_count=total_count,
                pass_rate=resolved_count / total_count if total_count else None,
                report_path=str(path),
            )
    return ScoreSummary(
        resolved_count=None,
        total_count=total_count,
        pass_rate=None,
        report_path=None,
        error="no official SWE-bench report found",
    )


def resolved_ids_from_report(payload: Any) -> set[str] | None:
    if isinstance(payload, dict):
        for key in ("resolved_ids", "resolved_instances"):
            value = payload.get(key)
            if isinstance(value, list):
                return {str(item) for item in value}
        if all(isinstance(value, dict) and "resolved" in value for value in payload.values()):
            return {str(key) for key, value in payload.items() if value.get("resolved") is True}
        for value in payload.values():
            nested = resolved_ids_from_report(value)
            if nested is not None:
                return nested
    return None


def write_usage_summary(agent_dir: Path) -> None:
    records = read_jsonl(agent_dir / "solve_results.jsonl")
    usage = aggregate_usage([record.get("usage_summary") for record in records if isinstance(record, dict)])
    write_json(agent_dir / "usage_summary.json", usage)


def aggregate_usage(usages: list[Any]) -> dict[str, Any]:
    numeric_keys = (
        "request_count",
        "rejected_request_count",
        "success_count",
        "error_count",
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "cached_tokens",
        "cache_write_tokens",
        "reasoning_tokens",
        "cost",
    )
    totals: dict[str, Any] = {key: 0 for key in numeric_keys}
    cost_available = False
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key in numeric_keys:
            value = usage.get(key)
            if isinstance(value, (int, float)):
                totals[key] += value
        requests = usage.get("requests")
        if isinstance(requests, list) and any(isinstance(req, dict) and req.get("cost") is not None for req in requests):
            cost_available = True
    totals["cost_available"] = cost_available
    if not cost_available:
        totals["cost"] = None
    return totals


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def build_comparison(
    *,
    king: dict[str, Any],
    king_agent: AgentIdentity,
    baseline_agent: AgentIdentity,
    manifest: BenchmarkManifest,
    job_dir: Path,
    scores: tuple[ScoreSummary, ScoreSummary],
    total_elapsed_seconds: float,
    model: str,
    provider_only: str,
    baseline_cached: bool,
) -> dict[str, Any]:
    king_score, baseline_score = scores
    king_usage = read_json(job_dir / "king" / "usage_summary.json")
    baseline_usage = read_json(job_dir / baseline_agent.name / "usage_summary.json")
    if baseline_agent.name == "mini-swe-agent":
        baseline_usage = mini_swe_usage_for_score(job_dir / baseline_agent.name, baseline_score) or baseline_usage
    delta = (
        king_score.pass_rate - baseline_score.pass_rate
        if king_score.pass_rate is not None and baseline_score.pass_rate is not None
        else None
    )
    started_at = read_json(job_dir / "job.json").get("started_at", utc_now())
    baseline_payload = baseline_agent.to_dict()
    return {
        "benchmark": "swebench_verified_sample_50",
        "status": "completed",
        "started_at": started_at,
        "finished_at": utc_now(),
        "king_commit_sha": king_agent.commit_sha,
        "king": king,
        "baseline": baseline_payload,
        "baseline_name": baseline_agent.name,
        "baseline_cached": baseline_cached,
        "pi": baseline_payload,
        "pi_baseline_cached": baseline_cached,
        "model": model,
        "provider_only": provider_only,
        "manifest": manifest.to_dict(),
        "scores": {
            "king": king_score.to_dict(),
            "baseline": baseline_score.to_dict(),
            "pi": baseline_score.to_dict(),
            "delta_pass_rate": delta,
        },
        "usage": {
            "king": king_usage,
            "baseline": baseline_usage,
            "pi": baseline_usage,
            "cost_available": bool(king_usage.get("cost_available") and baseline_usage.get("cost_available")),
        },
        "elapsed": {
            "wall_seconds": total_elapsed_seconds,
            "king_task_seconds": sum_task_elapsed(job_dir / "king" / "solve_results.jsonl"),
            "baseline_task_seconds": sum_task_elapsed(job_dir / baseline_agent.name / "solve_results.jsonl"),
            "pi_task_seconds": sum_task_elapsed(job_dir / baseline_agent.name / "solve_results.jsonl"),
        },
    }


def sum_task_elapsed(path: Path) -> float:
    return sum(float(record.get("elapsed_seconds", 0.0)) for record in read_jsonl(path))


def walk_json_objects(value: Any) -> Any:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_json_objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json_objects(child)


def mini_swe_usage_from_trajectories(agent_dir: Path) -> dict[str, Any] | None:
    outputs_dir = agent_dir / "mini_outputs"
    if not outputs_dir.is_dir():
        return None
    totals: dict[str, Any] = {
        "cost": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "trajectory_count": 0,
    }
    for trajectory_path in sorted(outputs_dir.glob("*/trajectory.json")):
        try:
            payload = read_json(trajectory_path)
        except Exception:
            continue
        totals["trajectory_count"] += 1
        for item in walk_json_objects(payload):
            usage = item.get("usage") if isinstance(item, dict) else None
            if not isinstance(usage, dict):
                continue
            cost = usage.get("cost")
            if isinstance(cost, (int, float)):
                totals["cost"] += float(cost)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    totals[key] += int(value)
    if not totals["trajectory_count"]:
        return None
    return {**totals, "cost_available": totals["cost"] > 0}


def mini_swe_usage_for_score(agent_dir: Path, score: ScoreSummary) -> dict[str, Any] | None:
    local_usage = mini_swe_usage_from_trajectories(agent_dir)
    if local_usage is not None:
        return local_usage
    if not score.report_path:
        return None
    report_path = Path(score.report_path)
    if report_path.parent.name != "official_scoring":
        return None
    return mini_swe_usage_from_trajectories(report_path.parent.parent)


def write_history_entry(path: Path, comparison: dict[str, Any]) -> None:
    existing = read_jsonl(path)
    if any(row.get("king_commit_sha") == comparison.get("king_commit_sha") for row in existing):
        return
    append_jsonl(path, comparison)


def benchmark_dashboard_payload(comparison: dict[str, Any]) -> dict[str, Any]:
    history_item = compact_dashboard_benchmark(comparison)
    return {
        "swebench_verified": {
            "latest": history_item,
            "history": [history_item],
        },
    }


def compact_dashboard_benchmark(comparison: dict[str, Any]) -> dict[str, Any]:
    scores = comparison.get("scores") if isinstance(comparison.get("scores"), dict) else {}
    return {
        "benchmark": comparison.get("benchmark", "swebench_verified_sample_50"),
        "status": comparison.get("status"),
        "started_at": comparison.get("started_at"),
        "finished_at": comparison.get("finished_at"),
        "king_commit_sha": comparison.get("king_commit_sha"),
        "baseline_name": comparison.get("baseline_name"),
        "baseline_cached": comparison.get("baseline_cached"),
        "pi_baseline_cached": comparison.get("pi_baseline_cached"),
        "model": comparison.get("model"),
        "provider_only": comparison.get("provider_only"),
        "manifest_hash": ((comparison.get("manifest") or {}).get("manifest_hash") if isinstance(comparison.get("manifest"), dict) else None),
        "king": scores.get("king"),
        "baseline": scores.get("baseline") or scores.get("pi"),
        "pi": scores.get("pi"),
        "delta_pass_rate": scores.get("delta_pass_rate"),
        "elapsed": comparison.get("elapsed"),
        "usage": comparison.get("usage"),
    }


def merge_benchmark_into_dashboard(dashboard_path: Path, comparison_or_benchmarks: dict[str, Any]) -> None:
    payload = read_json(dashboard_path) if dashboard_path.exists() else {"updated_at": utc_now()}
    benchmarks = payload.get("benchmarks") if isinstance(payload.get("benchmarks"), dict) else {}
    incoming = (
        comparison_or_benchmarks
        if "swebench_verified" in comparison_or_benchmarks
        else benchmark_dashboard_payload(comparison_or_benchmarks)
    )
    swebench_payload = incoming["swebench_verified"]
    old_swebench = benchmarks.get("swebench_verified") if isinstance(benchmarks.get("swebench_verified"), dict) else {}
    old_history = old_swebench.get("history") if isinstance(old_swebench.get("history"), list) else []
    new_history = swebench_payload.get("history") if isinstance(swebench_payload.get("history"), list) else []
    merged_history = dedupe_history([*old_history, *new_history])
    benchmarks["swebench_verified"] = {
        "latest": swebench_payload.get("latest") or (merged_history[-1] if merged_history else None),
        "history": merged_history[-20:],
    }
    payload["benchmarks"] = benchmarks
    payload["updated_at"] = utc_now()
    write_json(dashboard_path, payload)


def dedupe_history(items: list[Any]) -> list[dict[str, Any]]:
    by_commit: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        key = str(item.get("king_commit_sha") or item.get("finished_at") or len(by_commit))
        by_commit[key] = item
    return list(by_commit.values())


def publish_benchmark_payload(comparison: dict[str, Any]) -> bool:
    try:
        from r2 import publish_benchmark_data
    except Exception:
        return False
    return publish_benchmark_data(benchmark_payload=benchmark_dashboard_payload(comparison))


def run(command: list[str], *, cwd: Path | None, timeout: int, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed: {result.stderr[-1000:]}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run SWE-bench Verified when a new king is crowned.")
    parser.add_argument("--validate-root", type=Path, default=Path("workspace/validate/netuid-66"))
    parser.add_argument("--state-path", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=Path("data/swebench_verified_sample_50_seed66.json"))
    parser.add_argument("--baseline", choices=("pi", "mini-swe-agent"), default=DEFAULT_BASELINE)
    parser.add_argument("--pi-repo", default=DEFAULT_PI_REPO_URL)
    parser.add_argument("--pi-ref", default="main")
    parser.add_argument("--mini-swe-agent-repo", default=DEFAULT_MINI_SWE_AGENT_REPO_URL)
    parser.add_argument("--mini-swe-agent-ref", default="main")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--provider-only", default=DEFAULT_PROVIDER_ONLY)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--openrouter-api-key", default=None)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--poll-interval-seconds", type=int, default=DEFAULT_POLL_INTERVAL_SECONDS)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--skip-scoring", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args.validate_root = args.validate_root.expanduser().resolve()
    args.state_path = (args.state_path or (args.validate_root / "state.json")).expanduser().resolve()
    args.manifest = args.manifest.expanduser().resolve()
    if args.once:
        print(json.dumps(run_once(args), indent=2, sort_keys=True))
        return
    run_daemon(args)


if __name__ == "__main__":
    main()
