from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from config import RunConfig, SolverAgentSource
from github_miner import CommitCandidate, CommitFile
from pipeline import solve_task_run
from task_generation import GeneratedTask
from tau.rollouts.export_dpo import dpo_row
from tau.rollouts.export_hf import export_retired_rollouts_to_hf, export_task_rollouts_to_hf
from tau.rollouts.export_grpo import grpo_row
from tau.rollouts.redaction import public_rollout
from tau.rollouts.store import load_task_rollouts, rollout_record_path, update_rollout
from validate import DiffJudgeResult, _record_round_rollout_outcomes, _solution_rollout_id
from workspace import build_solution_paths, build_task_paths, resolve_task_paths, write_json


REQUIRED_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free"


def run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def write_agent(path: Path) -> None:
    path.write_text(
        """
import json
import subprocess
import urllib.request
from pathlib import Path


def solve(repo_path, issue, model, api_base, api_key):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with ok only."}],
        "max_tokens": 8,
    }
    request = urllib.request.Request(
        api_base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        json.loads(response.read().decode("utf-8"))
    subprocess.run(
        ["bash", "-lc", "printf 'hello validator\\n' > answer.txt"],
        cwd=repo_path,
        check=True,
    )
    subprocess.run(["git", "diff", "--", "answer.txt"], cwd=repo_path, check=True, capture_output=True, text=True)
    assert Path(repo_path, "answer.txt").read_text(encoding="utf-8") == "hello validator\\n"
    return {"success": True}
""".strip()
        + "\n",
        encoding="utf-8",
    )


def create_git_repo(path: Path) -> None:
    path.mkdir(parents=True)
    run(["git", "init"], cwd=path)
    (path / "answer.txt").write_text("hello world\n", encoding="utf-8")
    run(["git", "add", "answer.txt"], cwd=path)
    run(["git", "-c", "user.email=a@b.c", "-c", "user.name=Tau", "commit", "-m", "init"], cwd=path)


def write_task_fixture(workspace_root: Path, task_name: str) -> None:
    task_paths = build_task_paths(workspace_root / "workspace" / "tasks", task_name)
    if task_paths.task_json_path.exists():
        return
    task_paths.task_dir.mkdir(parents=True)
    task_paths.solutions_dir.mkdir(parents=True)
    task_paths.comparisons_dir.mkdir(parents=True)
    task_paths.evals_dir.mkdir(parents=True)
    create_git_repo(task_paths.original_dir)
    candidate = CommitCandidate(
        repo_full_name="local/pr50-validator-sim",
        repo_clone_url=str(task_paths.original_dir),
        commit_sha="b" * 40,
        parent_sha="a" * 40,
        message="Change answer text",
        html_url="https://example.invalid/local/pr50-validator-sim",
        author_name="Tau",
        event_id=task_name,
        files=[
            CommitFile(
                filename="answer.txt",
                status="modified",
                additions=1,
                deletions=1,
                changes=2,
                patch="@@ -1 +1 @@\n-hello world\n+hello validator",
            ),
        ],
    )
    task = GeneratedTask(
        title="Change answer text",
        description="Update answer.txt so it contains the validator greeting.",
        acceptance_criteria=["answer.txt contains hello validator"],
        raw_output="fixture",
        elapsed_seconds=0.0,
    )
    write_json(task_paths.commit_path, candidate.to_dict())
    write_json(task_paths.task_json_path, task.to_dict())
    task_paths.task_txt_path.write_text(task.prompt_text + "\n", encoding="utf-8")
    task_paths.reference_patch_path.write_text(candidate.combined_patch + "\n", encoding="utf-8")


def build_config(*, root: Path, agent_path: Path, model: str) -> RunConfig:
    return RunConfig(
        workspace_root=root,
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
        solver_backend="docker-file",
        solver_model=model,
        diff_judge_model=model,
        diff_judge_fallback_models="",
        agent_timeout=180,
        solver_max_requests=2,
        solver_max_completion_tokens=128,
        solver_max_cost=0.01,
        solve_agent=str(agent_path),
        solver_agent_source=SolverAgentSource(
            raw=str(agent_path),
            kind="local_file",
            local_path=str(agent_path),
            agent_file=agent_path.name,
        ),
        record_rollouts=True,
        rollout_root=root / "rollouts",
    )


def solve_names(prefix: str, count: int) -> list[str]:
    return [f"{prefix}-{index:02d}" for index in range(1, count + 1)]


def delete_failed_rollout(result: Any) -> None:
    rollout_path = getattr(result, "rollout_path", None)
    if rollout_path:
        Path(rollout_path).unlink(missing_ok=True)


def delete_solution_workspace(*, config: RunConfig, task_name: str, solution_name: str) -> None:
    task_paths = resolve_task_paths(config.tasks_root, task_name)
    shutil.rmtree(build_solution_paths(task_paths, solution_name).root, ignore_errors=True)


def solve_once_with_retries(
    *,
    config: RunConfig,
    task_name: str,
    solution_name: str,
    max_attempts: int,
) -> str:
    existing_rollout_id = _solution_rollout_id(task_name=task_name, solution_name=solution_name, config=config)
    if existing_rollout_id and rollout_record_path(config.resolved_rollout_root(), task_name, existing_rollout_id).exists():
        return existing_rollout_id
    last_exit_reason = None
    for attempt in range(1, max_attempts + 1):
        delete_solution_workspace(config=config, task_name=task_name, solution_name=solution_name)
        result = solve_task_run(task_name=task_name, solution_name=solution_name, config=config)
        if result.success:
            rollout_id = _solution_rollout_id(task_name=task_name, solution_name=solution_name, config=config)
            if rollout_id:
                return rollout_id
        last_exit_reason = result.exit_reason
        delete_failed_rollout(result)
        delete_solution_workspace(config=config, task_name=task_name, solution_name=solution_name)
        if result.exit_reason == "provider_endpoint_error":
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"{task_name}/{solution_name} failed after {max_attempts} attempts: {last_exit_reason}")


def solve_all(*, config: RunConfig, task_names: list[str], solution_names: list[str], max_attempts: int) -> list[dict[str, str]]:
    results = []
    for task_name in task_names:
        for solution_name in solution_names:
            rollout_id = solve_once_with_retries(
                config=config,
                task_name=task_name,
                solution_name=solution_name,
                max_attempts=max_attempts,
            )
            results.append({"task_name": task_name, "solution_name": solution_name, "rollout_id": rollout_id})
    return results


def prune_unreferenced_records(*, config: RunConfig, expected: list[dict[str, str]]) -> None:
    expected_by_task: dict[str, set[str]] = {}
    for item in expected:
        expected_by_task.setdefault(item["task_name"], set()).add(item["rollout_id"])
    for task_name, expected_ids in expected_by_task.items():
        for record in load_task_rollouts(config.resolved_rollout_root(), task_name):
            rollout_id = str(record.get("rollout_id") or "")
            if rollout_id and rollout_id not in expected_ids:
                rollout_record_path(config.resolved_rollout_root(), task_name, rollout_id).unlink(missing_ok=True)


def annotate_pairs(
    *,
    config: RunConfig,
    task_names: list[str],
    baselines: list[str],
    challengers: list[str],
    model: str,
) -> int:
    duel_id = 0
    for task_name in task_names:
        for challenger in challengers:
            challenger_rollout_id = _solution_rollout_id(task_name=task_name, solution_name=challenger, config=config)
            for baseline in baselines:
                duel_id += 1
                baseline_rollout_id = _solution_rollout_id(task_name=task_name, solution_name=baseline, config=config)
                _record_round_rollout_outcomes(
                    config=config,
                    task_name=task_name,
                    winner="challenger",
                    king_rollout_id=baseline_rollout_id,
                    challenger_rollout_id=challenger_rollout_id,
                    duel_id=duel_id,
                    diff_judge=DiffJudgeResult(
                        winner="challenger",
                        king_score=0.0,
                        challenger_score=1.0,
                        rationale=f"simulated validator pairwise: {challenger} vs {baseline}",
                        model=model,
                    ),
                )
                update_rollout(
                    config.resolved_rollout_root(),
                    task_name,
                    challenger_rollout_id,
                    {"baseline_solution_name": baseline, "challenger_solution_name": challenger},
                )
                update_rollout(
                    config.resolved_rollout_root(),
                    task_name,
                    baseline_rollout_id,
                    {"baseline_solution_name": baseline, "challenger_solution_name": challenger},
                )
    return duel_id


def event_counts(records: list[dict[str, Any]], model: str) -> dict[str, int]:
    llm = [
        record
        for record in records
        if any(
            event.get("type") == "llm_call"
            and (event.get("model_requested") == model or event.get("model_effective") == model)
            for event in record.get("trajectory", [])
        )
    ]
    command_or_edit = [
        record
        for record in records
        if any(event.get("type") in {"command", "edit"} for event in record.get("trajectory", []))
    ]
    judged = [record for record in records if record.get("judge") and record.get("pairwise")]
    return {"llm_records": len(llm), "command_edit_records": len(command_or_edit), "judged_records": len(judged)}


def leaked_secrets(*, records: list[dict[str, Any]], secrets: list[str]) -> list[str]:
    dumped = "\n".join(json.dumps(record, sort_keys=True) for record in records)
    return [name for name, value in (secret.split("=", 1) for secret in secrets if "=" in secret) if value and value in dumped]


def verify_exports(*, root: Path, task_names: list[str]) -> dict[str, Any]:
    uploads = []

    def fake_upload(**kwargs: Any) -> None:
        uploads.append(kwargs["path_in_repo"])

    config = SimpleNamespace(
        push_rollouts_to_hf=True,
        rollout_hf_dataset="owner/dataset",
        rollout_hf_token_env="PR50_FAKE_HF_TOKEN",
        resolved_rollout_root=lambda: root,
    )
    os.environ["PR50_FAKE_HF_TOKEN"] = "fake-hf-token"
    for task_name in task_names:
        export_task_rollouts_to_hf(config=config, task_name=task_name, upload_file=fake_upload)
    repeat_count = export_retired_rollouts_to_hf(config=config, active_task_names=set(), upload_file=fake_upload)
    return {"uploads": len(uploads), "repeat_uploads": repeat_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", type=Path, default=Path("tmp/pr50-real-validator-sim"))
    parser.add_argument("--tasks", type=int, default=10)
    parser.add_argument("--challengers", type=int, default=10)
    parser.add_argument("--baselines", type=int, default=10)
    parser.add_argument("--max-attempts", type=int, default=20)
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--model", default=REQUIRED_MODEL)
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY is required")
    if shutil.which("docker") is None:
        raise SystemExit("docker is required")

    root = args.workspace_root.resolve()
    if args.clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    agent_path = root / "agent.py"
    write_agent(agent_path)
    task_names = [f"pr50-rollout-task-{index:02d}" for index in range(1, args.tasks + 1)]
    for task_name in task_names:
        write_task_fixture(root, task_name)

    config = build_config(root=root, agent_path=agent_path, model=args.model)
    baselines = solve_names("past-king", args.baselines)
    challengers = solve_names("challenger", args.challengers)
    solved = solve_all(
        config=config,
        task_names=task_names,
        solution_names=[*baselines, *challengers],
        max_attempts=args.max_attempts,
    )
    prune_unreferenced_records(config=config, expected=solved)
    pairwise_count = annotate_pairs(
        config=config,
        task_names=task_names,
        baselines=baselines,
        challengers=challengers,
        model=args.model,
    )

    records = [record for task_name in task_names for record in load_task_rollouts(config.resolved_rollout_root(), task_name)]
    counts = event_counts(records, args.model)
    redacted_rows = [public_rollout(record) for record in records]
    training_rows = [
        dpo_row(task_name=task_names[0], chosen=records[0], rejected=records[1], source="pr50-real-validator-sim"),
        grpo_row(task_name=task_names[0], group_id="sim-1", rollouts=records[: min(4, len(records))]),
    ]
    export_summary = verify_exports(root=config.resolved_rollout_root(), task_names=task_names)
    leak_names = leaked_secrets(
        records=[*records, *redacted_rows, *training_rows],
        secrets=[f"OPENROUTER_API_KEY={os.environ.get('OPENROUTER_API_KEY', '')}", "PR50_FAKE_HF_TOKEN=fake-hf-token"],
    )
    summary = {
        "workspace_root": str(root),
        "model": args.model,
        "tasks": len(task_names),
        "challengers": len(challengers),
        "baselines": len(baselines),
        "rollout_records": len(records),
        "pairwise_comparisons": pairwise_count,
        "llm_records": counts["llm_records"],
        "command_edit_records": counts["command_edit_records"],
        "judged_records": counts["judged_records"],
        "credential_leaks": leak_names,
        "training_rows": len(training_rows),
        "hf_export_uploads": export_summary["uploads"],
        "hf_export_repeat_uploads": export_summary["repeat_uploads"],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    if leak_names:
        raise SystemExit("credential leakage found")


if __name__ == "__main__":
    main()
