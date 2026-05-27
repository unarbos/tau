import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from config import RunConfig, SolverAgentSource
from github_miner import CommitCandidate, CommitFile
from pipeline import solve_task_run
from task_generation import GeneratedTask
from tau.rollouts.store import load_task_rollouts
from validate import DiffJudgeResult, _record_round_rollout_outcomes, _solution_rollout_id
from workspace import build_task_paths, write_json


CHEAP_OPENROUTER_MODEL = "deepseek/deepseek-v4-flash"


def _llm_event_uses_model(event: dict, model: str) -> bool:
    return event.get("model_requested") == model or event.get("model_effective") == model


def _run(cmd: list[str], *, cwd: Path) -> None:
    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def _write_live_agent(path: Path) -> None:
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
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
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


def _create_git_repo(path: Path) -> None:
    path.mkdir(parents=True)
    _run(["git", "init"], cwd=path)
    (path / "answer.txt").write_text("hello world\n", encoding="utf-8")
    _run(["git", "add", "answer.txt"], cwd=path)
    _run(["git", "-c", "user.email=a@b.c", "-c", "user.name=Tau", "commit", "-m", "init"], cwd=path)


def _write_task_fixture(workspace_root: Path, task_name: str) -> None:
    task_paths = build_task_paths(workspace_root / "workspace" / "tasks", task_name)
    task_paths.task_dir.mkdir(parents=True)
    task_paths.solutions_dir.mkdir(parents=True)
    task_paths.comparisons_dir.mkdir(parents=True)
    task_paths.evals_dir.mkdir(parents=True)
    _create_git_repo(task_paths.original_dir)
    candidate = CommitCandidate(
        repo_full_name="local/validator-e2e",
        repo_clone_url=str(task_paths.original_dir),
        commit_sha="b" * 40,
        parent_sha="a" * 40,
        message="Change answer text",
        html_url="https://example.invalid/local/validator-e2e",
        author_name="Tau",
        event_id="validator-e2e",
        files=[
            CommitFile(
                filename="answer.txt",
                status="modified",
                additions=1,
                deletions=1,
                changes=2,
                patch='@@ -1 +1 @@\n-hello world\n+hello validator',
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


@unittest.skipUnless(
    os.environ.get("TAU_RUN_OPENROUTER_E2E") == "1",
    "set TAU_RUN_OPENROUTER_E2E=1 to run the live OpenRouter/Docker validator rollout e2e",
)
class ValidateRolloutsE2ETest(unittest.TestCase):
    def test_docker_solve_records_and_validator_annotates_rollout(self):
        if not os.environ.get("OPENROUTER_API_KEY"):
            self.skipTest("OPENROUTER_API_KEY is required for the live e2e")
        if shutil.which("docker") is None:
            self.skipTest("docker is required for the live e2e")

        model = os.environ.get("TAU_E2E_OPENROUTER_MODEL", CHEAP_OPENROUTER_MODEL)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_path = root / "agent.py"
            _write_live_agent(agent_path)
            task_name = "validate-rollout-e2e"
            _write_task_fixture(root, task_name)

            config = RunConfig(
                workspace_root=root,
                openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
                solver_backend="docker-file",
                solver_model=model,
                diff_judge_model=model,
                diff_judge_fallback_models="",
                agent_timeout=180,
                solver_max_requests=1,
                solver_max_completion_tokens=32,
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

            result = solve_task_run(
                task_name=task_name,
                solution_name="challenger-1-d1",
                config=config,
            )
            self.assertTrue(result.success, result.exit_reason)

            rollout_id = _solution_rollout_id(
                task_name=task_name,
                solution_name="challenger-1-d1",
                config=config,
            )
            self.assertIsNotNone(rollout_id)
            _record_round_rollout_outcomes(
                config=config,
                task_name=task_name,
                winner="challenger",
                king_rollout_id=None,
                challenger_rollout_id=rollout_id,
                duel_id=1,
                diff_judge=DiffJudgeResult(
                    winner="challenger",
                    king_score=0.0,
                    challenger_score=1.0,
                    rationale="e2e annotation",
                    model=model,
                ),
            )

            records = load_task_rollouts(config.resolved_rollout_root(), task_name)
            self.assertEqual(len(records), 1)
            record = records[0]
            self.assertEqual(record["rollout_id"], rollout_id)
            self.assertEqual(record["duel_id"], 1)
            self.assertEqual(record["pairwise"]["winner_role"], "challenger")
            self.assertTrue(any(event.get("type") == "llm_call" for event in record["trajectory"]))
            self.assertTrue(
                any(_llm_event_uses_model(event, model) for event in record["trajectory"])
            )
            self.assertTrue(any(event.get("type") == "command" for event in record["trajectory"]))
            self.assertTrue(any(event.get("type") == "edit" for event in record["trajectory"]))
            self.assertNotIn(os.environ["OPENROUTER_API_KEY"], json.dumps(record))


if __name__ == "__main__":
    unittest.main()
