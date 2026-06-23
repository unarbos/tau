import gzip
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import docker_solver
from tau.rollouts.export_dpo import dpo_row
from tau.rollouts.export_grpo import grpo_row
from tau.rollouts.export_hf import (
    clear_uploaded_rollout_tasks,
    export_retired_rollouts_to_hf,
    export_task_rollouts_to_hf,
    load_export_manifest,
    mark_task_rollouts_exported,
    uploaded_local_rollout_task_names,
)
from tau.rollouts.ids import event_id, rollout_id
from tau.rollouts.schema import build_llm_event, build_rollout_record
from tau.rollouts.store import append_rollout, load_task_rollouts, update_rollout


class RolloutHelpersTest(unittest.TestCase):
    def test_rollout_ids_are_stable(self):
        left = rollout_id(
            task_name="task-a",
            solution_name="challenger",
            agent_hash="abc",
            started_at="2026-01-01T00:00:00+00:00",
        )
        right = rollout_id(
            task_name="task-a",
            solution_name="challenger",
            agent_hash="abc",
            started_at="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(left, right)
        self.assertTrue(left.startswith("rol_"))
        self.assertTrue(event_id(rollout_id_value=left, event_index=0, event_type="llm_call").startswith("evt_"))

    def test_llm_event_redacts_auth_fields_and_secret_text(self):
        event = build_llm_event(
            method="POST",
            path="/v1/chat/completions",
            request_payload={
                "model": "miner/model",
                "messages": [{"role": "user", "content": "token secret-token-123"}],
                "Authorization": "Bearer secret-token-123",
            },
            response_payload={"choices": [{"message": {"content": "ok secret-token-123"}}]},
            status_code=200,
            latency_ms=7,
            request_model="miner/model",
            response_model="validator/model",
            usage={"total_tokens": 3},
            cost=0.01,
            started_at="s",
            finished_at="f",
            secrets=("secret-token-123",),
        )
        dumped = json.dumps(event)
        self.assertNotIn("secret-token-123", dumped)
        self.assertNotIn("Authorization", dumped)
        self.assertEqual(event["model_effective"], "validator/model")

    def test_store_update_and_export_gzip_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rollouts"
            record = build_rollout_record(
                rollout_id_value="rol_a",
                task_name="task-a",
                solution_name="challenger-1",
                role=None,
                repo="owner/repo",
                commit_sha="abc",
                issue="fix it",
                agent_hash="agent",
                agent_source=None,
                started_at="s",
                finished_at="f",
                trajectory=[{"type": "command", "cmd": "pytest"}],
                final_patch="diff --git a/x b/x",
                miner_logs="logs",
                steps=1,
                cost=0.1,
                success=True,
                exit_reason="completed",
                runner={"backend": "docker-file"},
            )
            append_rollout(root, record)
            self.assertTrue(update_rollout(root, "task-a", "rol_a", {"duel_id": 7}))
            self.assertEqual(load_task_rollouts(root, "task-a")[0]["duel_id"], 7)

            uploads = []

            def fake_upload(**kwargs):
                with gzip.open(kwargs["local_path"], "rt", encoding="utf-8") as handle:
                    kwargs = {**kwargs, "first_row": json.loads(handle.readline())}
                uploads.append(kwargs)

            config = SimpleNamespace(
                push_rollouts_to_hf=True,
                rollout_hf_dataset="owner/dataset",
                rollout_hf_token_env="HF_TOKEN",
                resolved_rollout_root=lambda: root,
            )
            with patch.dict(os.environ, {"HF_TOKEN": "token"}):
                path_in_repo = export_task_rollouts_to_hf(
                    config=config,
                    task_name="task-a",
                    upload_file=fake_upload,
                )
            self.assertIsNotNone(path_in_repo)
            self.assertEqual(len(uploads), 1)
            self.assertEqual(uploads[0]["first_row"]["rollout_id"], "rol_a")
            self.assertEqual(uploads[0]["first_row"]["duel_id"], 7)
            self.assertEqual(load_export_manifest(root)["tasks"]["task-a"]["hf_path"], path_in_repo)

            with patch.dict(os.environ, {"HF_TOKEN": "token"}):
                repeat_count = export_retired_rollouts_to_hf(
                    config=config,
                    active_task_names=set(),
                    upload_file=fake_upload,
                )
            self.assertEqual(repeat_count, 0)
            self.assertEqual(len(uploads), 1)

    def test_retired_export_skips_active_tasks_and_records_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rollouts"
            for task_name in ("active-task", "retired-task"):
                append_rollout(root, {
                    "schema_version": 1,
                    "rollout_id": f"rol_{task_name}",
                    "task_name": task_name,
                    "trajectory": [],
                    "issue": task_name,
                })
            uploads = []

            def fake_upload(**kwargs):
                uploads.append(kwargs["path_in_repo"])

            config = SimpleNamespace(
                push_rollouts_to_hf=True,
                rollout_hf_dataset="owner/dataset",
                rollout_hf_token_env="HF_TOKEN",
                resolved_rollout_root=lambda: root,
            )
            with patch.dict(os.environ, {"HF_TOKEN": "token"}):
                count = export_retired_rollouts_to_hf(
                    config=config,
                    active_task_names={"active-task"},
                    upload_file=fake_upload,
                )
                repeat_count = export_retired_rollouts_to_hf(
                    config=config,
                    active_task_names={"active-task"},
                    upload_file=fake_upload,
                )

            self.assertEqual(count, 1)
            self.assertEqual(repeat_count, 0)
            self.assertEqual(len(uploads), 1)
            self.assertTrue(uploads[0].endswith("/retired-task.jsonl.gz"))

    def test_retired_export_batches_hf_folder_uploads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rollouts"
            for task_name in ("task-a", "task-b", "task-c"):
                append_rollout(root, {
                    "schema_version": 1,
                    "rollout_id": f"rol_{task_name}",
                    "task_name": task_name,
                    "trajectory": [],
                    "issue": task_name,
                })

            uploads = []

            def fake_upload_folder(**kwargs):
                folder_path = Path(kwargs["folder_path"])
                uploaded = sorted(
                    path.relative_to(folder_path).as_posix()
                    for path in folder_path.rglob("*.jsonl.gz")
                )
                uploads.append({**kwargs, "uploaded": uploaded})

            config = SimpleNamespace(
                push_rollouts_to_hf=True,
                rollout_hf_dataset="owner/dataset",
                rollout_hf_token_env="HF_TOKEN",
                resolved_rollout_root=lambda: root,
            )
            with patch.dict(os.environ, {"HF_TOKEN": "token", "TAU_ROLLOUT_HF_BATCH_SIZE": "10"}):
                count = export_retired_rollouts_to_hf(
                    config=config,
                    active_task_names=set(),
                    upload_folder=fake_upload_folder,
                )

            self.assertEqual(count, 3)
            self.assertEqual(len(uploads), 1)
            self.assertEqual(uploads[0]["task_count"], 3)
            self.assertEqual(len(uploads[0]["uploaded"]), 3)
            self.assertTrue(all(path.startswith("rollouts/") for path in uploads[0]["uploaded"]))
            manifest_tasks = load_export_manifest(root)["tasks"]
            self.assertEqual(set(manifest_tasks), {"task-a", "task-b", "task-c"})

    def test_retired_export_continues_after_task_upload_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rollouts"
            for task_name in ("task-a", "task-b"):
                append_rollout(root, {
                    "schema_version": 1,
                    "rollout_id": f"rol_{task_name}",
                    "task_name": task_name,
                    "trajectory": [],
                    "issue": task_name,
                })

            uploads = []

            def fake_upload(**kwargs):
                if kwargs["task_name"] == "task-a":
                    raise RuntimeError("boom")
                uploads.append(kwargs["path_in_repo"])

            config = SimpleNamespace(
                push_rollouts_to_hf=True,
                rollout_hf_dataset="owner/dataset",
                rollout_hf_token_env="HF_TOKEN",
                resolved_rollout_root=lambda: root,
            )
            with patch.dict(os.environ, {"HF_TOKEN": "token"}):
                count = export_retired_rollouts_to_hf(
                    config=config,
                    active_task_names=set(),
                    upload_file=fake_upload,
                )

            self.assertEqual(count, 1)
            self.assertEqual(len(uploads), 1)
            self.assertTrue(uploads[0].endswith("/task-b.jsonl.gz"))
            manifest_tasks = load_export_manifest(root)["tasks"]
            self.assertNotIn("task-a", manifest_tasks)
            self.assertIn("task-b", manifest_tasks)

    def test_clear_uploaded_rollouts_only_removes_retired_exported_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "rollouts"
            for task_name in ("uploaded-task", "active-task", "local-only-task"):
                append_rollout(root, {
                    "schema_version": 1,
                    "rollout_id": f"rol_{task_name}",
                    "task_name": task_name,
                    "trajectory": [],
                    "issue": task_name,
                })

            mark_task_rollouts_exported(
                root,
                task_name="uploaded-task",
                path_in_repo="rollouts/2026-01-01-00/uploaded-task.jsonl.gz",
            )
            mark_task_rollouts_exported(
                root,
                task_name="active-task",
                path_in_repo="rollouts/2026-01-01-00/active-task.jsonl.gz",
            )

            uploaded_retired = uploaded_local_rollout_task_names(
                root=root,
                active_task_names={"active-task"},
            )
            cleared = clear_uploaded_rollout_tasks(
                root=root,
                active_task_names={"active-task"},
                max_dirs=10,
            )

            self.assertEqual(uploaded_retired, ["uploaded-task"])
            self.assertEqual(cleared, 1)
            self.assertFalse((root / "tasks" / "uploaded-task").exists())
            self.assertTrue((root / "tasks" / "active-task").exists())
            self.assertTrue((root / "tasks" / "local-only-task").exists())
            self.assertIn("uploaded-task", load_export_manifest(root)["tasks"])

    def test_trajectory_events_are_time_ordered_before_ids(self):
        record = build_rollout_record(
            rollout_id_value="rol_order",
            task_name="task-order",
            solution_name="challenger-1",
            role=None,
            repo=None,
            commit_sha=None,
            issue="fix it",
            agent_hash=None,
            agent_source=None,
            started_at="2026-01-01T00:00:00+00:00",
            finished_at="2026-01-01T00:00:03+00:00",
            trajectory=[
                {"type": "llm_call", "started_at": "2026-01-01T00:00:02+00:00"},
                {"type": "command", "started_at": "2026-01-01T00:00:01+00:00"},
            ],
            final_patch="",
            miner_logs=None,
            steps=None,
            cost=None,
            success=False,
            exit_reason="failed",
            runner={"backend": "docker-file"},
        )
        self.assertEqual([event["type"] for event in record["trajectory"]], ["command", "llm_call"])
        self.assertEqual([event["event_index"] for event in record["trajectory"]], [0, 1])

    def test_dpo_and_grpo_rows_are_training_native(self):
        chosen = {
            "task_name": "task-a",
            "rollout_id": "rol_chosen",
            "issue": "fix it",
            "role": "challenger",
            "final_patch": "diff --git a/a b/a",
            "judge": {"challenger_score": 8.0},
            "trajectory": [{"type": "command", "cmd": "pytest"}],
            "success": True,
        }
        rejected = {
            "task_name": "task-a",
            "rollout_id": "rol_rejected",
            "issue": "fix it",
            "role": "king",
            "trajectory": [
                {
                    "type": "llm_call",
                    "response": {"choices": [{"message": {"content": "try this"}}]},
                },
            ],
            "judge": {"king_score": 4.0},
            "success": False,
        }

        pair = dpo_row(task_name="task-a", chosen=chosen, rejected=rejected, source="judge")
        self.assertEqual(pair["chosen"], "diff --git a/a b/a")
        self.assertEqual(pair["rejected"], "try this")
        self.assertEqual(pair["chosen_reward"], 8.0)
        self.assertEqual(pair["rejected_reward"], 4.0)
        self.assertEqual(pair["chosen_rollout"]["trajectory"], chosen["trajectory"])

        group = grpo_row(task_name="task-a", group_id="duel-1", rollouts=[chosen, rejected])
        self.assertEqual(group["prompt"], "fix it")
        self.assertEqual([row["rollout_id"] for row in group["responses"]], ["rol_chosen", "rol_rejected"])

    def test_harness_event_sink_is_not_env_visible_and_overwrites_forged_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
            (repo / "README.md").write_text("hello\n", encoding="utf-8")
            subprocess.run(["git", "add", "README.md"], cwd=repo, check=True, capture_output=True)
            subprocess.run(
                ["git", "-c", "user.email=a@b.c", "-c", "user.name=Tau", "commit", "-m", "init"],
                cwd=repo,
                check=True,
                capture_output=True,
            )
            agent = root / "agent.py"
            agent.write_text(
                """
import os
import subprocess
from pathlib import Path

def solve(repo_path, issue, model, api_base, api_key):
    if os.environ.get("TAU_RUNNER_EVENTS_FILE") is not None:
        raise RuntimeError("event sink leaked into miner env")
    Path(os.environ["TAU_HARNESS_RUNNER"]).with_name("tau_events.jsonl").write_text(
        '{"type":"llm_call","source":"forged"}\\n',
        encoding="utf-8",
    )
    subprocess.run(["python", "-c", "print('event-ok')"])
    return {"success": True}
""".strip()
                + "\n",
                encoding="utf-8",
            )
            prompt = root / "prompt.txt"
            prompt.write_text("fix it", encoding="utf-8")
            runner = root / "runner.py"
            runner.write_text(docker_solver._harness_runner_script(), encoding="utf-8")
            env = os.environ | {
                "TAU_AGENT_FILE": str(agent),
                "TAU_REPO_DIR": str(repo),
                "TAU_PROMPT_FILE": str(prompt),
                "TAU_HARNESS_RUNNER": str(runner),
                "AGENT_MODEL": "test/model",
                "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
                "OPENAI_API_KEY": "test-key",
            }
            env.pop("TAU_RUNNER_EVENTS_FILE", None)
            proc = subprocess.run([sys.executable, str(runner)], env=env, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
            events_path = runner.with_name("tau_events.jsonl")
            rows = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]
            self.assertTrue(rows)
            self.assertFalse(any(row.get("source") == "forged" for row in rows))



if __name__ == "__main__":
    unittest.main()
