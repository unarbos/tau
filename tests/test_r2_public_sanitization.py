import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from r2 import (
    _is_public_task_leakage_key,
    duel_to_summary,
    publish_duel_data,
    publish_round_data,
    publish_training_data,
)
from workspace import build_compare_paths, build_solution_paths, build_task_paths


class FakeS3Client:
    def __init__(self):
        self.puts = []
        self.deletes = []

    def put_object(self, **kwargs):
        self.puts.append(kwargs)
        return {}

    def delete_object(self, **kwargs):
        self.deletes.append(kwargs)
        return {}


def _json_body(put):
    body = put["Body"]
    if isinstance(body, bytes):
        body = body.decode()
    return json.loads(body)


class R2PublicSanitizationTest(unittest.TestCase):
    def test_publish_round_data_keeps_requested_public_round_artifacts(self):
        client = FakeS3Client()
        with tempfile.TemporaryDirectory() as tmp:
            tasks_root = Path(tmp)
            task_paths = build_task_paths(tasks_root, "validate-1")
            task_paths.task_dir.mkdir(parents=True)
            task_paths.solutions_dir.mkdir()
            task_paths.comparisons_dir.mkdir()

            task_paths.task_json_path.write_text(
                json.dumps(
                    {
                        "repo_full_name": "source/repo",
                        "commit_sha": "target-sha",
                        "task": {"prompt_text": "private task prompt", "title": "private title"},
                    }
                )
            )
            task_paths.task_txt_path.write_text("private task prompt\n")
            task_paths.reference_patch_path.write_text("private reference patch\n")
            task_paths.commit_path.write_text(
                json.dumps({"commit_sha": "target-sha", "combined_patch": "private reference patch"})
            )

            for name in ("baseline", "king", "challenger"):
                sol_paths = build_solution_paths(task_paths, name)
                sol_paths.root.mkdir(parents=True)
                sol_paths.solution_diff_path.write_text(f"{name} public diff\n")
                sol_paths.rollout_jsonl_path.write_text('{"prompt":"private task prompt"}\n')
                sol_paths.solve_json_path.write_text(
                    json.dumps(
                        {
                            "stage": "solve",
                            "task_name": "validate-1",
                            "solution_name": name,
                            "agent_source": {
                                "commit_sha": "agent-sha",
                                "local_path": "/private/agent-cache/agent.py",
                                "raw": "source/repo@agent-sha",
                            },
                            "repo_full_name": "source/repo",
                            "commit_sha": "target-sha",
                            "result": {
                                "raw_output": "private task prompt",
                                "rollout_format": "single-file-json",
                                "solution_diff": f"{name} public diff\n",
                                "session_id": "internal-session",
                                "rollout_filename": "rollout.jsonl",
                                "model": "solver/model",
                                "exit_reason": "completed",
                                "error_summary": "solver_error: returncode=1; harness_json=no; patch=empty",
                                "error_details": {
                                    "failure_kind": "no_harness_json",
                                    "returncode": 1,
                                    "stdout_bytes": 48,
                                    "stderr_bytes": 0,
                                },
                                "total_tokens": 123,
                            },
                        }
                    )
                )

            for cmp_name in ("king--vs--baseline", "challenger--vs--baseline", "king--vs--challenger"):
                cmp_paths = build_compare_paths(task_paths, cmp_name)
                cmp_paths.root.mkdir(parents=True)
                cmp_paths.compare_json_path.write_text(
                    json.dumps(
                        {
                            "repo_full_name": "source/repo",
                            "commit_sha": "target-sha",
                            "result": {"similarity_ratio": 0.5},
                        }
                    )
                )

            with patch("r2._get_s3_client", return_value=client):
                self.assertTrue(
                    publish_round_data(
                        duel_id=7,
                        task_name="validate-1",
                        tasks_root=tasks_root,
                    )
                )

        put_keys = {item["Key"] for item in client.puts}
        all_uploaded = "\n".join(
            (item["Body"].decode() if isinstance(item["Body"], bytes) else str(item["Body"]))
            for item in client.puts
        )

        self.assertNotIn("sn66/duels/000007/rounds/validate-1/task.txt", put_keys)
        self.assertNotIn("sn66/duels/000007/rounds/validate-1/task.json", put_keys)
        self.assertNotIn("sn66/duels/000007/rounds/validate-1/reference.patch", put_keys)
        self.assertNotIn("sn66/duels/000007/rounds/validate-1/commit.json", put_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/solutions/king.diff", put_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/solutions/challenger.diff", put_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/solutions/king.solve.json", put_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/solutions/challenger.solve.json", put_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/comparisons/king--vs--reference.json", put_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/comparisons/challenger--vs--reference.json", put_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/comparisons/king--vs--challenger.json", put_keys)
        self.assertNotIn("sn66/duels/000007/rounds/validate-1/solutions/baseline.diff", put_keys)
        self.assertNotIn("sn66/duels/000007/rounds/validate-1/solutions/baseline.solve.json", put_keys)
        self.assertFalse(any(key.endswith(".rollout.jsonl.gz") for key in put_keys))
        self.assertNotIn("private task prompt", all_uploaded)
        self.assertNotIn("private reference patch", all_uploaded)
        self.assertIn("king public diff", all_uploaded)
        self.assertIn("challenger public diff", all_uploaded)
        self.assertNotIn("baseline public diff", all_uploaded)
        self.assertNotIn("target-sha", all_uploaded)
        self.assertNotIn("agent-sha", all_uploaded)
        self.assertNotIn("/private/agent-cache", all_uploaded)

        solve_put = next(item for item in client.puts if item["Key"].endswith("/solutions/king.solve.json"))
        solve_payload = _json_body(solve_put)
        self.assertNotIn("agent_source", solve_payload)
        self.assertNotIn("repo_full_name", solve_payload)
        self.assertNotIn("commit_sha", solve_payload)
        self.assertNotIn("raw_output", solve_payload["result"])
        self.assertNotIn("rollout_format", solve_payload["result"])
        self.assertNotIn("solution_diff", solve_payload["result"])
        self.assertNotIn("session_id", solve_payload["result"])
        self.assertNotIn("rollout_filename", solve_payload["result"])
        self.assertEqual(solve_payload["result"]["model"], "solver/model")
        self.assertEqual(
            solve_payload["result"]["error_summary"],
            "solver_error: returncode=1; harness_json=no; patch=empty",
        )
        self.assertEqual(solve_payload["result"]["error_details"]["failure_kind"], "no_harness_json")
        self.assertEqual(solve_payload["result"]["error_details"]["stdout_bytes"], 48)

        compare_put = next(
            item
            for item in client.puts
            if item["Key"].endswith("/comparisons/king--vs--reference.json")
        )
        compare_payload = _json_body(compare_put)
        self.assertNotIn("repo_full_name", compare_payload)
        self.assertNotIn("commit_sha", compare_payload)
        self.assertEqual(compare_payload["result"]["similarity_ratio"], 0.5)

        deleted_keys = {item["Key"] for item in client.deletes}
        self.assertIn("sn66/duels/000007/rounds/validate-1/task.txt", deleted_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/reference.patch", deleted_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/solutions/baseline.diff", deleted_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/solutions/baseline.solve.json", deleted_keys)
        self.assertIn("sn66/duels/000007/rounds/validate-1/solutions/king.rollout.jsonl.gz", deleted_keys)

    def test_publish_duel_data_strips_private_round_fields(self):
        client = FakeS3Client()
        duel = {
            "duel_id": 9,
            "rounds": [
                {
                    "task_name": "validate-1",
                    "winner": "king",
                    "task_root": "/private/task/root",
                    "king_compare_root": "/private/king/compare",
                    "challenger_compare_root": "/private/challenger/compare",
                    "llm_judge_rationale": "King wins because the implementation handles validation; challenger misses the error path.",
                    "llm_judge_rounds": [
                        {
                            "round": 1,
                            "model": "judge-a",
                            "shared_message": {"counterpoints": ["public"]},
                            "final_decision": {"winner": "king", "king_score": 0.8},
                        }
                    ],
                    "king_score": 0.8,
                }
            ],
        }

        with patch("r2._get_s3_client", return_value=client):
            self.assertTrue(publish_duel_data(duel_id=9, duel_dict=duel))

        payload = _json_body(client.puts[0])
        round_payload = payload["rounds"][0]
        self.assertNotIn("task_root", round_payload)
        self.assertNotIn("king_compare_root", round_payload)
        self.assertNotIn("challenger_compare_root", round_payload)
        self.assertEqual(
            round_payload["llm_judge_rationale"],
            "King wins because the implementation handles validation; challenger misses the error path.",
        )
        self.assertEqual(round_payload["king_score"], 0.8)
        self.assertEqual(
            round_payload["llm_judge_rounds"],
            [{"round": 1, "model": "judge-a", "shared_message": {"counterpoints": ["public"]}}],
        )

    def test_duel_summary_keeps_public_judge_rationale(self):
        summary = duel_to_summary(
            {
                "duel_id": 12,
                "king_before": {},
                "challenger": {},
                "rounds": [
                    {
                        "task_name": "validate-1",
                        "winner": "king",
                        "error": None,
                        "llm_judge_rationale": "King handles validation; challenger misses the error path.",
                        "llm_judge_models": ["judge-a", "judge-b"],
                        "llm_judge_consensus_status": "agreed",
                        "llm_judge_consensus_round": 2,
                        "challenger_exit_reason": "solver_error",
                        "challenger_error_summary": "solver_error: returncode=1; harness_json=no; patch=empty",
                        "challenger_error_details": {
                            "failure_kind": "no_harness_json",
                            "request_count": 21,
                        },
                        "challenger_agent_timeout_seconds": 120,
                        "llm_judge_rounds": [
                            {
                                "round": 1,
                                "model": "judge-a",
                                "shared_message": {"counterpoints": ["public"]},
                                "final_decision": {"winner": "king"},
                            }
                        ],
                        "king_score": 0.8,
                    }
                ],
            }
        )

        self.assertEqual(
            summary["rounds"][0]["llm_judge_rationale"],
            "King handles validation; challenger misses the error path.",
        )
        self.assertEqual(summary["rounds"][0]["llm_judge_models"], ["judge-a", "judge-b"])
        self.assertEqual(summary["rounds"][0]["llm_judge_consensus_status"], "agreed")
        self.assertEqual(summary["rounds"][0]["llm_judge_consensus_round"], 2)
        self.assertEqual(summary["rounds"][0]["challenger_exit_reason"], "solver_error")
        self.assertEqual(
            summary["rounds"][0]["challenger_error_summary"],
            "solver_error: returncode=1; harness_json=no; patch=empty",
        )
        self.assertEqual(
            summary["rounds"][0]["challenger_error_details"],
            {"failure_kind": "no_harness_json", "request_count": 21},
        )
        self.assertEqual(summary["rounds"][0]["challenger_agent_timeout_seconds"], 120)
        self.assertEqual(summary["rounds"][0]["llm_judge_rounds"][0]["model"], "judge-a")
        self.assertEqual(
            summary["rounds"][0]["llm_judge_rounds"],
            [{"round": 1, "model": "judge-a", "shared_message": {"counterpoints": ["public"]}}],
        )

    def test_duel_summary_links_pr_submission_to_pr_not_fork_repo(self):
        summary = duel_to_summary(
            {
                "duel_id": 13,
                "king_before": {},
                "challenger": {
                    "uid": 42,
                    "hotkey": "miner-hotkey",
                    "repo_full_name": "miner/ninja",
                    "commit_sha": "a" * 40,
                    "commitment": "github-pr:unarbos/ninja#7@" + "a" * 40,
                    "pr_number": 7,
                    "base_repo_full_name": "unarbos/ninja",
                },
                "rounds": [],
            }
        )

        self.assertEqual(summary["challenger_repo"], "miner/ninja")
        self.assertEqual(summary["challenger_pr_url"], "https://github.com/unarbos/ninja/pull/7")
        self.assertEqual(summary["challenger_repo_url"], "https://github.com/unarbos/ninja/pull/7")

    def test_duel_summary_marks_confirmation_retests(self):
        summary = duel_to_summary(
            {
                "duel_id": 43,
                "king_before": {},
                "challenger": {},
                "rounds": [],
                "task_set_phase": "confirmation_retest",
                "manual_retest_of_duel_id": 41,
                "confirmation_of_duel_id": 42,
                "confirmation_failure_reason": "confirmation retest duel 43 aborted",
            }
        )

        self.assertEqual(summary["task_set_phase"], "confirmation_retest")
        self.assertEqual(summary["manual_retest_of_duel_id"], 41)
        self.assertEqual(summary["confirmation_of_duel_id"], 42)
        self.assertEqual(
            summary["confirmation_failure_reason"],
            "confirmation retest duel 43 aborted",
        )

    def test_publish_training_data_deletes_legacy_public_file_without_uploading(self):
        client = FakeS3Client()

        with patch("r2._get_s3_client", return_value=client):
            self.assertFalse(
                publish_training_data(
                    duel_id=11,
                    duel_dict={"rounds": []},
                    tasks_root=Path("/unused"),
                )
            )

        self.assertEqual(client.puts, [])
        self.assertEqual(client.deletes[0]["Key"], "sn66/duels/000011/training.jsonl")

    def test_public_task_leakage_key_detection_covers_legacy_public_objects(self):
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/task.txt"))
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/reference.patch"))
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/commit.json"))
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/task.json"))
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/solutions/baseline.solve.json"))
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/solutions/baseline.diff"))
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/solutions/king.rollout.jsonl.gz"))
        self.assertTrue(_is_public_task_leakage_key("sn66/duels/000001/training.jsonl"))
        self.assertFalse(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/solutions/king.solve.json"))
        self.assertFalse(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/solutions/challenger.solve.json"))
        self.assertFalse(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/solutions/king.diff"))
        self.assertFalse(_is_public_task_leakage_key("sn66/duels/000001/rounds/a/solutions/challenger.diff"))
        self.assertFalse(_is_public_task_leakage_key("sn66/dashboard.json"))


if __name__ == "__main__":
    unittest.main()
