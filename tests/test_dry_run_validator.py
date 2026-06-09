"""No-Docker smoke tests for the offline validator dry-run.

Covers the mock chain surface, the offline GitHub client, and the bootstrap
seeding helpers — validated against the *real* validator predicates. The full
Docker duel is the manual ``scripts/dry_run_validator.py`` run.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "src", ROOT / "scripts"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import dry_run_validator as drv  # noqa: E402

import task_pool_manager  # noqa: E402
import tau.bittensor as bt  # noqa: E402
import validate  # noqa: E402
from private_submission import (  # noqa: E402
    accepted_private_submission_entries,
    private_submission_check_passed,
)
from tau.io.github import LocalGitHubClient  # noqa: E402


def _init_git_repo(path: Path) -> str:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(path), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.email", "t@t.co"], check=True)
    subprocess.run(["git", "-C", str(path), "config", "user.name", "t"], check=True)
    (path / "agent.py").write_text("def solve():\n    return 1\n")
    subprocess.run(["git", "-C", str(path), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "init"], check=True)
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


SNAPSHOT = {
    "block": 8_200_000,
    "validator": {"hotkey": "5Validator", "uid": 1},
    "miners": [
        {"hotkey": "5Challenger", "coldkey": "5Coldkey", "uid": 2, "registration_block": 2},
    ],
}


class BittensorMockSurfaceTest(unittest.TestCase):
    def setUp(self):
        bt.init(mode="test", snapshot=SNAPSHOT)

    def tearDown(self):
        bt.init(mode="live")

    def test_subtensor_is_context_manager(self):
        sub = bt.SubtensorApi(websocket_shutdown_timer=0)
        with sub as entered:
            self.assertIs(entered, sub)

    def test_block_and_determine_block_hash(self):
        sub = bt.SubtensorApi()
        self.assertEqual(sub.block, 8_200_000)
        self.assertIsNone(sub.determine_block_hash(None))
        self.assertIsNone(sub.determine_block_hash())

    def test_uid_lookup(self):
        sub = bt.SubtensorApi()
        self.assertEqual(sub.subnets.get_uid_for_hotkey_on_subnet("5Challenger", 66), 2)
        self.assertEqual(sub.subnets.get_uid_for_hotkey_on_subnet("5Validator", 66), 1)
        self.assertIsNone(sub.subnets.get_uid_for_hotkey_on_subnet("unknown", 66))

    def test_neurons_include_burn_validator_and_miner(self):
        sub = bt.SubtensorApi()
        uids = {n.uid for n in sub.neurons.neurons_lite(66)}
        self.assertEqual(uids, {0, 1, 2})

    def test_substrate_registration_block_and_owner(self):
        sub = bt.SubtensorApi()
        reg = sub.substrate.query("SubtensorModule", "BlockAtRegistration", params=[66, 2])
        self.assertEqual(reg.value, 2)
        owner = sub.substrate.query("SubtensorModule", "Owner", params=["5Challenger"])
        self.assertEqual(owner.value, "5Coldkey")

    def test_set_weights_success(self):
        sub = bt.SubtensorApi()
        resp = sub.extrinsics.set_weights(
            wallet=bt.Wallet(), netuid=66, uids=[0, 1, 2], weights=[0.5, 0.3, 0.2]
        )
        self.assertTrue(resp.success)

    def test_empty_snapshot_defaults(self):
        bt.init(mode="test")
        sub = bt.SubtensorApi()
        self.assertEqual(sub.block, 1)
        self.assertEqual({n.uid for n in sub.neurons.neurons_lite(66)}, {0})


class LocalGitHubClientTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self._tmp.name) / "ninja"
        self.sha = _init_git_repo(self.repo)
        self.client = LocalGitHubClient({"unarbos/ninja": self.repo})

    def tearDown(self):
        self._tmp.cleanup()

    def test_repo_exists(self):
        resp = self.client.get("/repos/unarbos/ninja")
        self.assertEqual(resp.status_code, 200)
        self.assertFalse(resp.json()["private"])

    def test_unknown_repo_404(self):
        self.assertEqual(self.client.get("/repos/foo/bar").status_code, 404)

    def test_branch_head(self):
        resp = self.client.get("/repos/unarbos/ninja/branches/master")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["commit"]["sha"], self.sha)

    def test_commit_resolves_full_and_short(self):
        self.assertEqual(self.client.get(f"/repos/unarbos/ninja/commits/{self.sha}").json()["sha"], self.sha)
        self.assertEqual(self.client.get(f"/repos/unarbos/ninja/commits/{self.sha[:10]}").json()["sha"], self.sha)
        self.assertEqual(self.client.get("/repos/unarbos/ninja/commits/deadbeef00").status_code, 404)

    def test_compare_identical(self):
        resp = self.client.get(f"/repos/unarbos/ninja/compare/{self.sha}...master")
        self.assertEqual(resp.json()["status"], "identical")

    def test_contents_base64(self):
        import base64

        resp = self.client.get("/repos/unarbos/ninja/contents/agent.py", params={"ref": "master"})
        body = resp.json()
        self.assertEqual(body["encoding"], "base64")
        self.assertIn(b"def solve", base64.b64decode(body["content"]))

    def test_write_verbs_blocked(self):
        with self.assertRaises(RuntimeError):
            self.client.post("/repos/unarbos/ninja/merges")


class SeedingHelpersTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        bt.init(mode="test", snapshot=SNAPSHOT)

    def tearDown(self):
        bt.init(mode="live")
        self._tmp.cleanup()

    def test_private_submission_seed_passes_checks(self):
        priv = self.root / "private-submissions"
        sid, sha = drv.seed_private_submission(
            root=priv,
            agent_py_text=drv._DEFAULT_AGENT_PY,
            hotkey="5Challenger",
            coldkey="5Coldkey",
            registration_block=2,
        )
        self.assertTrue(
            private_submission_check_passed(
                priv, sid, sha, hotkey="5Challenger",
                signature_verifier=validate._verify_hotkey_signature,
            )
        )
        entries = accepted_private_submission_entries(root=priv)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["hotkey"], "5Challenger")
        self.assertEqual(entries[0]["agent_sha256"], sha)

    def test_private_api_submission_from_entry_yields_submission(self):
        priv = self.root / "private-submissions"
        drv.seed_private_submission(
            root=priv,
            agent_py_text=drv._DEFAULT_AGENT_PY,
            hotkey="5Challenger",
            coldkey="5Coldkey",
            registration_block=2,
        )
        config = drv.build_dry_run_config(
            workspace_root=self.root / "ws",
            ninja_repo=self.root / "ninja",
            chain_snapshot=self.root / "snap.json",
            wallet_name="dryrun",
            wallet_hotkey="default",
            duel_rounds=2,
        )
        # private-submission root defaults to <validate_root>/private-submissions
        config.validate_private_submission_root = priv
        sub = validate._open_subtensor(config)
        with sub as subtensor:
            entry = accepted_private_submission_entries(root=priv)[0]
            submission = validate._private_api_submission_from_entry(
                subtensor=subtensor, config=config, state=None, entry=entry,
            )
        self.assertIsNotNone(submission)
        self.assertEqual(submission.source, "private")
        self.assertEqual(submission.uid, 2)
        self.assertTrue(submission.commitment.startswith("private-submission:"))

    def test_synthesize_minimal_saved_task_is_fillable(self):
        tasks_root = self.root / "workspace" / "tasks"
        task_dir = drv.synthesize_minimal_saved_task(tasks_root=tasks_root)
        self.assertTrue(task_pool_manager.is_complete_saved_task_dir(task_dir))
        self.assertTrue(task_pool_manager.saved_task_can_fill_pool(task_dir))
        self.assertGreaterEqual(
            validate._count_patch_lines(task_dir / "task" / "reference.patch"), 100
        )

    def test_seed_burn_king_state_roundtrips(self):
        validate_root = self.root / "validate"
        ninja = self.root / "ninja"
        state_path = drv.seed_burn_king_state(validate_root=validate_root, ninja_repo=ninja, sha="a" * 40)
        state = validate._load_state(state_path)
        self.assertIsNotNone(state.current_king)
        self.assertTrue(validate._is_burn_king(state.current_king))
        self.assertEqual(state.current_king.repo_url, f"file://{ninja}")
        self.assertEqual(state.current_king.commit_sha, "a" * 40)


class GatingTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def test_nonstatic_pool_is_ready_immediately(self):
        config = drv.build_dry_run_config(
            workspace_root=self.root,
            ninja_repo=self.root / "ninja",
            chain_snapshot=self.root / "snap.json",
            wallet_name="dryrun",
            wallet_hotkey="default",
            duel_rounds=2,
        )
        config.validate_task_pool_static = False
        pool = validate.TaskPool(config.validate_root / "task-pool")
        king = validate.ValidatorSubmission(
            hotkey="burn-uid-0", uid=0, repo_full_name="unarbos/ninja",
            repo_url="file:///tmp/ninja", commit_sha="a" * 40,
            commitment="burn:uid-0", commitment_block=1, source="burn",
        )
        ready, _reason = validate._static_pool_ready_for_king(
            config=config, pool=pool, king=king, pool_label="primary",
        )
        self.assertTrue(ready)


class WiringTest(unittest.TestCase):
    """The dry-run seams in validate.py select the offline implementations."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.config = drv.build_dry_run_config(
            workspace_root=self.root,
            ninja_repo=self.root / "ninja",
            chain_snapshot=self.root / "snap.json",
            wallet_name="dryrun",
            wallet_hotkey="default",
            duel_rounds=2,
        )

    def tearDown(self):
        bt.init(mode="live")
        self._tmp.cleanup()

    def test_github_clients_are_local_in_dry_run(self):
        self.assertIsInstance(validate._build_github_client(self.config), LocalGitHubClient)
        self.assertIsInstance(validate._build_github_merge_client(self.config), LocalGitHubClient)

    def test_repo_url_is_file_for_ninja_in_dry_run(self):
        self.assertEqual(
            validate._repo_url_for("unarbos/ninja", self.config),
            f"file://{self.root / 'ninja'}",
        )
        # other repos still resolve to GitHub HTTPS
        self.assertEqual(
            validate._repo_url_for("other/repo", self.config),
            "https://github.com/other/repo.git",
        )

    def test_open_subtensor_returns_mock_after_init(self):
        bt.init(mode="test", snapshot=SNAPSHOT)
        with validate._open_subtensor(self.config) as subtensor:
            self.assertEqual(subtensor.block, 8_200_000)
            self.assertEqual(subtensor.subnets.get_uid_for_hotkey_on_subnet("5Challenger", 66), 2)


def _utf8_artifact(path: str, text: str) -> dict:
    import hashlib

    return {
        "path": path,
        "encoding": "utf-8",
        "content": text,
        "sha256": hashlib.sha256(text.encode()).hexdigest(),
        "size_bytes": len(text.encode()),
    }


def _make_task_archive_record(*, task_name: str, king_commit: str) -> dict:
    """A self-consistent task-archive record whose expanded king cache passes
    ``_pool_task_has_healthy_king_cache`` for a king at *king_commit*."""
    timeout = validate._effective_pool_task_agent_timeout(cursor_elapsed=0.0, stored_timeout=0)
    matched, sim, total_b = 120, 0.5, 130
    compare_name = validate.derive_compare_name(validate._reference_compare_solution_names("king"))
    reference_patch = "--- a/f.py\n+++ b/f.py\n@@ -0,0 +1,120 @@\n" + "\n".join(f"+l{i}" for i in range(120))
    artifacts = [
        _utf8_artifact("task/task.json", json.dumps({"name": task_name})),
        _utf8_artifact("task/task.txt", "Fix the bug.\n"),
        _utf8_artifact("task/commit.json", json.dumps({"sha": king_commit})),
        _utf8_artifact("task/reference.patch", reference_patch + "\n"),
        _utf8_artifact("task/reference/f.py", "\n".join(f"l{i}" for i in range(120)) + "\n"),
        _utf8_artifact("task/original/f.py", ""),
        _utf8_artifact("solutions/king/solution.diff", reference_patch + "\n"),
        _utf8_artifact("solutions/king/solve.json", json.dumps({"agent_timeout_seconds": timeout, "result": {"exit_reason": "completed"}})),
        _utf8_artifact("solutions/king/repo/.keep", ""),
        _utf8_artifact("solutions/baseline/solution.diff", reference_patch + "\n"),
        _utf8_artifact("solutions/baseline/solve.json", json.dumps({"result": {"exit_reason": "completed", "elapsed_seconds": 1.0}})),
        _utf8_artifact(
            f"comparisons/{compare_name}/compare.json",
            json.dumps({"result": {"matched_changed_lines": matched, "similarity_ratio": sim, "total_changed_lines_b": total_b}}),
        ),
    ]
    pool_task = {
        "task_name": task_name,
        "task_root": "/production/path/to/be/rewritten",
        "creation_block": 1,
        "cursor_elapsed": 0.0,
        "king_lines": matched,
        "king_similarity": sim,
        "baseline_lines": total_b,
        "agent_timeout_seconds": timeout,
        "king_hotkey": "5ProductionKingHotkey",
        "king_commit_sha": king_commit,
    }
    return {"task_name": task_name, "task_root_name": task_name, "pool_task": pool_task, "artifacts": artifacts}


class TaskArchiveTest(unittest.TestCase):
    """Ingesting the HF tasks/ archive (the correct --tasks input)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _write_jsonl(self, rel: str, records: list[dict]) -> Path:
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in records))
        return path

    def test_classify_distinguishes_inputs(self):
        archive = self._write_jsonl("tasks/primary/h_primary.jsonl", [_make_task_archive_record(task_name="validate-a", king_commit="b" * 40)])
        self.assertEqual(drv.classify_tasks_input(archive), "task_archive")
        self.assertEqual(drv.classify_tasks_input(archive.parent), "task_archive")

        rollouts = self._write_jsonl("rollouts/h/validate-x.jsonl", [
            {"role": "king", "task_name": "validate-x", "final_patch": "diff", "issue": "i", "trajectory": []},
        ])
        self.assertEqual(drv.classify_tasks_input(rollouts), "rollouts")
        self.assertEqual(drv.classify_tasks_input(rollouts.parent), "rollouts")

        dirs = self.root / "dirs"
        drv.synthesize_minimal_saved_task(tasks_root=dirs, name="validate-d")
        self.assertEqual(drv.classify_tasks_input(dirs), "task_dirs")

    def test_expand_task_archive_prefills_ready_pool(self):
        king_commit = "c" * 40
        archive = self._write_jsonl(
            "tasks/primary/h_primary.jsonl",
            [_make_task_archive_record(task_name="validate-arch-1", king_commit=king_commit)],
        )
        # duel_rounds=1 -> pool target 1, matching the single seeded task (the
        # bootstrap caps rounds/target to the number of tasks actually seeded).
        config = drv.build_dry_run_config(
            workspace_root=self.root / "ws",
            ninja_repo=self.root / "ninja",
            chain_snapshot=self.root / "snap.json",
            wallet_name="dryrun",
            wallet_hotkey="default",
            duel_rounds=1,
        )
        seeded, commit = drv.expand_task_archive(
            files=drv.archive_jsonl_files(archive),
            tasks_root=config.tasks_root,
            validate_root=config.validate_root,
            target=config.validate_task_pool_target,
            king_hotkey="burn-uid-0",
        )
        self.assertEqual(seeded, 1)
        self.assertEqual(commit, king_commit)
        # artifacts expanded
        self.assertTrue((config.tasks_root / "validate-arch-1" / "task" / "task.txt").is_file())
        self.assertTrue((config.tasks_root / "validate-arch-1" / "solutions" / "king" / "solve.json").is_file())
        # pool json rewritten to workspace + burn-king hotkey
        pool_json = json.loads((config.validate_root / "task-pool" / "validate-arch-1.json").read_text())
        self.assertEqual(pool_json["king_hotkey"], "burn-uid-0")
        self.assertEqual(pool_json["task_root"], str(config.tasks_root / "validate-arch-1"))
        # static pool is ready for the burn king with NO further king solve
        king = validate.ValidatorSubmission(
            hotkey="burn-uid-0", uid=0, repo_full_name="unarbos/ninja",
            repo_url="file:///tmp/ninja", commit_sha=king_commit,
            commitment="burn:uid-0", commitment_block=1, source="burn",
        )
        pool = validate.TaskPool(config.validate_root / "task-pool")
        ready, reason = validate._static_pool_ready_for_king(
            config=config, pool=pool, king=king, pool_label="primary",
        )
        self.assertTrue(ready, reason)

    def test_skips_kings_with_no_patch(self):
        rec = _make_task_archive_record(task_name="validate-empty", king_commit="d" * 40)
        rec["pool_task"]["king_lines"] = 0
        archive = self._write_jsonl("tasks/primary/h_primary.jsonl", [rec])
        config = drv.build_dry_run_config(
            workspace_root=self.root / "ws", ninja_repo=self.root / "ninja",
            chain_snapshot=self.root / "snap.json", wallet_name="d", wallet_hotkey="d", duel_rounds=2,
        )
        seeded, _commit = drv.expand_task_archive(
            files=drv.archive_jsonl_files(archive), tasks_root=config.tasks_root,
            validate_root=config.validate_root, target=1, king_hotkey="burn-uid-0",
        )
        self.assertEqual(seeded, 0)


if __name__ == "__main__":
    unittest.main()
