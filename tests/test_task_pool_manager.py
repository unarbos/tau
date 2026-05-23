import json
import tempfile
import threading
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import task_pool_manager as manager
import validate
from config import RunConfig
from validate import ActiveDuelLease, PoolTask, TaskPool, ValidatorState, ValidatorSubmission


class TaskPoolManagerTest(unittest.TestCase):
    def _task(self, config: RunConfig, name: str = "validate-20260101000000-000001") -> PoolTask:
        task_root = config.tasks_root / name
        task_dir = task_root / "task"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "task.json").write_text(json.dumps({"issue": "fix"}) + "\n")
        (task_dir / "commit.json").write_text(json.dumps({"sha": "abc"}) + "\n")
        (task_dir / "task.txt").write_text("hello\n")
        return PoolTask(
            task_name=name,
            task_root=str(task_root),
            creation_block=1,
            cursor_elapsed=1.0,
            king_lines=1,
            king_similarity=0.5,
            baseline_lines=1,
        )

    def _submission(self, hotkey: str = "king-hotkey") -> ValidatorSubmission:
        return ValidatorSubmission(
            hotkey=hotkey,
            uid=1,
            repo_full_name="owner/repo",
            repo_url="https://github.com/owner/repo",
            commit_sha="a" * 40,
            commitment="owner/repo@" + "a" * 40,
            commitment_block=1,
        )

    def _saved_task(self, config: RunConfig, name: str, baseline_exit: str = "completed") -> Path:
        task_root = config.tasks_root / name
        task_dir = task_root / "task"
        task_dir.mkdir(parents=True, exist_ok=True)
        for artifact in ("task.json", "task.txt", "commit.json", "reference.patch"):
            (task_dir / artifact).write_text("{}\n")
        (task_dir / "original").mkdir(exist_ok=True)
        (task_dir / "reference").mkdir(exist_ok=True)
        baseline_dir = task_root / "solutions" / "baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        (baseline_dir / "solution.diff").write_text("diff\n")
        (baseline_dir / "solve.json").write_text(
            json.dumps({"result": {"exit_reason": baseline_exit, "elapsed_seconds": 1.0}}) + "\n"
        )
        return task_root

    def _age_archive_entry(self, config: RunConfig, task_name: str, seconds: int = 3600) -> None:
        ledger_path = manager.task_archive_ledger_path(config)
        ledger = manager.load_task_archive_ledger(ledger_path)
        entry = dict(ledger["tasks"][task_name])
        entry["updated_at"] = (datetime.now(tz=UTC) - timedelta(seconds=seconds)).isoformat()
        ledger["tasks"][task_name] = entry
        manager.write_task_archive_ledger(ledger_path, ledger)

    def tearDown(self):
        with manager._SAVED_TASK_FILL_LOCK:
            manager._SAVED_TASK_FILL_IN_FLIGHT.clear()

    def test_pool_filler_worker_count_oversubscribes_solve_slots(self):
        config = RunConfig(validate_pool_filler_concurrency=25)

        self.assertEqual(manager.pool_filler_worker_count(config), 75)
        self.assertEqual(manager.pool_filler_executor_workers(config), 150)

    def test_pool_solve_slot_uses_shared_semaphore(self):
        semaphore = threading.BoundedSemaphore(1)

        with manager.pool_solve_slot(semaphore):
            self.assertFalse(semaphore.acquire(blocking=False))

        self.assertTrue(semaphore.acquire(blocking=False))
        semaphore.release()

    def test_pool_solve_slot_allows_noop_context(self):
        with manager.pool_solve_slot(None):
            entered = True

        self.assertTrue(entered)

    def test_archive_quota_is_per_pool_and_hour(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(workspace_root=Path(td), validate_task_archive_per_hour=2)
            for name, pool_label, hour in (
                ("task-a", "primary", "2026-05-22-01"),
                ("task-b", "primary", "2026-05-22-01"),
                ("task-c", "retest", "2026-05-22-01"),
            ):
                manager.record_task_archive_status(
                    config=config,
                    task_name=name,
                    pool_label=pool_label,
                    status="uploaded_deleted",
                    archive_hour_value=hour,
                )

            self.assertEqual(manager.archive_quota_remaining(config, pool_label="primary", hour="2026-05-22-01"), 0)
            self.assertEqual(manager.archive_quota_remaining(config, pool_label="retest", hour="2026-05-22-01"), 1)
            self.assertEqual(manager.archive_quota_remaining(config, pool_label="primary", hour="2026-05-22-02"), 2)

    def test_archive_quota_reservation_blocks_parallel_rotation_work(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(workspace_root=Path(td), validate_task_archive_per_hour=1)

            reserved = manager.reserve_archive_quota(
                config=config,
                task_name="validate-20260101000000-000001",
                pool_label="primary",
                hour="2026-05-22-01",
            )

            self.assertIsNotNone(reserved)
            self.assertEqual(manager.archive_quota_remaining(config, pool_label="primary", hour="2026-05-22-01"), 0)
            self.assertIsNone(
                manager.reserve_archive_quota(
                    config=config,
                    task_name="validate-20260101000000-000002",
                    pool_label="primary",
                    hour="2026-05-22-01",
                )
            )

            manager.release_archive_reservation(config=config, task_name="validate-20260101000000-000001")

            self.assertEqual(manager.archive_quota_remaining(config, pool_label="primary", hour="2026-05-22-01"), 1)

    def test_select_rotation_archive_task_skips_replacement_and_leases(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(workspace_root=Path(td))
            tasks = [
                self._task(config, "validate-20260101000000-000001"),
                self._task(config, "validate-20260101000000-000002"),
                self._task(config, "validate-20260101000000-000003"),
            ]
            tasks[0].creation_block = 1
            tasks[1].creation_block = 2
            tasks[2].creation_block = 3

            selected = manager.select_rotation_archive_task(
                tasks,
                candidate_name=tasks[2].task_name,
                leased_task_names={tasks[0].task_name},
            )

            self.assertIsNotNone(selected)
            assert selected is not None
            self.assertEqual(selected.task_name, tasks[1].task_name)

    def test_full_pool_still_prepares_when_archive_rotation_quota_remains(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(
                workspace_root=Path(td),
                validate_task_pool_target=1,
                validate_task_pool_static=False,
                validate_task_archive_enabled=True,
                validate_task_archive_hf_dataset="owner/dataset",
                validate_task_archive_per_hour=1,
            )
            pool = TaskPool(Path(td) / "pool")
            pool.add(self._task(config), keep=1)

            with patch.dict("os.environ", {"HF_TOKEN": "token"}):
                should_prepare, reason, archive_rotation = manager.pool_should_prepare_task(
                    config=config,
                    pool=pool,
                    king=self._submission(),
                    pool_label="primary",
                )

            self.assertTrue(should_prepare)
            self.assertTrue(archive_rotation)
            self.assertIn("hourly archive rotation quota remaining=1", reason)

            manager.record_task_archive_status(
                config=config,
                task_name="validate-20260101000000-000002",
                pool_label="primary",
                status="uploaded_deleted",
                archive_hour_value=manager.archive_hour(),
            )
            with patch.dict("os.environ", {"HF_TOKEN": "token"}):
                should_prepare, reason, archive_rotation = manager.pool_should_prepare_task(
                    config=config,
                    pool=pool,
                    king=self._submission(),
                    pool_label="primary",
                )

            self.assertFalse(should_prepare)
            self.assertFalse(archive_rotation)
            self.assertIn("hourly archive quota exhausted", reason)

    def test_below_target_pool_still_prepares_after_archive_quota_exhausted(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(
                workspace_root=Path(td),
                validate_task_pool_target=1,
                validate_task_pool_static=False,
                validate_task_archive_enabled=True,
                validate_task_archive_hf_dataset="owner/dataset",
                validate_task_archive_per_hour=1,
            )
            pool = TaskPool(Path(td) / "pool")
            manager.record_task_archive_status(
                config=config,
                task_name="validate-20260101000000-000001",
                pool_label="primary",
                status="uploaded_deleted",
                archive_hour_value=manager.archive_hour(),
            )

            with patch.dict("os.environ", {"HF_TOKEN": "token"}):
                should_prepare, _reason, archive_rotation = manager.pool_should_prepare_task(
                    config=config,
                    pool=pool,
                    king=self._submission(),
                    pool_label="primary",
                )

            self.assertTrue(should_prepare)
            self.assertFalse(archive_rotation)

    def test_generated_task_fill_uses_cursor_baseline_for_timeout_and_reference_for_scoring(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(
                workspace_root=root,
                validate_task_pool_target=1,
                validate_task_pool_static=False,
                validate_task_pool_fill_from_saved=False,
            )
            pool = TaskPool(root / "pool")
            king = self._submission()
            validate._save_state(config.validate_root / "state.json", ValidatorState(current_king=king))
            solved: list[str] = []

            def fake_generate_task_run(*, task_name: str, config: RunConfig):
                task_root = config.tasks_root / task_name
                task_dir = task_root / "task"
                task_dir.mkdir(parents=True, exist_ok=True)
                (task_dir / "task.json").write_text("{}\n")
                (task_dir / "task.txt").write_text("task\n")
                (task_dir / "commit.json").write_text("{}\n")
                (task_dir / "reference.patch").write_text("\n".join(f"+line-{idx}" for idx in range(101)))
                return SimpleNamespace(task_root=str(task_root))

            def fake_solve_task_run(*, task_name: str, solution_name: str, config: RunConfig):
                solved.append(solution_name)
                solution_dir = config.tasks_root / task_name / "solutions" / solution_name
                solution_dir.mkdir(parents=True, exist_ok=True)
                (solution_dir / "solution.diff").write_text("diff\n")
                (solution_dir / "solve.json").write_text(
                    json.dumps(
                        {
                            "agent_timeout_seconds": config.agent_timeout,
                            "result": {
                                "exit_reason": "completed",
                                "elapsed_seconds": 1.0,
                            },
                        }
                    )
                    + "\n"
                )
                return SimpleNamespace(exit_reason="completed", elapsed_seconds=1.0)

            def fake_compare_task_run(*, task_name: str, solution_names: list[str], config: RunConfig):
                self.assertEqual(solution_names, ["king", "reference"])
                compare_dir = config.tasks_root / task_name / "comparisons" / "--vs--".join(solution_names)
                compare_dir.mkdir(parents=True, exist_ok=True)
                (compare_dir / "compare.json").write_text(
                    json.dumps(
                        {
                            "result": {
                                "matched_changed_lines": 1,
                                "similarity_ratio": 0.5,
                                "total_changed_lines_b": 1,
                            }
                        }
                    )
                    + "\n"
                )
                return SimpleNamespace(
                    matched_changed_lines=1,
                    similarity_ratio=0.5,
                    total_changed_lines_b=1,
                    comparison_root=str(compare_dir),
                )

            with patch("task_pool_manager.generate_task_run", side_effect=fake_generate_task_run), patch(
                "task_pool_manager.solve_task_run",
                side_effect=fake_solve_task_run,
            ), patch("task_pool_manager.compare_task_run", side_effect=fake_compare_task_run), patch(
                "task_pool_manager.v._build_agent_config",
                side_effect=lambda config, _sub: config,
            ):
                did_work = manager._prepare_one_task_for_pool(
                    config=config,
                    pool=pool,
                    pool_label="primary",
                    state_lock=threading.Lock(),
                    pool_solve_semaphore=None,
                )

            self.assertTrue(did_work)
            self.assertEqual(solved, ["baseline", "king"])
            self.assertEqual(pool.size(), 1)
            task = pool.list_tasks()[0]
            self.assertEqual(task.cursor_elapsed, 1.0)
            self.assertEqual(task.agent_timeout_seconds, validate._agent_timeout_from_cursor_elapsed(1.0))

    def test_claim_saved_task_for_pool_skips_archived_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(workspace_root=root)
            pool = TaskPool(root / "pool")
            for name in ("validate-20260101000000-000001", "validate-20260101000000-000002"):
                self._saved_task(config, name)
            manager.record_task_archive_status(
                config=config,
                task_name="validate-20260101000000-000001",
                pool_label="primary",
                status="uploaded_deleted",
                archive_hour_value="2026-05-22-01",
            )

            claimed = manager.claim_saved_task_for_pool(config, pool, "primary")
            try:
                self.assertIsNotNone(claimed)
                assert claimed is not None
                self.assertEqual(claimed.name, "validate-20260101000000-000002")
            finally:
                manager.release_saved_task_claim(claimed.name if claimed else None)

    def test_claim_saved_task_for_pool_round_robins_complete_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(workspace_root=root)
            pool = TaskPool(root / "pool")
            for name in ("validate-20260101000000-000001", "validate-20260101000000-000002"):
                self._saved_task(config, name)

            first = manager.claim_saved_task_for_pool(config, pool, "primary")
            try:
                self.assertIsNotNone(first)
                assert first is not None
                self.assertEqual(first.name, "validate-20260101000000-000001")
            finally:
                manager.release_saved_task_claim(first.name if first else None)
            pool.add(
                PoolTask(
                    task_name=first.name,
                    task_root=str(first),
                    creation_block=1,
                    cursor_elapsed=1.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                )
            )

            second = manager.claim_saved_task_for_pool(config, pool, "primary")
            try:
                self.assertIsNotNone(second)
                assert second is not None
                self.assertEqual(second.name, "validate-20260101000000-000002")
            finally:
                manager.release_saved_task_claim(second.name if second else None)

    def test_claim_saved_task_for_pool_skips_partial_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(workspace_root=root)
            pool = TaskPool(root / "pool")
            partial_task = config.tasks_root / "validate-20260101000000-000001" / "task"
            partial_task.mkdir(parents=True)
            (partial_task / "commit.json").write_text("{}\n")

            self.assertIsNone(manager.claim_saved_task_for_pool(config, pool, "primary"))

    def test_claim_saved_task_for_pool_skips_bad_cached_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(workspace_root=root)
            pool = TaskPool(root / "pool")
            self._saved_task(
                config,
                "validate-20260101000000-000001",
                baseline_exit="time_limit_exceeded",
            )
            good_root = self._saved_task(config, "validate-20260101000000-000002")

            claimed = manager.claim_saved_task_for_pool(config, pool, "primary")
            try:
                self.assertIsNotNone(claimed)
                assert claimed is not None
                self.assertEqual(claimed.name, good_root.name)
            finally:
                manager.release_saved_task_claim(claimed.name if claimed else None)

    def test_claim_saved_task_for_pool_skips_complete_task_without_cached_baseline(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(workspace_root=root)
            pool = TaskPool(root / "pool")
            task_dir = config.tasks_root / "validate-20260101000000-000001" / "task"
            task_dir.mkdir(parents=True)
            for artifact in ("task.json", "task.txt", "commit.json", "reference.patch"):
                (task_dir / artifact).write_text("{}\n")
            (task_dir / "original").mkdir()
            (task_dir / "reference").mkdir()

            self.assertIsNone(manager.claim_saved_task_for_pool(config, pool, "primary"))

    def test_task_archive_jsonl_row_embeds_text_and_binary_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(workspace_root=Path(td))
            task = self._task(config)
            (Path(task.task_root) / "task" / "binary.bin").write_bytes(b"\xff\x00")

            row = manager.task_archive_jsonl_row(
                task=task,
                pool_label="primary",
                archive_hour_value="2026-05-22-01",
                king=None,
            )

            artifacts = {item["path"]: item for item in row["artifacts"]}
            self.assertEqual(artifacts["task/task.txt"]["encoding"], "utf-8")
            self.assertEqual(artifacts["task/task.txt"]["content"], "hello\n")
            self.assertEqual(artifacts["task/binary.bin"]["encoding"], "base64")
            self.assertEqual(artifacts["task/binary.bin"]["content_base64"], "/wA=")
            self.assertEqual(row["task_metadata"], {"issue": "fix"})
            self.assertEqual(row["commit_metadata"], {"sha": "abc"})

    def test_archive_upload_success_removes_pool_and_defers_local_delete(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(
                workspace_root=Path(td),
                validate_task_archive_enabled=True,
                validate_task_archive_hf_dataset="owner/dataset",
            )
            pool = TaskPool(Path(td) / "pool")
            task = self._task(config)
            pool.add(task)
            uploaded = []

            with patch.dict("os.environ", {"HF_TOKEN": "token"}):
                manager.archive_pool_task_to_hf_jsonl(
                    config=config,
                    pool=pool,
                    task=task,
                    pool_label="primary",
                    king=None,
                    leased_task_names=set(),
                    upload_jsonl=lambda **kwargs: uploaded.append(kwargs) or "ok",
                )

            self.assertEqual(pool.size(), 0)
            self.assertTrue(Path(task.task_root).exists())
            self.assertEqual(uploaded[0]["dataset_id"], "owner/dataset")
            ledger = manager.load_task_archive_ledger(manager.task_archive_ledger_path(config))
            self.assertEqual(ledger["tasks"][task.task_name]["status"], "uploaded_delete_pending")

            self.assertEqual(manager.retry_pending_archived_task_deletes(config, (pool,)), 0)
            self.assertTrue(Path(task.task_root).exists())
            self._age_archive_entry(config, task.task_name)
            self.assertEqual(manager.retry_pending_archived_task_deletes(config, (pool,)), 1)
            self.assertFalse(Path(task.task_root).exists())
            ledger = manager.load_task_archive_ledger(manager.task_archive_ledger_path(config))
            self.assertEqual(ledger["tasks"][task.task_name]["status"], "uploaded_deleted")

    def test_archive_upload_success_with_active_lease_defers_delete_until_released(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(
                workspace_root=Path(td),
                validate_task_archive_enabled=True,
                validate_task_archive_hf_dataset="owner/dataset",
            )
            pool = TaskPool(Path(td) / "pool")
            task = self._task(config)
            pool.add(task)
            lease = ActiveDuelLease(
                duel_id=1,
                started_at="now",
                king=self._submission("king"),
                challenger=self._submission("challenger"),
                task_names=[task.task_name],
            )
            validate._save_state(config.validate_root / "state.json", ValidatorState(active_duel=lease))

            with patch.dict("os.environ", {"HF_TOKEN": "token"}):
                manager.archive_pool_task_to_hf_jsonl(
                    config=config,
                    pool=pool,
                    task=task,
                    pool_label="primary",
                    king=None,
                    leased_task_names={task.task_name},
                    upload_jsonl=lambda **_kwargs: "ok",
                )

            self.assertEqual(pool.size(), 0)
            self.assertTrue(Path(task.task_root).exists())
            self._age_archive_entry(config, task.task_name)
            self.assertEqual(manager.retry_pending_archived_task_deletes(config, (pool,)), 0)
            self.assertTrue(Path(task.task_root).exists())

            validate._save_state(config.validate_root / "state.json", ValidatorState(active_duel=None))
            self.assertEqual(manager.retry_pending_archived_task_deletes(config, (pool,)), 1)
            self.assertEqual(pool.size(), 0)
            self.assertFalse(Path(task.task_root).exists())

    def test_archive_upload_failure_keeps_pool_and_local_task(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(
                workspace_root=Path(td),
                validate_task_archive_enabled=True,
                validate_task_archive_hf_dataset="owner/dataset",
            )
            pool = TaskPool(Path(td) / "pool")
            task = self._task(config)
            pool.add(task)

            with patch.dict("os.environ", {"HF_TOKEN": "token"}):
                manager.archive_pool_task_to_hf_jsonl(
                    config=config,
                    pool=pool,
                    task=task,
                    pool_label="primary",
                    king=None,
                    leased_task_names=set(),
                    upload_jsonl=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("hf down")),
                )

            self.assertEqual(pool.size(), 1)
            self.assertTrue(Path(task.task_root).exists())
            ledger = manager.load_task_archive_ledger(manager.task_archive_ledger_path(config))
            self.assertEqual(ledger["tasks"][task.task_name]["status"], "upload_failed")
            self.assertIn("hf down", ledger["tasks"][task.task_name]["error"])

    def test_retry_failed_task_upload_completes_archive(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(
                workspace_root=Path(td),
                validate_task_archive_enabled=True,
                validate_task_archive_hf_dataset="owner/dataset",
            )
            pool = TaskPool(Path(td) / "pool")
            task = self._task(config)
            pool.add(task)

            with patch.dict("os.environ", {"HF_TOKEN": "token"}):
                manager.archive_pool_task_to_hf_jsonl(
                    config=config,
                    pool=pool,
                    task=task,
                    pool_label="primary",
                    king=None,
                    leased_task_names=set(),
                    upload_jsonl=lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("hf down")),
                )
                retried = manager.retry_failed_task_uploads(
                    config=config,
                    pools_by_label={"primary": pool},
                    king=None,
                    upload_jsonl=lambda **_kwargs: "ok",
                )

            self.assertEqual(retried, 1)
            self.assertEqual(pool.size(), 0)
            self.assertTrue(Path(task.task_root).exists())
            ledger = manager.load_task_archive_ledger(manager.task_archive_ledger_path(config))
            self.assertEqual(ledger["tasks"][task.task_name]["status"], "uploaded_delete_pending")

            self.assertEqual(manager.retry_pending_archived_task_deletes(config, (pool,)), 0)
            self.assertTrue(Path(task.task_root).exists())
            self._age_archive_entry(config, task.task_name)
            self.assertEqual(manager.retry_pending_archived_task_deletes(config, (pool,)), 1)
            self.assertFalse(Path(task.task_root).exists())
            ledger = manager.load_task_archive_ledger(manager.task_archive_ledger_path(config))
            self.assertEqual(ledger["tasks"][task.task_name]["status"], "uploaded_deleted")

    def test_archive_upload_defers_delete_when_task_is_leased_during_upload(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(
                workspace_root=Path(td),
                validate_task_archive_enabled=True,
                validate_task_archive_hf_dataset="owner/dataset",
            )
            pool = TaskPool(Path(td) / "pool")
            task = self._task(config)
            pool.add(task)

            def upload_and_lease(**_kwargs):
                lease = ActiveDuelLease(
                    duel_id=1,
                    started_at="now",
                    king=self._submission("king"),
                    challenger=self._submission("challenger"),
                    task_names=[task.task_name],
                )
                validate._save_state(config.validate_root / "state.json", ValidatorState(active_duel=lease))
                return "ok"

            with patch.dict("os.environ", {"HF_TOKEN": "token"}):
                manager.archive_pool_task_to_hf_jsonl(
                    config=config,
                    pool=pool,
                    task=task,
                    pool_label="primary",
                    king=None,
                    leased_task_names=set(),
                    upload_jsonl=upload_and_lease,
                )

            self.assertEqual(pool.size(), 0)
            self.assertTrue(Path(task.task_root).exists())
            self._age_archive_entry(config, task.task_name)
            self.assertEqual(manager.retry_pending_archived_task_deletes(config, (pool,)), 0)
            self.assertTrue(Path(task.task_root).exists())

            validate._save_state(config.validate_root / "state.json", ValidatorState(active_duel=None))
            self.assertEqual(manager.retry_pending_archived_task_deletes(config, (pool,)), 1)
            self.assertFalse(Path(task.task_root).exists())

    def test_cleanup_old_task_workspaces_preserves_pool_and_active_duel_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(workspace_root=Path(td), validate_task_cleanup_min_age_seconds=0)
            pool = TaskPool(Path(td) / "pool")
            kept_pool_task = self._task(config, "validate-20260101000000-000001")
            pool.add(kept_pool_task)
            active_task = self._task(config, "validate-20260101000000-000002")
            old_task = self._task(config, "validate-20260101000000-000003")
            lease = ActiveDuelLease(
                duel_id=1,
                started_at="now",
                king=self._submission("king"),
                challenger=self._submission("challenger"),
                task_names=[active_task.task_name],
            )
            validate._save_state(config.validate_root / "state.json", ValidatorState(active_duel=lease))

            manager.cleanup_old_task_workspaces(config, (pool,))

            self.assertTrue(Path(kept_pool_task.task_root).exists())
            self.assertTrue(Path(active_task.task_root).exists())
            self.assertFalse(Path(old_task.task_root).exists())

    def test_cleanup_old_task_workspaces_preserves_archive_delete_pending_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(
                workspace_root=Path(td),
                validate_task_archive_enabled=True,
                validate_task_archive_hf_dataset="owner/dataset",
                validate_task_cleanup_min_age_seconds=0,
            )
            pool = TaskPool(Path(td) / "pool")
            task = self._task(config)
            pool.add(task)

            with patch.dict("os.environ", {"HF_TOKEN": "token"}):
                manager.archive_pool_task_to_hf_jsonl(
                    config=config,
                    pool=pool,
                    task=task,
                    pool_label="primary",
                    king=None,
                    leased_task_names=set(),
                    upload_jsonl=lambda **_kwargs: "ok",
                )

            manager.cleanup_old_task_workspaces(config, (pool,))

            self.assertTrue(Path(task.task_root).exists())
            self._age_archive_entry(config, task.task_name)
            self.assertEqual(manager.retry_pending_archived_task_deletes(config, (pool,)), 1)
            self.assertFalse(Path(task.task_root).exists())

    def test_append_hf_jsonl_reraises_transient_download_failure(self):
        from huggingface_hub.errors import LocalEntryNotFoundError

        with patch(
            "huggingface_hub.hf_hub_download",
            side_effect=LocalEntryNotFoundError("offline and not cached"),
        ):
            with self.assertRaises(LocalEntryNotFoundError):
                manager.append_hf_dataset_jsonl(
                    dataset_id="owner/dataset",
                    token="token",
                    path_in_repo="tasks/primary/2026-05-22-01.jsonl",
                    row={"task_name": "task-a"},
                )

    def test_hf_download_missing_detection_excludes_local_cache_miss(self):
        from huggingface_hub.errors import EntryNotFoundError, LocalEntryNotFoundError

        self.assertTrue(manager.hf_download_error_is_missing(EntryNotFoundError("missing")))
        self.assertFalse(manager.hf_download_error_is_missing(LocalEntryNotFoundError("offline")))

    def test_delete_retry_completes_after_local_delete_failure(self):
        with tempfile.TemporaryDirectory() as td:
            config = RunConfig(
                workspace_root=Path(td),
                validate_task_archive_enabled=True,
                validate_task_archive_hf_dataset="owner/dataset",
            )
            pool = TaskPool(Path(td) / "pool")
            task = self._task(config)
            pool.add(task)

            with patch.dict("os.environ", {"HF_TOKEN": "token"}):
                manager.archive_pool_task_to_hf_jsonl(
                    config=config,
                    pool=pool,
                    task=task,
                    pool_label="primary",
                    king=None,
                    leased_task_names=set(),
                    upload_jsonl=lambda **_kwargs: "ok",
                )

            self._age_archive_entry(config, task.task_name)
            with patch("task_pool_manager.shutil.rmtree", side_effect=RuntimeError("busy")):
                self.assertEqual(manager.retry_pending_archived_task_deletes(config, (pool,)), 0)

            self.assertTrue(Path(task.task_root).exists())
            ledger = manager.load_task_archive_ledger(manager.task_archive_ledger_path(config))
            self.assertEqual(ledger["tasks"][task.task_name]["status"], "uploaded_delete_pending")
            self.assertIn("busy", ledger["tasks"][task.task_name]["error"])

            self._age_archive_entry(config, task.task_name)
            self.assertEqual(manager.retry_pending_archived_task_deletes(config, (pool,)), 1)
            self.assertFalse(Path(task.task_root).exists())


if __name__ == "__main__":
    unittest.main()
