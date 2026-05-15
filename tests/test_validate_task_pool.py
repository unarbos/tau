import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import wait as futures_wait
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import validate
from config import RunConfig
from validate import PoolTask, TaskPool, TaskPoolRefreshBudget, _prepare_validate_paths


class TaskPoolTest(unittest.TestCase):
    @staticmethod
    def _write_minimal_task_metadata(task_root: Path) -> None:
        task_dir = task_root / "task"
        task_dir.mkdir(parents=True, exist_ok=True)
        (task_dir / "task.json").write_text("{}\n")
        (task_dir / "commit.json").write_text("{}\n")

    @classmethod
    def _write_healthy_king_cache(
        cls,
        *,
        config: RunConfig,
        task_name: str,
        king_lines: int,
        king_similarity: float,
        baseline_lines: int,
        king_agent_timeout_seconds: int = 120,
    ) -> None:
        task_root = config.tasks_root / task_name
        cls._write_minimal_task_metadata(task_root)
        baseline_dir = task_root / "solutions" / "baseline"
        king_dir = task_root / "solutions" / "king"
        compare_dir = task_root / "comparisons" / "king--vs--baseline"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        king_dir.mkdir(parents=True, exist_ok=True)
        compare_dir.mkdir(parents=True, exist_ok=True)
        (baseline_dir / "solve.json").write_text("{}\n")
        (baseline_dir / "solution.diff").write_text("diff\n")
        (king_dir / "solve.json").write_text(
            json.dumps({"agent_timeout_seconds": king_agent_timeout_seconds}) + "\n"
        )
        (king_dir / "solution.diff").write_text("king diff\n")
        (compare_dir / "compare.json").write_text(
            json.dumps(
                {
                    "result": {
                        "matched_changed_lines": king_lines,
                        "similarity_ratio": king_similarity,
                        "total_changed_lines_b": baseline_lines,
                    }
                }
            )
        )

    def tearDown(self):
        with validate._POOL_GENERATION_BACKOFF_LOCK:
            validate._pool_generation_backoff_until = 0.0
        with validate._SAVED_TASK_FILL_LOCK:
            validate._SAVED_TASK_FILL_IN_FLIGHT.clear()

    def test_prepare_validate_paths_creates_primary_and_retest_pools(self):
        with tempfile.TemporaryDirectory() as td:
            paths = _prepare_validate_paths(Path(td))

            self.assertTrue(paths.pool_dir.exists())
            self.assertTrue(paths.retest_pool_dir.exists())
            self.assertNotEqual(paths.pool_dir, paths.retest_pool_dir)

    def test_claim_saved_task_for_pool_round_robins_complete_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(workspace_root=root)
            pool = TaskPool(root / "pool")
            for name in ("validate-20260101000000-000001", "validate-20260101000000-000002"):
                task_dir = config.tasks_root / name / "task"
                task_dir.mkdir(parents=True)
                for artifact in ("task.json", "task.txt", "commit.json", "reference.patch"):
                    (task_dir / artifact).write_text("{}\n")

            first = validate._claim_saved_task_for_pool(config, pool, "primary")
            try:
                self.assertIsNotNone(first)
                assert first is not None
                self.assertEqual(first.name, "validate-20260101000000-000001")
            finally:
                validate._release_saved_task_claim(first.name if first else None)
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

            second = validate._claim_saved_task_for_pool(config, pool, "primary")
            try:
                self.assertIsNotNone(second)
                assert second is not None
                self.assertEqual(second.name, "validate-20260101000000-000002")
            finally:
                validate._release_saved_task_claim(second.name if second else None)

    def test_claim_saved_task_for_pool_skips_partial_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(workspace_root=root)
            pool = TaskPool(root / "pool")
            partial_task = config.tasks_root / "validate-20260101000000-000001" / "task"
            partial_task.mkdir(parents=True)
            (partial_task / "commit.json").write_text("{}\n")

            self.assertIsNone(validate._claim_saved_task_for_pool(config, pool, "primary"))

    def test_pool_filler_continues_while_duel_is_gathering(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(
                workspace_root=root,
                validate_task_pool_fill_from_saved=True,
                validate_task_pool_target=1,
            )
            pool = TaskPool(root / "pool")
            task_name = "validate-20260101000000-000001"
            task_root = config.tasks_root / task_name
            task_dir = task_root / "task"
            task_dir.mkdir(parents=True)
            for artifact in ("task.json", "task.txt", "commit.json"):
                (task_dir / artifact).write_text("{}\n")
            (task_dir / "reference.patch").write_text("+line\n" * (validate._MIN_PATCH_LINES + 1))
            baseline_dir = task_root / "solutions" / "baseline"
            baseline_dir.mkdir(parents=True)
            (baseline_dir / "solution.diff").write_text("diff\n")
            (baseline_dir / "solve.json").write_text(
                json.dumps({"result": {"exit_reason": "completed", "elapsed_seconds": 1.0}})
            )
            king = validate.ValidatorSubmission(
                hotkey="king-hotkey",
                uid=1,
                repo_full_name="king/ninja",
                repo_url="https://github.com/king/ninja",
                commit_sha="a" * 40,
                commitment="king/ninja@" + "a" * 40,
                commitment_block=1,
                source="chain",
            )
            challenger = validate.ValidatorSubmission(
                hotkey="challenger-hotkey",
                uid=2,
                repo_full_name="challenger/ninja",
                repo_url="https://github.com/challenger/ninja",
                commit_sha="b" * 40,
                commitment="challenger/ninja@" + "b" * 40,
                commitment_block=1,
                source="chain",
            )
            state = validate.ValidatorState(
                current_king=king,
                active_duel=validate.ActiveDuelLease(
                    duel_id=57,
                    started_at="2026-01-01T00:00:00+00:00",
                    king=king,
                    challenger=challenger,
                    status="gathering_tasks",
                ),
            )
            stop_event = threading.Event()

            def compare_once(**_kwargs):
                stop_event.set()
                return SimpleNamespace(
                    matched_changed_lines=3,
                    similarity_ratio=0.5,
                    total_changed_lines_b=3,
                )

            with (
                patch("validate.solve_task_run", return_value=SimpleNamespace(exit_reason="completed")),
                patch("validate.compare_task_run", side_effect=compare_once),
                patch("validate._open_subtensor", side_effect=RuntimeError("offline")),
            ):
                validate._pool_filler_loop(
                    config,
                    state,
                    pool,
                    stop_event,
                    threading.Lock(),
                    threading.Event(),
                    pool_label="primary",
                )

            self.assertEqual(pool.size(), 1)
            self.assertEqual(pool.list_tasks()[0].task_name, task_name)

    def test_take_returns_fastest_cached_task(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pool = TaskPool(root / "pool")
            slow_root = root / "slow"
            fast_root = root / "fast"
            slow_root.mkdir()
            fast_root.mkdir()
            for task_root in (slow_root, fast_root):
                baseline_dir = task_root / "solutions" / "baseline"
                baseline_dir.mkdir(parents=True)
                (baseline_dir / "solve.json").write_text("{}\n")
                (baseline_dir / "solution.diff").write_text("diff\n")
            pool.add(
                PoolTask(
                    task_name="slow",
                    task_root=str(slow_root),
                    creation_block=20,
                    cursor_elapsed=300.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                )
            )
            pool.add(
                PoolTask(
                    task_name="fast",
                    task_root=str(fast_root),
                    creation_block=20,
                    cursor_elapsed=20.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                )
            )

            task = pool.take(min_block=10)

        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.task_name, "fast")

    def test_take_reuses_cached_task_older_than_min_block(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pool = TaskPool(root / "pool")
            task_root = root / "cached"
            task_root.mkdir()
            baseline_dir = task_root / "solutions" / "baseline"
            baseline_dir.mkdir(parents=True)
            (baseline_dir / "solve.json").write_text("{}\n")
            (baseline_dir / "solution.diff").write_text("diff\n")
            pool.add(
                PoolTask(
                    task_name="cached",
                    task_root=str(task_root),
                    creation_block=20,
                    cursor_elapsed=20.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                )
            )

            task = pool.take(min_block=100)

        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.task_name, "cached")

    def test_gather_pool_tasks_respects_initial_exclude_set(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pool = TaskPool(root / "pool")
            for name, elapsed in (
                ("already-selected", 1.0),
                ("new-task", 2.0),
            ):
                task_root = root / name
                baseline_dir = task_root / "solutions" / "baseline"
                baseline_dir.mkdir(parents=True)
                (baseline_dir / "solve.json").write_text("{}\n")
                (baseline_dir / "solution.diff").write_text("diff\n")
                pool.add(
                    PoolTask(
                        task_name=name,
                        task_root=str(task_root),
                        creation_block=20,
                        cursor_elapsed=elapsed,
                        king_lines=1,
                        king_similarity=0.1,
                        baseline_lines=1,
                    )
                )

            tasks = validate._gather_pool_tasks(
                pool,
                1,
                min_block=10,
                timeout=1,
                exclude={"already-selected"},
            )

        self.assertEqual([task.task_name for task in tasks], ["new-task"])

    def test_pool_task_metadata_tracks_king_and_can_be_listed(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td))
            pool.add(
                PoolTask(
                    task_name="cached",
                    task_root="/tmp/cached",
                    creation_block=20,
                    cursor_elapsed=20.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                    king_hotkey="hotkey-a",
                    king_commit_sha="a" * 40,
                )
            )

            tasks = pool.list_tasks()

            self.assertEqual(len(tasks), 1)
            self.assertEqual(tasks[0].king_hotkey, "hotkey-a")
            self.assertEqual(tasks[0].king_commit_sha, "a" * 40)
            self.assertTrue(pool.remove("cached"))
            self.assertEqual(pool.list_tasks(), [])

    def test_add_with_keep_prunes_oldest_tasks_back_to_target(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td))
            first = PoolTask(
                task_name="validate-20260101000000-000001",
                task_root="/tmp/task-1",
                creation_block=1,
                cursor_elapsed=10.0,
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
            )
            second = PoolTask(
                task_name="validate-20260101000000-000002",
                task_root="/tmp/task-2",
                creation_block=1,
                cursor_elapsed=20.0,
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
            )
            third = PoolTask(
                task_name="validate-20260101000000-000003",
                task_root="/tmp/task-3",
                creation_block=1,
                cursor_elapsed=30.0,
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
            )

            self.assertEqual(pool.add(first, keep=2), 0)
            self.assertEqual(pool.add(second, keep=2), 0)
            self.assertEqual(pool.add(third, keep=2), 1)
            self.assertEqual(pool.size(), 2)
            self.assertEqual(
                {task.task_name for task in pool.list_tasks()},
                {
                    "validate-20260101000000-000002",
                    "validate-20260101000000-000003",
                },
            )

    def test_add_with_keep_prunes_requested_tasks_before_oldest(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td))
            first = PoolTask(
                task_name="validate-20260101000000-000001",
                task_root="/tmp/task-1",
                creation_block=1,
                cursor_elapsed=10.0,
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
            )
            bad_middle = PoolTask(
                task_name="validate-20260101000000-000002",
                task_root="/tmp/task-2",
                creation_block=1,
                cursor_elapsed=20.0,
                king_lines=0,
                king_similarity=0.0,
                baseline_lines=1,
            )
            third = PoolTask(
                task_name="validate-20260101000000-000003",
                task_root="/tmp/task-3",
                creation_block=1,
                cursor_elapsed=30.0,
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
            )
            replacement = PoolTask(
                task_name="validate-20260101000000-000004",
                task_root="/tmp/task-4",
                creation_block=1,
                cursor_elapsed=40.0,
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
            )

            self.assertEqual(pool.add(first, keep=3), 0)
            self.assertEqual(pool.add(bad_middle, keep=3), 0)
            self.assertEqual(pool.add(third, keep=3), 0)
            self.assertEqual(
                pool.add(
                    replacement,
                    keep=3,
                    prune_first={"validate-20260101000000-000002"},
                ),
                1,
            )

            self.assertEqual(pool.size(), 3)
            self.assertEqual(
                {task.task_name for task in pool.list_tasks()},
                {
                    "validate-20260101000000-000001",
                    "validate-20260101000000-000003",
                    "validate-20260101000000-000004",
                },
            )

    def test_add_with_keep_preserves_active_task_names(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td))
            preserved = PoolTask(
                task_name="validate-20260101000000-000001",
                task_root="/tmp/task-1",
                creation_block=1,
                cursor_elapsed=10.0,
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
            )
            old = PoolTask(
                task_name="validate-20260101000000-000002",
                task_root="/tmp/task-2",
                creation_block=1,
                cursor_elapsed=20.0,
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
            )
            replacement = PoolTask(
                task_name="validate-20260101000000-000003",
                task_root="/tmp/task-3",
                creation_block=1,
                cursor_elapsed=30.0,
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
            )

            self.assertEqual(pool.add(preserved, keep=2), 0)
            self.assertEqual(pool.add(old, keep=2), 0)
            self.assertEqual(
                pool.add(
                    replacement,
                    keep=2,
                    prune_first={"validate-20260101000000-000001"},
                    preserve={"validate-20260101000000-000001"},
                ),
                1,
            )

            self.assertEqual(
                {task.task_name for task in pool.list_tasks()},
                {
                    "validate-20260101000000-000001",
                    "validate-20260101000000-000003",
                },
            )

    def test_add_with_keep_zero_does_not_leave_task_in_pool(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td))
            task = PoolTask(
                task_name="validate-20260101000000-000001",
                task_root="/tmp/task-1",
                creation_block=1,
                cursor_elapsed=10.0,
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
            )

            self.assertEqual(pool.add(task, keep=0), 0)
            self.assertEqual(pool.size(), 1)
            self.assertEqual([item.task_name for item in pool.list_tasks()], [task.task_name])

    def test_normalize_pool_size_prunes_existing_overflow(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td))
            for idx in range(1, 4):
                pool.add(
                    PoolTask(
                        task_name=f"validate-20260101000000-00000{idx}",
                        task_root=f"/tmp/task-{idx}",
                        creation_block=1,
                        cursor_elapsed=float(idx),
                        king_lines=1,
                        king_similarity=0.1,
                        baseline_lines=1,
                    )
                )

            removed = validate._normalize_pool_size(pool=pool, keep=2, pool_label="primary")

            self.assertEqual(removed, 1)
            self.assertEqual(pool.size(), 2)

    def test_normalize_pool_size_with_zero_target_keeps_existing_pool(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td))
            for idx in range(1, 4):
                pool.add(
                    PoolTask(
                        task_name=f"validate-20260101000000-00000{idx}",
                        task_root=f"/tmp/task-{idx}",
                        creation_block=1,
                        cursor_elapsed=float(idx),
                        king_lines=1,
                        king_similarity=0.1,
                        baseline_lines=1,
                    )
                )

            removed = validate._normalize_pool_size(pool=pool, keep=0, pool_label="primary")

            self.assertEqual(removed, 0)
            self.assertEqual(pool.size(), 3)

    def test_static_pool_flushes_tasks_from_prior_king(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td))
            pool.add(
                PoolTask(
                    task_name="validate-20260101000000-000001",
                    task_root="/tmp/task-1",
                    creation_block=1,
                    cursor_elapsed=10.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                    king_hotkey="old-hotkey",
                    king_commit_sha="a" * 40,
                )
            )
            pool.add(
                PoolTask(
                    task_name="validate-20260101000000-000002",
                    task_root="/tmp/task-2",
                    creation_block=1,
                    cursor_elapsed=20.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                    king_hotkey="old-hotkey",
                    king_commit_sha="a" * 40,
                )
            )
            king = validate.ValidatorSubmission(
                hotkey="new-hotkey",
                uid=1,
                repo_full_name="unarbos/ninja",
                repo_url="https://github.com/unarbos/ninja",
                commit_sha="b" * 40,
                commitment="unarbos/ninja@" + "b" * 40,
                commitment_block=1,
                source="chain",
            )

            removed = validate._flush_static_pool_if_stale_for_king(
                config=RunConfig(workspace_root=Path(td), validate_task_pool_static=True),
                pool=pool,
                king=king,
                pool_label="primary",
                pool_starved=threading.Event(),
            )

            self.assertEqual(removed, 2)
            self.assertEqual(pool.size(), 0)

    def test_static_pool_ready_requires_exact_target_for_current_king(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(
                workspace_root=root,
                validate_task_pool_static=True,
                validate_task_pool_target=2,
            )
            pool = TaskPool(root / "pool")
            king = validate.ValidatorSubmission(
                hotkey="new-hotkey",
                uid=1,
                repo_full_name="unarbos/ninja",
                repo_url="https://github.com/unarbos/ninja",
                commit_sha="b" * 40,
                commitment="unarbos/ninja@" + "b" * 40,
                commitment_block=1,
                source="chain",
            )

            ready, reason = validate._static_pool_ready_for_king(
                config=config,
                pool=pool,
                king=king,
                pool_label="primary",
            )
            self.assertFalse(ready)
            self.assertIn("0/2", reason)

            pool.add(
                PoolTask(
                    task_name="validate-20260101000000-000001",
                    task_root="/tmp/task-1",
                    creation_block=1,
                    cursor_elapsed=10.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                    king_hotkey=king.hotkey,
                    king_commit_sha=king.commit_sha,
                )
            )
            self._write_healthy_king_cache(
                config=config,
                task_name="validate-20260101000000-000001",
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
            )
            ready, reason = validate._static_pool_ready_for_king(
                config=config,
                pool=pool,
                king=king,
                pool_label="primary",
            )
            self.assertFalse(ready)
            self.assertIn("1/2", reason)

            pool.add(
                PoolTask(
                    task_name="validate-20260101000000-000002",
                    task_root="/tmp/task-2",
                    creation_block=1,
                    cursor_elapsed=20.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                    king_hotkey=king.hotkey,
                    king_commit_sha=king.commit_sha,
                )
            )
            self._write_healthy_king_cache(
                config=config,
                task_name="validate-20260101000000-000002",
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
            )
            ready, reason = validate._static_pool_ready_for_king(
                config=config,
                pool=pool,
                king=king,
                pool_label="primary",
            )
            self.assertTrue(ready)
            self.assertEqual(reason, "")

    def test_static_pool_ready_for_king_ignores_pool_size_when_target_zero(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(
                workspace_root=root,
                validate_task_pool_static=True,
                validate_task_pool_target=0,
            )
            pool = TaskPool(root / "pool")
            king = validate.ValidatorSubmission(
                hotkey="new-hotkey",
                uid=1,
                repo_full_name="unarbos/ninja",
                repo_url="https://github.com/unarbos/ninja",
                commit_sha="b" * 40,
                commitment="unarbos/ninja@" + "b" * 40,
                commitment_block=1,
                source="chain",
            )
            pool.add(
                PoolTask(
                    task_name="validate-20260101000000-000001",
                    task_root="/tmp/task-1",
                    creation_block=1,
                    cursor_elapsed=10.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                    king_hotkey="old-hotkey",
                    king_commit_sha="a" * 40,
                )
            )

            ready, reason = validate._static_pool_ready_for_king(
                config=config,
                pool=pool,
                king=king,
                pool_label="primary",
            )
            self.assertTrue(ready)
            self.assertEqual(reason, "")

    def test_pool_needs_fill_uses_valid_current_king_count_not_raw_size(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(
                workspace_root=root,
                validate_task_pool_static=True,
                validate_task_pool_target=1,
            )
            pool = TaskPool(root / "pool")
            king = validate.ValidatorSubmission(
                hotkey="current-hotkey",
                uid=7,
                repo_full_name="unarbos/ninja",
                repo_url="https://github.com/unarbos/ninja.git",
                commit_sha="a" * 40,
                commitment="unarbos/ninja@" + "a" * 40,
                commitment_block=1,
            )
            pool.add(
                PoolTask(
                    task_name="validate-20260101000000-000001",
                    task_root=str(config.tasks_root / "validate-20260101000000-000001"),
                    creation_block=1,
                    cursor_elapsed=1.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                    king_hotkey="old-hotkey",
                    king_commit_sha="b" * 40,
                )
            )

            needs_fill, reason = validate._pool_needs_fill_for_king(
                config=config,
                pool=pool,
                king=king,
                pool_label="primary",
            )
            self.assertTrue(needs_fill)
            self.assertEqual(reason, "primary pool has 0/1 valid tasks")

            healthy_task_name = "validate-20260101000000-000002"
            pool.add(
                PoolTask(
                    task_name=healthy_task_name,
                    task_root=str(config.tasks_root / healthy_task_name),
                    creation_block=1,
                    cursor_elapsed=1.0,
                    king_lines=3,
                    king_similarity=0.25,
                    baseline_lines=9,
                    king_hotkey=king.hotkey,
                    king_commit_sha=king.commit_sha,
                )
            )
            self._write_healthy_king_cache(
                config=config,
                task_name=healthy_task_name,
                king_lines=3,
                king_similarity=0.25,
                baseline_lines=9,
            )

            needs_fill, reason = validate._pool_needs_fill_for_king(
                config=config,
                pool=pool,
                king=king,
                pool_label="primary",
            )
            self.assertFalse(needs_fill)
            self.assertEqual(reason, "primary pool has 1/1 valid tasks")

    def test_both_static_pools_ready_rejects_stale_or_incomplete_pool(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(
                workspace_root=root,
                validate_task_pool_static=True,
                validate_task_pool_target=1,
            )
            primary = TaskPool(root / "primary")
            retest = TaskPool(root / "retest")
            king = validate.ValidatorSubmission(
                hotkey="new-hotkey",
                uid=1,
                repo_full_name="unarbos/ninja",
                repo_url="https://github.com/unarbos/ninja",
                commit_sha="b" * 40,
                commitment="unarbos/ninja@" + "b" * 40,
                commitment_block=1,
                source="chain",
            )

            primary.add(
                PoolTask(
                    task_name="validate-20260101000000-000001",
                    task_root="/tmp/task-1",
                    creation_block=1,
                    cursor_elapsed=10.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                    king_hotkey=king.hotkey,
                    king_commit_sha=king.commit_sha,
                )
            )
            self._write_healthy_king_cache(
                config=config,
                task_name="validate-20260101000000-000001",
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
            )
            retest.add(
                PoolTask(
                    task_name="validate-20260101000000-000002",
                    task_root="/tmp/task-2",
                    creation_block=1,
                    cursor_elapsed=20.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                    king_hotkey="old-hotkey",
                    king_commit_sha="a" * 40,
                )
            )
            self._write_healthy_king_cache(
                config=config,
                task_name="validate-20260101000000-000002",
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
            )

            ready, reasons = validate._both_static_pools_ready_for_king(
                config=config,
                king=king,
                pool=primary,
                retest_pool=retest,
            )
            self.assertFalse(ready)
            self.assertGreaterEqual(len(reasons), 1)
            self.assertTrue(any("stale" in reason for reason in reasons))

    def test_pool_task_health_requires_current_king_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(
                workspace_root=root,
                validate_task_pool_static=True,
                validate_task_pool_target=1,
            )
            king = validate.ValidatorSubmission(
                hotkey="new-hotkey",
                uid=1,
                repo_full_name="unarbos/ninja",
                repo_url="https://github.com/unarbos/ninja",
                commit_sha="b" * 40,
                commitment="unarbos/ninja@" + "b" * 40,
                commitment_block=1,
                source="chain",
            )
            task_name = "validate-20260101000000-000001"
            task_root = config.tasks_root / task_name
            self._write_healthy_king_cache(
                config=config,
                task_name=task_name,
                king_lines=12,
                king_similarity=0.25,
                baseline_lines=48,
                king_agent_timeout_seconds=300,
            )
            compare_dir = task_root / "comparisons" / "king--vs--baseline"

            task = PoolTask(
                task_name=task_name,
                task_root=str(task_root),
                creation_block=1,
                cursor_elapsed=10.0,
                king_lines=12,
                king_similarity=0.25,
                baseline_lines=48,
                king_hotkey=king.hotkey,
                king_commit_sha=king.commit_sha,
            )

            healthy, reason = validate._pool_task_has_healthy_king_cache(
                config=config,
                task=task,
            )
            self.assertTrue(healthy)
            self.assertEqual(reason, "")

            (compare_dir / "compare.json").unlink()
            healthy, reason = validate._pool_task_has_healthy_king_cache(
                config=config,
                task=task,
            )
            self.assertFalse(healthy)
            self.assertIn("missing", reason)

    def test_pool_task_health_requires_king_timeout_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(
                workspace_root=root,
                validate_task_pool_static=True,
                validate_task_pool_target=1,
            )
            task_name = "validate-20260101000000-000001"
            task_root = config.tasks_root / task_name
            self._write_healthy_king_cache(
                config=config,
                task_name=task_name,
                king_lines=12,
                king_similarity=0.25,
                baseline_lines=48,
            )
            king_solve_json = task_root / "solutions" / "king" / "solve.json"
            king_solve_json.write_text("{}\n")

            task = PoolTask(
                task_name=task_name,
                task_root=str(task_root),
                creation_block=1,
                cursor_elapsed=10.0,
                king_lines=12,
                king_similarity=0.25,
                baseline_lines=48,
                king_hotkey="current-hotkey",
                king_commit_sha="b" * 40,
            )

            healthy, reason = validate._pool_task_has_healthy_king_cache(
                config=config,
                task=task,
            )
            self.assertFalse(healthy)
            self.assertEqual(reason, "king solve timeout metadata is missing")

    def test_pool_task_health_rejects_mismatched_king_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(
                workspace_root=root,
                validate_task_pool_static=True,
                validate_task_pool_target=1,
            )
            task_name = "validate-20260101000000-000001"
            task_root = config.tasks_root / task_name
            self._write_healthy_king_cache(
                config=config,
                task_name=task_name,
                king_lines=12,
                king_similarity=0.25,
                baseline_lines=48,
                king_agent_timeout_seconds=300,
            )

            task = PoolTask(
                task_name=task_name,
                task_root=str(task_root),
                creation_block=1,
                cursor_elapsed=10.0,
                king_lines=12,
                king_similarity=0.25,
                baseline_lines=48,
                agent_timeout_seconds=321,
                king_hotkey="current-hotkey",
                king_commit_sha="b" * 40,
            )

            healthy, reason = validate._pool_task_has_healthy_king_cache(
                config=config,
                task=task,
            )
            self.assertFalse(healthy)
            self.assertEqual(reason, "king solve timeout mismatch (300 != 321)")

    def test_pool_task_health_rejects_empty_king_patch(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(
                workspace_root=root,
                validate_task_pool_static=True,
                validate_task_pool_target=1,
            )
            task_name = "validate-20260101000000-000001"
            task_root = config.tasks_root / task_name
            self._write_healthy_king_cache(
                config=config,
                task_name=task_name,
                king_lines=0,
                king_similarity=0.0,
                baseline_lines=12,
                king_agent_timeout_seconds=300,
            )
            task = PoolTask(
                task_name=task_name,
                task_root=str(task_root),
                creation_block=1,
                cursor_elapsed=10.0,
                king_lines=0,
                king_similarity=0.0,
                baseline_lines=12,
                king_hotkey="current-hotkey",
                king_commit_sha="b" * 40,
            )

            healthy, reason = validate._pool_task_has_healthy_king_cache(
                config=config,
                task=task,
            )
            self.assertFalse(healthy)
            self.assertEqual(reason, "king produced no matched changed lines")

    def test_pool_refresh_fallback_records_attempted_king_timeout(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(
                workspace_root=root,
                validate_task_pool_static=True,
                validate_task_pool_target=1,
            )
            task_name = "validate-20260101000000-000001"
            task_root = config.tasks_root / task_name
            self._write_minimal_task_metadata(task_root)
            (task_root / "task" / "original").mkdir(parents=True)
            task = PoolTask(
                task_name=task_name,
                task_root=str(task_root),
                creation_block=1,
                cursor_elapsed=10.0,
                king_lines=12,
                king_similarity=0.25,
                baseline_lines=48,
                agent_timeout_seconds=222,
                king_hotkey="current-hotkey",
                king_commit_sha="b" * 40,
            )
            king = validate.ValidatorSubmission(
                hotkey="current-hotkey",
                uid=1,
                repo_full_name="unarbos/ninja",
                repo_url="https://github.com/unarbos/ninja",
                commit_sha="b" * 40,
                commitment="unarbos/ninja@" + "b" * 40,
                commitment_block=1,
                source="chain",
            )

            with (
                patch("validate._build_agent_config", return_value=config),
                patch("validate.solve_task_run", side_effect=RuntimeError("boom")),
                patch(
                    "validate.compare_task_run",
                    return_value=SimpleNamespace(
                        matched_changed_lines=12,
                        similarity_ratio=0.25,
                        total_changed_lines_b=48,
                    ),
                ),
            ):
                refreshed = validate._refresh_pool_task_for_king(
                    config=config,
                    king=king,
                    task=task,
                    pool_label="primary",
                )

            payload = json.loads((task_root / "solutions" / "king" / "solve.json").read_text())
            self.assertNotEqual(config.agent_timeout, 222)
            self.assertEqual(payload["agent_timeout_seconds"], 222)
            assert refreshed is not None
            self.assertEqual(refreshed.agent_timeout_seconds, 222)

    def test_static_pool_ready_rejects_inconsistent_king_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(
                workspace_root=root,
                validate_task_pool_static=True,
                validate_task_pool_target=1,
            )
            pool = TaskPool(root / "pool")
            king = validate.ValidatorSubmission(
                hotkey="new-hotkey",
                uid=1,
                repo_full_name="unarbos/ninja",
                repo_url="https://github.com/unarbos/ninja",
                commit_sha="b" * 40,
                commitment="unarbos/ninja@" + "b" * 40,
                commitment_block=1,
                source="chain",
            )
            task_name = "validate-20260101000000-000001"
            task_root = config.tasks_root / task_name
            self._write_healthy_king_cache(
                config=config,
                task_name=task_name,
                king_lines=99,
                king_similarity=0.25,
                baseline_lines=48,
            )
            compare_dir = task_root / "comparisons" / "king--vs--baseline"
            (compare_dir / "compare.json").write_text(
                json.dumps(
                    {
                        "result": {
                            "matched_changed_lines": 99,
                            "similarity_ratio": 0.25,
                            "total_changed_lines_b": 48,
                        }
                    }
                )
            )
            pool.add(
                PoolTask(
                    task_name=task_name,
                    task_root=str(task_root),
                    creation_block=1,
                    cursor_elapsed=10.0,
                    king_lines=12,
                    king_similarity=0.25,
                    baseline_lines=48,
                    king_hotkey=king.hotkey,
                    king_commit_sha=king.commit_sha,
                )
            )

            ready, reason = validate._static_pool_ready_for_king(
                config=config,
                pool=pool,
                king=king,
                pool_label="primary",
            )
            self.assertFalse(ready)
            self.assertIn("unhealthy king cache", reason)

    def test_prune_king_cache_keeps_only_healthy_pooled_current_king_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(
                workspace_root=root,
                validate_task_pool_static=True,
                validate_task_pool_target=2,
            )
            primary = TaskPool(root / "primary")
            retest = TaskPool(root / "retest")
            king = validate.ValidatorSubmission(
                hotkey="new-hotkey",
                uid=1,
                repo_full_name="unarbos/ninja",
                repo_url="https://github.com/unarbos/ninja",
                commit_sha="b" * 40,
                commitment="unarbos/ninja@" + "b" * 40,
                commitment_block=1,
                source="chain",
            )

            healthy_name = "validate-20260101000000-000001"
            unhealthy_name = "validate-20260101000000-000002"
            stray_name = "validate-20260101000000-000003"
            self._write_healthy_king_cache(
                config=config,
                task_name=healthy_name,
                king_lines=12,
                king_similarity=0.25,
                baseline_lines=48,
            )
            self._write_healthy_king_cache(
                config=config,
                task_name=unhealthy_name,
                king_lines=99,
                king_similarity=0.25,
                baseline_lines=48,
            )
            self._write_healthy_king_cache(
                config=config,
                task_name=stray_name,
                king_lines=5,
                king_similarity=0.1,
                baseline_lines=10,
            )
            primary.add(
                PoolTask(
                    task_name=healthy_name,
                    task_root=str(config.tasks_root / healthy_name),
                    creation_block=1,
                    cursor_elapsed=10.0,
                    king_lines=12,
                    king_similarity=0.25,
                    baseline_lines=48,
                    king_hotkey=king.hotkey,
                    king_commit_sha=king.commit_sha,
                )
            )
            primary.add(
                PoolTask(
                    task_name=unhealthy_name,
                    task_root=str(config.tasks_root / unhealthy_name),
                    creation_block=1,
                    cursor_elapsed=20.0,
                    king_lines=12,
                    king_similarity=0.25,
                    baseline_lines=48,
                    king_hotkey=king.hotkey,
                    king_commit_sha=king.commit_sha,
                )
            )

            counts = validate._prune_king_cache_to_current_pools(
                config=config,
                king=king,
                pool=primary,
                retest_pool=retest,
                pool_starved=threading.Event(),
                retest_pool_starved=threading.Event(),
            )

            self.assertEqual(primary.size(), 1)
            self.assertEqual(primary.list_tasks()[0].task_name, healthy_name)
            self.assertEqual(counts["dropped_primary_pool_tasks"], 1)
            self.assertEqual(counts["dropped_retest_pool_tasks"], 0)
            self.assertTrue((config.tasks_root / healthy_name / "solutions" / "king").exists())
            self.assertFalse((config.tasks_root / unhealthy_name / "solutions" / "king").exists())
            self.assertFalse((config.tasks_root / stray_name / "solutions" / "king").exists())

    def test_take_respects_exclude_when_sorting_by_speed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            pool = TaskPool(root / "pool")
            fast_root = root / "fast"
            slow_root = root / "slow"
            fast_root.mkdir()
            slow_root.mkdir()
            for task_root in (fast_root, slow_root):
                baseline_dir = task_root / "solutions" / "baseline"
                baseline_dir.mkdir(parents=True)
                (baseline_dir / "solve.json").write_text("{}\n")
                (baseline_dir / "solution.diff").write_text("diff\n")
            pool.add(
                PoolTask(
                    task_name="fast",
                    task_root=str(fast_root),
                    creation_block=20,
                    cursor_elapsed=20.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                )
            )
            pool.add(
                PoolTask(
                    task_name="slow",
                    task_root=str(slow_root),
                    creation_block=20,
                    cursor_elapsed=300.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                )
            )

            task = pool.take(min_block=10, exclude={"fast"})

        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.task_name, "slow")

    def test_duel_agent_timeout_matches_stored_king_timeout(self):
        task = PoolTask(
            task_name="cached",
            task_root="/tmp/cached",
            creation_block=20,
            cursor_elapsed=10.0,
            king_lines=1,
            king_similarity=0.1,
            baseline_lines=1,
            agent_timeout_seconds=321,
        )

        loaded = PoolTask.from_dict(task.to_dict())

        self.assertEqual(validate._duel_agent_timeout(loaded), 321)

    def test_cached_pool_task_timeout_is_upgraded_to_current_policy(self):
        loaded = PoolTask.from_dict(
            {
                "task_name": "old-policy",
                "task_root": "/tmp/old-policy",
                "creation_block": 20,
                "cursor_elapsed": 81.0,
                "king_lines": 1,
                "king_similarity": 0.1,
                "baseline_lines": 1,
                "agent_timeout_seconds": 163,
            }
        )

        self.assertEqual(loaded.agent_timeout_seconds, 244)
        self.assertEqual(validate._duel_agent_timeout(loaded), 244)

    def test_duel_agent_timeout_does_not_floor_stored_king_timeout(self):
        task = PoolTask(
            task_name="fast",
            task_root="/tmp/fast",
            creation_block=20,
            cursor_elapsed=10.0,
            king_lines=1,
            king_similarity=0.1,
            baseline_lines=1,
            agent_timeout_seconds=120,
        )

        self.assertEqual(validate._duel_agent_timeout(task), 120)

    def test_duel_task_submission_order_preserves_gathered_order(self):
        tasks = [
            PoolTask(
                task_name=f"task-{idx:02d}",
                task_root=f"/tmp/task-{idx:02d}",
                creation_block=20,
                cursor_elapsed=float(idx),
                king_lines=1,
                king_similarity=0.1,
                baseline_lines=1,
                agent_timeout_seconds=idx * 100,
            )
            for idx in range(1, 11)
        ]

        ordered = validate._order_duel_tasks_for_submission(tasks)

        self.assertEqual({task.task_name for task in ordered}, {task.task_name for task in tasks})
        self.assertEqual(
            [task.task_name for task in ordered],
            [task.task_name for task in tasks],
        )

    def test_legacy_pool_task_backfills_agent_timeout(self):
        loaded = PoolTask.from_dict(
            {
                "task_name": "legacy",
                "task_root": "/tmp/legacy",
                "creation_block": 20,
                "cursor_elapsed": 50.0,
                "king_lines": 1,
                "king_similarity": 0.1,
                "baseline_lines": 1,
            }
        )

        self.assertEqual(loaded.agent_timeout_seconds, 151)
        self.assertEqual(validate._duel_agent_timeout(loaded), 151)

    def test_pool_generation_backs_off_on_github_rate_limit(self):
        self.assertTrue(
            validate._is_github_rate_limit_error(
                RuntimeError("gh: API rate limit exceeded for user ID 123 (HTTP 403)")
            )
        )
        self.assertTrue(
            validate._is_github_rate_limit_error(
                RuntimeError("GitHub PR fetch failed for unarbos/ninja#360: HTTP 403")
            )
        )
        self.assertTrue(
            validate._is_github_rate_limit_error(
                RuntimeError(
                    "Client error '429 too many requests' for url "
                    "'https://api.github.com/events?page=1&per_page=30'"
                )
            )
        )
        self.assertFalse(validate._is_github_rate_limit_error(RuntimeError("docker failed")))

        validate._note_pool_generation_rate_limit("primary")

        self.assertGreater(validate._pool_generation_backoff_remaining(), 0.0)

    def test_missing_runtime_secrets_require_openrouter_key(self):
        self.assertEqual(
            validate._missing_runtime_secrets(RunConfig(openrouter_api_key=None)),
            ["OPENROUTER_API_KEY"],
        )
        self.assertEqual(
            validate._missing_runtime_secrets(RunConfig(openrouter_api_key="set")),
            [],
        )

    def test_zero_scored_duel_reason_includes_sample_errors(self):
        reason = validate._zero_scored_duel_reason(
            4101,
            [
                validate.ValidationRoundResult(
                    task_name="task-a",
                    winner="error",
                    king_lines=0,
                    challenger_lines=0,
                    king_similarity_ratio=0.0,
                    challenger_similarity_ratio=0.0,
                    king_challenger_similarity=0.0,
                    task_root="/tmp/task-a",
                    king_compare_root="",
                    challenger_compare_root="",
                    error="OPENROUTER_API_KEY is not set",
                )
            ],
        )

        self.assertIn("zero scored rounds", reason)
        self.assertIn("OPENROUTER_API_KEY is not set", reason)

    def test_partial_parallel_duel_task_set_is_retryable(self):
        with self.assertRaises(validate.RetryableDuelError) as ctx:
            validate._raise_if_insufficient_duel_tasks(4189, 50, [object()] * 5)

        self.assertIn("gathered only 5/50 tasks", str(ctx.exception))
        validate._raise_if_insufficient_duel_tasks(4190, 50, [object()] * 50)

    def test_parallel_duel_stops_unstarted_rounds_when_king_mathematically_safe(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td) / "pool")
            for idx in range(8):
                task_root = Path(td) / f"task-{idx:02d}"
                self._write_minimal_task_metadata(task_root)
                baseline_dir = task_root / "solutions" / "baseline"
                baseline_dir.mkdir(parents=True, exist_ok=True)
                (baseline_dir / "solve.json").write_text("{}\n")
                (baseline_dir / "solution.diff").write_text("diff\n")
                pool.add(
                    PoolTask(
                        task_name=f"task-{idx:02d}",
                        task_root=str(task_root),
                        creation_block=1,
                        cursor_elapsed=float(idx + 1),
                        king_lines=1,
                        king_similarity=0.1,
                        baseline_lines=1,
                    )
                )
            config = RunConfig(
                workspace_root=Path(td),
                validate_duel_rounds=8,
                validate_round_concurrency=1,
                validate_win_margin=3,
            )
            king = validate.ValidatorSubmission(
                hotkey="king-hotkey",
                uid=1,
                repo_full_name="king/ninja",
                repo_url="https://github.com/king/ninja",
                commit_sha="a" * 40,
                commitment="unarbos/ninja@" + "a" * 40,
                commitment_block=1,
                source="chain",
            )
            challenger = validate.ValidatorSubmission(
                hotkey="challenger-hotkey",
                uid=2,
                repo_full_name="challenger/ninja",
                repo_url="https://github.com/challenger/ninja",
                commit_sha="b" * 40,
                commitment="unarbos/ninja@" + "b" * 40,
                commitment_block=1,
                source="chain",
            )

            def king_round(*, task, king, challenger, config, duel_id, pool=None):
                return validate.ValidationRoundResult(
                    task_name=task.task_name,
                    winner="king",
                    king_lines=1,
                    challenger_lines=1,
                    king_similarity_ratio=1.0,
                    challenger_similarity_ratio=0.0,
                    king_challenger_similarity=0.0,
                    task_root=task.task_root,
                    king_compare_root="",
                    challenger_compare_root="",
                )

            with patch("validate._solve_and_compare_round", side_effect=king_round) as solve_round:
                result = validate._run_parallel_duel(
                    config=config,
                    state=validate.ValidatorState(current_king=king),
                    king=king,
                    challenger=challenger,
                    duel_id=99,
                    pool=pool,
                )

        self.assertFalse(result.king_replaced)
        self.assertEqual(result.losses, 3)
        self.assertEqual(len(result.rounds), 3)
        self.assertEqual(solve_round.call_count, 3)

    def test_parallel_duel_cancels_in_flight_rounds_when_king_mathematically_safe(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td) / "pool")
            for idx in range(8):
                task_root = Path(td) / f"task-{idx:02d}"
                self._write_minimal_task_metadata(task_root)
                baseline_dir = task_root / "solutions" / "baseline"
                baseline_dir.mkdir(parents=True, exist_ok=True)
                (baseline_dir / "solve.json").write_text("{}\n")
                (baseline_dir / "solution.diff").write_text("diff\n")
                pool.add(
                    PoolTask(
                        task_name=f"task-{idx:02d}",
                        task_root=str(task_root),
                        creation_block=1,
                        cursor_elapsed=float(idx + 1),
                        king_lines=1,
                        king_similarity=0.1,
                        baseline_lines=1,
                    )
                )
            config = RunConfig(
                workspace_root=Path(td),
                validate_duel_rounds=8,
                validate_round_concurrency=4,
                validate_win_margin=3,
            )
            king = validate.ValidatorSubmission(
                hotkey="king-hotkey",
                uid=1,
                repo_full_name="king/ninja",
                repo_url="https://github.com/king/ninja",
                commit_sha="a" * 40,
                commitment="unarbos/ninja@" + "a" * 40,
                commitment_block=1,
                source="chain",
            )
            challenger = validate.ValidatorSubmission(
                hotkey="challenger-hotkey",
                uid=2,
                repo_full_name="challenger/ninja",
                repo_url="https://github.com/challenger/ninja",
                commit_sha="b" * 40,
                commitment="unarbos/ninja@" + "b" * 40,
                commitment_block=1,
                source="chain",
            )

            def king_round(*, task, king, challenger, config, duel_id, pool=None):
                if task.task_name == "task-03":
                    time.sleep(1.0)
                return validate.ValidationRoundResult(
                    task_name=task.task_name,
                    winner="king",
                    king_lines=1,
                    challenger_lines=1,
                    king_similarity_ratio=1.0,
                    challenger_similarity_ratio=0.0,
                    king_challenger_similarity=0.0,
                    task_root=task.task_root,
                    king_compare_root="",
                    challenger_compare_root="",
                )

            with (
                patch("validate._solve_and_compare_round", side_effect=king_round) as solve_round,
                patch("validate._kill_stale_containers") as kill_containers,
            ):
                result = validate._run_parallel_duel(
                    config=config,
                    state=validate.ValidatorState(current_king=king),
                    king=king,
                    challenger=challenger,
                    duel_id=99,
                    pool=pool,
                )

        self.assertFalse(result.king_replaced)
        self.assertEqual(result.losses, 3)
        self.assertEqual(len(result.rounds), 3)
        self.assertGreaterEqual(solve_round.call_count, 3)
        kill_containers.assert_called_once()

    def test_parallel_duel_preserves_done_round_at_hard_deadline_without_submitting_more(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td) / "pool")
            for idx in range(3):
                task_root = Path(td) / f"task-{idx:02d}"
                self._write_minimal_task_metadata(task_root)
                baseline_dir = task_root / "solutions" / "baseline"
                baseline_dir.mkdir(parents=True, exist_ok=True)
                (baseline_dir / "solve.json").write_text("{}\n")
                (baseline_dir / "solution.diff").write_text("diff\n")
                pool.add(
                    PoolTask(
                        task_name=f"task-{idx:02d}",
                        task_root=str(task_root),
                        creation_block=1,
                        cursor_elapsed=float(idx + 1),
                        king_lines=1,
                        king_similarity=0.1,
                        baseline_lines=1,
                    )
                )
            config = RunConfig(
                workspace_root=Path(td),
                validate_duel_rounds=3,
                validate_round_concurrency=1,
                validate_win_margin=0,
            )
            king = validate.ValidatorSubmission(
                hotkey="king-hotkey",
                uid=1,
                repo_full_name="king/ninja",
                repo_url="https://github.com/king/ninja",
                commit_sha="a" * 40,
                commitment="unarbos/ninja@" + "a" * 40,
                commitment_block=1,
                source="chain",
            )
            challenger = validate.ValidatorSubmission(
                hotkey="challenger-hotkey",
                uid=2,
                repo_full_name="challenger/ninja",
                repo_url="https://github.com/challenger/ninja",
                commit_sha="b" * 40,
                commitment="unarbos/ninja@" + "b" * 40,
                commitment_block=1,
                source="chain",
            )

            def challenger_round(*, task, king, challenger, config, duel_id, pool=None):
                return validate.ValidationRoundResult(
                    task_name=task.task_name,
                    winner="challenger",
                    king_lines=1,
                    challenger_lines=1,
                    king_similarity_ratio=0.0,
                    challenger_similarity_ratio=1.0,
                    king_challenger_similarity=0.0,
                    task_root=task.task_root,
                    king_compare_root="",
                    challenger_compare_root="",
                )

            def wait_for_submitted_round(pending, timeout=None, return_when=None):
                done, _ = futures_wait(pending)
                return done, set()

            with (
                patch("validate._solve_and_compare_round", side_effect=challenger_round) as solve_round,
                patch("validate._futures_wait", side_effect=wait_for_submitted_round),
                patch("validate._PARALLEL_DUEL_HARD_TIMEOUT", 0.0),
                patch("validate._kill_stale_containers") as kill_containers,
            ):
                result = validate._run_parallel_duel(
                    config=config,
                    state=validate.ValidatorState(current_king=king),
                    king=king,
                    challenger=challenger,
                    duel_id=99,
                    pool=pool,
                )

        self.assertEqual(solve_round.call_count, 1)
        self.assertEqual(result.wins, 1)
        self.assertEqual(len(result.rounds), 3)
        self.assertEqual(result.rounds[0].winner, "challenger")
        self.assertEqual([round_result.winner for round_result in result.rounds[1:]], ["error", "error"])
        self.assertTrue(all("not started (hard duel deadline)" in r.error for r in result.rounds[1:]))
        kill_containers.assert_called_once()

    def test_parallel_duel_does_not_timeout_when_last_round_finishes_at_deadline(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td) / "pool")
            task_root = Path(td) / "task-00"
            self._write_minimal_task_metadata(task_root)
            baseline_dir = task_root / "solutions" / "baseline"
            baseline_dir.mkdir(parents=True, exist_ok=True)
            (baseline_dir / "solve.json").write_text("{}\n")
            (baseline_dir / "solution.diff").write_text("diff\n")
            pool.add(
                PoolTask(
                    task_name="task-00",
                    task_root=str(task_root),
                    creation_block=1,
                    cursor_elapsed=1.0,
                    king_lines=1,
                    king_similarity=0.1,
                    baseline_lines=1,
                )
            )
            config = RunConfig(
                workspace_root=Path(td),
                validate_duel_rounds=1,
                validate_round_concurrency=1,
                validate_win_margin=0,
            )
            king = validate.ValidatorSubmission(
                hotkey="king-hotkey",
                uid=1,
                repo_full_name="king/ninja",
                repo_url="https://github.com/king/ninja",
                commit_sha="a" * 40,
                commitment="unarbos/ninja@" + "a" * 40,
                commitment_block=1,
                source="chain",
            )
            challenger = validate.ValidatorSubmission(
                hotkey="challenger-hotkey",
                uid=2,
                repo_full_name="challenger/ninja",
                repo_url="https://github.com/challenger/ninja",
                commit_sha="b" * 40,
                commitment="unarbos/ninja@" + "b" * 40,
                commitment_block=1,
                source="chain",
            )

            def challenger_round(*, task, king, challenger, config, duel_id, pool=None):
                return validate.ValidationRoundResult(
                    task_name=task.task_name,
                    winner="challenger",
                    king_lines=1,
                    challenger_lines=1,
                    king_similarity_ratio=0.0,
                    challenger_similarity_ratio=1.0,
                    king_challenger_similarity=0.0,
                    task_root=task.task_root,
                    king_compare_root="",
                    challenger_compare_root="",
                )

            def wait_for_submitted_round(pending, timeout=None, return_when=None):
                done, _ = futures_wait(pending)
                return done, set()

            with (
                patch("validate._solve_and_compare_round", side_effect=challenger_round) as solve_round,
                patch("validate._futures_wait", side_effect=wait_for_submitted_round),
                patch("validate._PARALLEL_DUEL_HARD_TIMEOUT", 0.0),
                patch("validate._kill_stale_containers") as kill_containers,
            ):
                result = validate._run_parallel_duel(
                    config=config,
                    state=validate.ValidatorState(current_king=king),
                    king=king,
                    challenger=challenger,
                    duel_id=100,
                    pool=pool,
                )

        self.assertEqual(solve_round.call_count, 1)
        self.assertEqual(result.wins, 1)
        self.assertEqual(len(result.rounds), 1)
        self.assertEqual(result.rounds[0].winner, "challenger")
        kill_containers.assert_not_called()

    def test_refresh_budget_allows_bounded_hourly_batch(self):
        config = RunConfig(
            validate_task_pool_refresh_count=2,
            validate_task_pool_refresh_interval_seconds=3600,
        )
        budget = TaskPoolRefreshBudget()

        claimed, started = budget.claim(config=config)
        self.assertFalse(claimed)
        self.assertFalse(started)

        budget._next_refresh_at = 0

        claimed, started = budget.claim(config=config)
        self.assertTrue(claimed)
        self.assertTrue(started)

        claimed, started = budget.claim(config=config)
        self.assertTrue(claimed)
        self.assertFalse(started)

        claimed, started = budget.claim(config=config)
        self.assertFalse(claimed)
        self.assertFalse(started)

        self.assertFalse(budget.finish(config=config, success=True))
        self.assertTrue(budget.finish(config=config, success=True))

        claimed, started = budget.claim(config=config)
        self.assertFalse(claimed)
        self.assertFalse(started)

    def test_refresh_budget_retries_failed_replacement(self):
        config = RunConfig(
            validate_task_pool_refresh_count=1,
            validate_task_pool_refresh_interval_seconds=3600,
        )
        budget = TaskPoolRefreshBudget()
        budget._next_refresh_at = 0

        claimed, _ = budget.claim(config=config)
        self.assertTrue(claimed)
        self.assertFalse(budget.finish(config=config, success=False))

        claimed, started = budget.claim(config=config)
        self.assertTrue(claimed)
        self.assertFalse(started)

    def test_diff_judge_prompt_does_not_include_timeout_flags(self):
        prompt = validate._build_diff_judge_prompt(
            task_prompt="fix the bug",
            reference_patch="diff --git a/ref b/ref",
            king_patch="diff --git a/king b/king",
            challenger_patch="diff --git a/challenger b/challenger",
        )

        payload = json.loads(prompt[prompt.index("{\n  \"challenger_patch\"") :])
        self.assertNotIn("king_timed_out", payload)
        self.assertNotIn("challenger_timed_out", payload)
        self.assertNotIn("timeout", prompt.lower())


if __name__ == "__main__":
    unittest.main()
