import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import validate
from config import RunConfig
from validate import PoolTask, TaskPool, TaskPoolRefreshBudget, _prepare_validate_paths


class TaskPoolTest(unittest.TestCase):
    def tearDown(self):
        with validate._POOL_GENERATION_BACKOFF_LOCK:
            validate._pool_generation_backoff_until = 0.0

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

    def test_take_returns_fastest_cached_task(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td))
            pool.add(
                PoolTask(
                    task_name="slow",
                    task_root="/tmp/slow",
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
                    task_root="/tmp/fast",
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
                )
            )

            task = pool.take(min_block=100)

        self.assertIsNotNone(task)
        assert task is not None
        self.assertEqual(task.task_name, "cached")

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

    def test_take_respects_exclude_when_sorting_by_speed(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td))
            pool.add(
                PoolTask(
                    task_name="fast",
                    task_root="/tmp/fast",
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
                    task_root="/tmp/slow",
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

    def test_duel_task_submission_order_spreads_timeout_budgets(self):
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
            [task.task_name for task in ordered[:5]],
            ["task-01", "task-03", "task-05", "task-07", "task-09"],
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

        self.assertEqual(loaded.agent_timeout_seconds, 300)
        self.assertEqual(validate._duel_agent_timeout(loaded), 300)

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

    def test_pool_refresh_solves_king_without_baseline_compare(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = RunConfig(workspace_root=root)
            task_name = "validate-20260101000000-000001"
            task_root = config.tasks_root / task_name
            (task_root / "task").mkdir(parents=True)
            (task_root / "task" / "task.json").write_text("{}\n")
            (task_root / "task" / "commit.json").write_text("{}\n")
            task = PoolTask(
                task_name=task_name,
                task_root=str(task_root),
                creation_block=20,
                cursor_elapsed=0.0,
                king_lines=0,
                king_similarity=0.0,
                baseline_lines=0,
                agent_timeout_seconds=300,
            )
            king = validate.ValidatorSubmission(
                hotkey="king-hotkey",
                uid=1,
                repo_full_name="king/ninja",
                repo_url="https://github.com/king/ninja",
                commit_sha="a" * 40,
                commitment="github-pr:unarbos/ninja#1@" + "a" * 40,
                commitment_block=1,
                source="github_pr",
            )

            with (
                patch("validate.solve_task_run", return_value=SimpleNamespace(exit_reason="completed")) as solve,
                patch("validate._solution_patch_lines", return_value=42),
                patch("validate.compare_task_run", side_effect=AssertionError("baseline compare should not run")),
            ):
                refreshed = validate._refresh_pool_task_for_king(
                    config=config,
                    king=king,
                    task=task,
                    pool_label="primary",
                )

        self.assertIsNotNone(refreshed)
        assert refreshed is not None
        self.assertEqual(refreshed.king_lines, 42)
        self.assertEqual(refreshed.king_similarity, 0.0)
        self.assertEqual(refreshed.baseline_lines, 0)
        solve.assert_called_once()

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

    def test_parallel_duel_stops_when_king_mathematically_safe(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td) / "pool")
            for idx in range(8):
                pool.add(
                    PoolTask(
                        task_name=f"task-{idx:02d}",
                        task_root=f"/tmp/task-{idx:02d}",
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
                commitment="github-pr:unarbos/ninja#1@" + "a" * 40,
                commitment_block=1,
                source="github_pr",
            )
            challenger = validate.ValidatorSubmission(
                hotkey="challenger-hotkey",
                uid=2,
                repo_full_name="challenger/ninja",
                repo_url="https://github.com/challenger/ninja",
                commit_sha="b" * 40,
                commitment="github-pr:unarbos/ninja#2@" + "b" * 40,
                commitment_block=1,
                source="github_pr",
            )

            def king_round(*, task, challenger, config, duel_id):
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

    def test_parallel_duel_stops_in_flight_when_challenger_mathematically_locked(self):
        with tempfile.TemporaryDirectory() as td:
            pool = TaskPool(Path(td) / "pool")
            for idx in range(8):
                pool.add(
                    PoolTask(
                        task_name=f"task-{idx:02d}",
                        task_root=f"/tmp/task-{idx:02d}",
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
                validate_round_concurrency=6,
                validate_win_margin=0,
            )
            king = validate.ValidatorSubmission(
                hotkey="king-hotkey",
                uid=1,
                repo_full_name="king/ninja",
                repo_url="https://github.com/king/ninja",
                commit_sha="a" * 40,
                commitment="github-pr:unarbos/ninja#1@" + "a" * 40,
                commitment_block=1,
                source="github_pr",
            )
            challenger = validate.ValidatorSubmission(
                hotkey="challenger-hotkey",
                uid=2,
                repo_full_name="challenger/ninja",
                repo_url="https://github.com/challenger/ninja",
                commit_sha="b" * 40,
                commitment="github-pr:unarbos/ninja#2@" + "b" * 40,
                commitment_block=1,
                source="github_pr",
            )
            release_blocked_rounds = threading.Event()

            def challenger_round(*, task, challenger, config, duel_id):
                if int(task.task_name.rsplit("-", 1)[1]) >= 5:
                    release_blocked_rounds.wait(timeout=5)
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

            def release_containers():
                release_blocked_rounds.set()

            with (
                patch("validate._solve_and_compare_round", side_effect=challenger_round) as solve_round,
                patch("validate._kill_stale_containers", side_effect=release_containers) as kill_stale,
            ):
                result = validate._run_parallel_duel(
                    config=config,
                    state=validate.ValidatorState(current_king=king),
                    king=king,
                    challenger=challenger,
                    duel_id=100,
                    pool=pool,
                )

        release_blocked_rounds.set()
        self.assertTrue(result.king_replaced)
        self.assertEqual(result.wins, 5)
        self.assertEqual(len(result.rounds), 5)
        self.assertGreaterEqual(solve_round.call_count, 6)
        kill_stale.assert_called_once()

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


if __name__ == "__main__":
    unittest.main()
