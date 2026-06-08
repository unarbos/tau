from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from config import RunConfig
from task_generation import GeneratedTask
from tau.solver.base import SolveRequest, SolveResult
from tau.solver.caching_solver import CacheMissError, CachingSolver
from tau.solver.constants import COMPLETED_EXIT_REASON
from tau.solver.dummy_solver import DummySolver


def _task() -> GeneratedTask:
    return GeneratedTask(
        title="Fix bug",
        description="There is a bug.",
        acceptance_criteria=["tests pass"],
        raw_output="",
        elapsed_seconds=0.0,
    )


def _request(
    task_name: str = "task-a",
    solution_name: str = "sol-1",
    commit_sha: str = "abc123",
) -> SolveRequest:
    return SolveRequest(
        repo_dir=Path("/tmp/repo"),
        task=_task(),
        task_name=task_name,
        solution_name=solution_name,
        commit_sha=commit_sha,
    )


def _config() -> RunConfig:
    return RunConfig()


def _dummy(model: str = "test/model") -> DummySolver:
    return DummySolver(model=model, timeout=60, config=_config())


class CachingSolverReadWriteTest(unittest.TestCase):
    def test_cache_miss_calls_inner_and_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            solver = CachingSolver(model=None, timeout=0, config=_config(), inner=_dummy(), cache_dir=cache_dir)
            req = _request()

            result = solver.solve(req)

            self.assertTrue(result.success)
            self.assertEqual(len(list(cache_dir.glob("*.json"))), 1)

    def test_cache_hit_returns_cached_result_without_calling_inner(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            solver = CachingSolver(model=None, timeout=0, config=_config(), inner=_dummy("first/model"), cache_dir=cache_dir)
            req = _request()
            solver.solve(req)

            solver2 = CachingSolver(model=None, timeout=0, config=_config(), inner=_dummy("second/model"), cache_dir=cache_dir)
            result = solver2.solve(req)

            self.assertEqual(result.model, "first/model")

    def test_cache_key_differs_by_task_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            solver = CachingSolver(model=None, timeout=0, config=_config(), inner=_dummy(), cache_dir=cache_dir)
            solver.solve(_request(task_name="task-a"))
            solver.solve(_request(task_name="task-b"))

            self.assertEqual(len(list(cache_dir.glob("*.json"))), 2)

    def test_cache_key_differs_by_solution_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            solver = CachingSolver(model=None, timeout=0, config=_config(), inner=_dummy(), cache_dir=cache_dir)
            solver.solve(_request(solution_name="sol-1"))
            solver.solve(_request(solution_name="sol-2"))

            self.assertEqual(len(list(cache_dir.glob("*.json"))), 2)

    def test_cache_key_differs_by_commit_sha(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            solver = CachingSolver(model=None, timeout=0, config=_config(), inner=_dummy(), cache_dir=cache_dir)
            solver.solve(_request(commit_sha="aaa"))
            solver.solve(_request(commit_sha="bbb"))

            self.assertEqual(len(list(cache_dir.glob("*.json"))), 2)

    def test_same_request_fields_produce_same_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            solver = CachingSolver(model=None, timeout=0, config=_config(), inner=_dummy(), cache_dir=cache_dir)
            solver.solve(_request())
            solver.solve(_request())

            self.assertEqual(len(list(cache_dir.glob("*.json"))), 1)


class CachingSolverWriteOnlyTest(unittest.TestCase):
    def test_write_only_always_calls_inner(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            solver = CachingSolver(
                model=None, timeout=0, config=_config(),
                inner=_dummy("model-1"), cache_dir=cache_dir,
                read=False, write=True,
            )
            req = _request()
            solver.solve(req)

            solver2 = CachingSolver(
                model=None, timeout=0, config=_config(),
                inner=_dummy("model-2"), cache_dir=cache_dir,
                read=False, write=True,
            )
            result = solver2.solve(req)

            self.assertEqual(result.model, "model-2")

    def test_write_only_still_persists_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            solver = CachingSolver(
                model=None, timeout=0, config=_config(),
                inner=_dummy(), cache_dir=cache_dir,
                read=False, write=True,
            )
            req = _request()
            solver.solve(req)

            self.assertEqual(len(list(cache_dir.glob("*.json"))), 1)


class CachingSolverReadOnlyTest(unittest.TestCase):
    def test_read_only_returns_cached_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            req = _request()
            write_solver = CachingSolver(
                model=None, timeout=0, config=_config(),
                inner=_dummy("cached/model"), cache_dir=cache_dir,
            )
            write_solver.solve(req)

            read_solver = CachingSolver(
                model=None, timeout=0, config=_config(),
                inner=_dummy("live/model"), cache_dir=cache_dir,
                read=True, write=False,
            )
            result = read_solver.solve(req)

            self.assertEqual(result.model, "cached/model")

    def test_read_only_miss_calls_inner_but_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            solver = CachingSolver(
                model=None, timeout=0, config=_config(),
                inner=_dummy(), cache_dir=cache_dir,
                read=True, write=False,
            )
            result = solver.solve(_request())

            self.assertTrue(result.success)
            self.assertEqual(len(list(cache_dir.glob("*.json"))), 0)


class CachingSolverNoInnerTest(unittest.TestCase):
    def test_no_inner_raises_on_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            solver = CachingSolver(
                model=None, timeout=0, config=_config(),
                inner=None, cache_dir=Path(tmp),
            )
            with self.assertRaises(CacheMissError):
                solver.solve(_request())

    def test_no_inner_returns_cached_result_on_hit(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            req = _request()
            pre_cached = SolveResult(
                success=True,
                elapsed_seconds=1.0,
                raw_output="pre-cached",
                model="stored/model",
                solution_diff="diff",
                exit_reason=COMPLETED_EXIT_REASON,
            )
            write_solver = CachingSolver(
                model=None, timeout=0, config=_config(),
                inner=_dummy(), cache_dir=cache_dir,
            )
            write_solver.save(req, pre_cached)

            read_solver = CachingSolver(
                model=None, timeout=0, config=_config(),
                inner=None, cache_dir=cache_dir,
            )
            result = read_solver.solve(req)

            self.assertEqual(result.raw_output, "pre-cached")
            self.assertEqual(result.model, "stored/model")

    def test_no_inner_with_read_false_always_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            req = _request()
            write_solver = CachingSolver(
                model=None, timeout=0, config=_config(),
                inner=_dummy(), cache_dir=cache_dir,
            )
            write_solver.solve(req)

            solver = CachingSolver(
                model=None, timeout=0, config=_config(),
                inner=None, cache_dir=cache_dir,
                read=False, write=True,
            )
            with self.assertRaises(CacheMissError):
                solver.solve(req)


class CachingSolverRoundTripTest(unittest.TestCase):
    def test_result_fields_survive_serialization_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            req = _request()
            solver = CachingSolver(
                model=None, timeout=0, config=_config(),
                inner=_dummy("rt/model"), cache_dir=cache_dir,
            )
            original = solver.solve(req)

            solver2 = CachingSolver(
                model=None, timeout=0, config=_config(),
                inner=None, cache_dir=cache_dir,
            )
            restored = solver2.solve(req)

            self.assertEqual(restored.success, original.success)
            self.assertEqual(restored.model, original.model)
            self.assertEqual(restored.solution_diff, original.solution_diff)
            self.assertEqual(restored.exit_reason, original.exit_reason)


if __name__ == "__main__":
    unittest.main()
