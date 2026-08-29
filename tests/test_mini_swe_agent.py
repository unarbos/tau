import importlib.util
import os
import subprocess
import tempfile
import unittest
from pathlib import Path


def _load_agent(path: Path):
    spec = importlib.util.spec_from_file_location("mini_swe_agent_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MiniSweAgentWrapperTest(unittest.TestCase):
    def test_timeout_budget_reserves_bounded_cleanup_window(self):
        agent = _load_agent(Path("agents/mini_swe_agent/agent.py").resolve())

        self.assertEqual(agent._cleanup_reserve_seconds(None), 30)
        self.assertEqual(agent._cleanup_reserve_seconds(120), 20)
        self.assertEqual(agent._cleanup_reserve_seconds(600), 90)

    def test_wrapper_runs_real_mini_entrypoint_and_returns_diff(self):
        agent = _load_agent(Path("agents/mini_swe_agent/agent.py").resolve())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "file.txt").write_text("before\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "add", "file.txt"], cwd=repo, check=True)
            subprocess.run(
                ["git", "-c", "user.email=a@b.c", "-c", "user.name=test", "commit", "-qm", "init"],
                cwd=repo,
                check=True,
            )

            bin_dir = root / "bin"
            bin_dir.mkdir()
            mini = bin_dir / "mini"
            mini.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf 'after\\n' > file.txt\n",
                encoding="utf-8",
            )
            mini.chmod(0o755)

            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}:{old_path}"
            try:
                result = agent.solve(
                    repo_path=str(repo),
                    issue="change file",
                    model="test-model",
                    api_base="http://127.0.0.1:1/v1",
                    api_key="test-key",
                )
            finally:
                os.environ["PATH"] = old_path

            self.assertTrue(result["success"])
            self.assertIn("-before", result["diff"])
            self.assertIn("+after", result["diff"])

    def test_wrapper_returns_partial_diff_when_mini_times_out(self):
        agent = _load_agent(Path("agents/mini_swe_agent/agent.py").resolve())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            (repo / "file.txt").write_text("before\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "add", "file.txt"], cwd=repo, check=True)
            subprocess.run(
                ["git", "-c", "user.email=a@b.c", "-c", "user.name=test", "commit", "-qm", "init"],
                cwd=repo,
                check=True,
            )

            bin_dir = root / "bin"
            bin_dir.mkdir()
            mini = bin_dir / "mini"
            mini.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                "printf 'after\\n' > file.txt\n"
                "sleep 5\n",
                encoding="utf-8",
            )
            mini.chmod(0o755)

            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{bin_dir}:{old_path}"
            try:
                result = agent.solve(
                    repo_path=str(repo),
                    issue="change file",
                    model="test-model",
                    api_base="http://127.0.0.1:1/v1",
                    api_key="test-key",
                    timeout_seconds=2,
                )
            finally:
                os.environ["PATH"] = old_path

            self.assertTrue(result["success"])
            self.assertIn("mini timed out", result["message"])
            self.assertIn("inner_timeout=1", result["message"])
            self.assertIn("-before", result["diff"])
            self.assertIn("+after", result["diff"])


if __name__ == "__main__":
    unittest.main()
