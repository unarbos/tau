import hashlib
import tempfile
import unittest
from pathlib import Path

from config import SolverAgentSource
from pipeline import _solver_agent_file_path, _solver_agent_file_sha256


class PipelineAgentSourceTest(unittest.TestCase):
    def test_solver_agent_file_sha256_hashes_local_agent_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent_path = Path(tmp) / "agent.py"
            agent_py = "def solve(repo_path, issue, model=None, api_base=None, api_key=None):\n    return {}\n"
            agent_path.write_text(agent_py, encoding="utf-8")
            source = SolverAgentSource(
                raw="private-submission/sub-1@abc",
                kind="local_file",
                local_path=str(agent_path),
                agent_file="agent.py",
                commit_sha=hashlib.sha256(agent_py.encode("utf-8")).hexdigest(),
            )

            self.assertEqual(_solver_agent_file_path(source), agent_path)
            self.assertEqual(_solver_agent_file_sha256(source), source.commit_sha)

    def test_solver_agent_file_path_uses_agent_file_for_local_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent_dir = Path(tmp)
            agent_path = agent_dir / "agent.py"
            agent_path.write_text("x = 1\n", encoding="utf-8")
            source = SolverAgentSource(
                raw="repo@sha",
                kind="local_path",
                local_path=str(agent_dir),
                agent_file="agent.py",
            )

            self.assertEqual(_solver_agent_file_path(source), agent_path)

    def test_solver_agent_file_sha256_ignores_remote_sources(self):
        source = SolverAgentSource(
            raw="repo@sha",
            kind="github_repo",
            repo_url="https://github.com/example/repo.git",
            agent_file="agent.py",
            commit_sha="a" * 40,
        )

        self.assertIsNone(_solver_agent_file_path(source))
        self.assertIsNone(_solver_agent_file_sha256(source))
