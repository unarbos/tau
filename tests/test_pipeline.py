import hashlib
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from config import SolverAgentSource
from cli import _resolve_agent_source
from tau.solver.docker_solver import _harness_runner_script, _materialize_agent_source
from pipeline import _solver_agent_file_path, _solver_agent_file_sha256


class PipelineAgentSourceTest(unittest.TestCase):
    def test_cli_resolves_local_agent_directory_as_local_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent_dir = Path(tmp) / "multi-agent"
            agent_dir.mkdir()
            (agent_dir / "agent.py").write_text("def solve(**kwargs):\n    return {}\n", encoding="utf-8")

            source = _resolve_agent_source(str(agent_dir), cwd=Path(tmp))

            self.assertEqual(source.kind, "local_path")
            self.assertEqual(source.local_path, str(agent_dir))
            self.assertEqual(source.agent_file, "agent.py")

    def test_materialize_agent_source_preserves_local_directory_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            agent_dir = root / "multi-agent"
            agent_dir.mkdir()
            agent_path = agent_dir / "agent.py"
            agent_path.write_text("def solve(**kwargs):\n    return {}\n", encoding="utf-8")
            (agent_dir / "helper.py").write_text("VALUE = 7\n", encoding="utf-8")

            config = type(
                "Config",
                (),
                {
                    "solver_agent_source": SolverAgentSource(
                        raw=str(agent_dir),
                        kind="local_path",
                        local_path=str(agent_dir),
                        agent_file="agent.py",
                    )
                },
            )()

            materialized_root, materialized_file = _materialize_agent_source(config=config, target_dir=root / "copy")

            self.assertEqual(materialized_root, agent_dir)
            self.assertEqual(materialized_file, agent_path)

    def test_harness_runner_allows_agent_sibling_imports(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            repo.mkdir()
            agent_dir = root / "agent-src"
            agent_dir.mkdir()
            (agent_dir / "helper.py").write_text("VALUE = 11\n", encoding="utf-8")
            agent = agent_dir / "agent.py"
            agent.write_text(
                """
from helper import VALUE

def solve(repo_path, issue, model, api_base, api_key):
    return {"success": VALUE == 11}
""".strip()
                + "\n",
                encoding="utf-8",
            )
            prompt = root / "prompt.txt"
            prompt.write_text("fix it", encoding="utf-8")
            runner = root / "runner.py"
            runner.write_text(_harness_runner_script(), encoding="utf-8")
            env = os.environ | {
                "TAU_AGENT_FILE": str(agent),
                "TAU_REPO_DIR": str(repo),
                "TAU_PROMPT_FILE": str(prompt),
                "TAU_HARNESS_RUNNER": str(runner),
                "AGENT_MODEL": "test/model",
                "OPENAI_BASE_URL": "http://127.0.0.1:1/v1",
                "OPENAI_API_KEY": "test-key",
            }

            proc = subprocess.run([sys.executable, str(runner)], env=env, text=True, capture_output=True)

            self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)

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
