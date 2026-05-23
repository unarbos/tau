from __future__ import annotations

import subprocess
import tempfile
import unittest
from pathlib import Path

from compare import compare_solution_repos


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, text=True)


def _init_repo(repo: Path) -> None:
    repo.mkdir(parents=True)
    _git(repo, "init")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")


class CompareSolutionReposTests(unittest.TestCase):
    def test_compare_treats_directory_at_changed_file_path_as_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = root / "original"
            repo_a = root / "repo-a"
            repo_b = root / "repo-b"

            _init_repo(original)
            (original / "thing").write_text("before\n")
            _git(original, "add", "thing")
            _git(original, "commit", "-m", "initial")

            subprocess.run(["cp", "-a", str(original), str(repo_a)], check=True)
            subprocess.run(["cp", "-a", str(original), str(repo_b)], check=True)

            (repo_a / "thing").write_text("after\n")
            (repo_b / "thing").unlink()
            (repo_b / "thing").mkdir()
            (repo_b / "thing" / "nested.txt").write_text("after\n")

            result = compare_solution_repos(
                original_dir=original,
                repo_a_dir=repo_a,
                repo_b_dir=repo_b,
            )

            self.assertGreaterEqual(result.total_changed_lines_a, 1)
            self.assertGreaterEqual(result.total_changed_lines_b, 1)


if __name__ == "__main__":
    unittest.main()
