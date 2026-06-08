import subprocess
import tempfile
import unittest
from pathlib import Path

from tau.solver.utils_docker import _git_metadata_sanitize_script


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=check,
        capture_output=True,
        text=True,
    )


class GitMetadataSanitizeTest(unittest.TestCase):
    def test_sanitize_prunes_unreachable_reference_commit(self):
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo with spaces"
            repo.mkdir()
            _git(repo, "init")
            _git(repo, "config", "user.email", "test@example.com")
            _git(repo, "config", "user.name", "Test User")

            (repo / "answer.txt").write_text("parent\n")
            _git(repo, "add", "answer.txt")
            _git(repo, "commit", "-m", "parent")
            parent_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

            (repo / "answer.txt").write_text("reference answer\n")
            _git(repo, "commit", "-am", "reference")
            reference_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

            _git(repo, "checkout", "--detach", parent_sha)
            git_dir = repo / ".git"
            (git_dir / "FETCH_HEAD").write_text(f"{reference_sha}\t\tbranch 'main'\n")
            (git_dir / "ORIG_HEAD").write_text(reference_sha + "\n")
            (git_dir / "refs" / "remotes" / "origin").mkdir(parents=True, exist_ok=True)
            (git_dir / "refs" / "remotes" / "origin" / "main").write_text(reference_sha + "\n")

            subprocess.run(
                ["bash", "-lc", _git_metadata_sanitize_script(str(repo))],
                check=True,
                capture_output=True,
                text=True,
            )

            self.assertEqual(_git(repo, "rev-parse", "HEAD").stdout.strip(), parent_sha)
            self.assertFalse((git_dir / "FETCH_HEAD").exists())
            self.assertFalse((git_dir / "ORIG_HEAD").exists())
            self.assertFalse((git_dir / "refs" / "remotes").exists())

            self.assertEqual(_git(repo, "cat-file", "-e", f"{parent_sha}^{{commit}}").returncode, 0)
            reference_lookup = _git(repo, "cat-file", "-e", f"{reference_sha}^{{commit}}", check=False)
            self.assertNotEqual(reference_lookup.returncode, 0)


if __name__ == "__main__":
    unittest.main()
