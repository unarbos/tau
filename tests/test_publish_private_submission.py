from __future__ import annotations

import base64
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from config import RunConfig
from private_submission import run_private_submission_checks, write_private_submission_bundle
from validate import (
    ValidatorSubmission,
    _GITHUB_AGENT_MANIFEST_FILENAME,
    _is_private_submission,
    _publish_promoted_private_submission,
)

HOTKEY = "5GpkoLVCZWmxCN4hCchz2qsTDunw5wYbAsGuj8dyHRLqQrza"
SIGNATURE = "signed-by-hotkey"
BASE_HEAD = "a" * 40
BASE_TREE = "b" * 40
NEW_TREE = "c" * 40
NEW_COMMIT = "d" * 40
BLOB_SHA = "e" * 40
BASE_AGENT = """\
from typing import Optional

def solve(repo_path: str, issue: str, model: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None):
    return {"patch": "", "logs": "", "steps": 0, "cost": None, "success": True}
"""
MULTI_FILES = {
    "agent.py": BASE_AGENT,
    "agent/helper.py": "def helper():\n    return 1\n",
}
PASSING_JUDGE = {
    "verdict": "pass",
    "overall_score": 92,
    "real_edit_score": 92,
    "summary": "Real multi-file contribution.",
}


class FakeResponse:
    def __init__(self, status_code: int, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


class RecordingGitHubClient:
    def __init__(self, *, branch_head: str = BASE_HEAD, manifest: list[str] | None = None):
        self.branch_head = branch_head
        self.manifest = manifest
        self.calls: list[tuple[str, str, dict | None]] = []

    def get(self, url: str, params=None):
        self.calls.append(("GET", url, params))
        if url.endswith(f"/git/commits/{BASE_HEAD}"):
            return FakeResponse(200, {"tree": {"sha": BASE_TREE}})
        if url.endswith(f"/contents/{_GITHUB_AGENT_MANIFEST_FILENAME}"):
            if self.manifest is None:
                return FakeResponse(404)
            content = json.dumps(self.manifest, indent=2) + "\n"
            encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
            return FakeResponse(200, {"sha": BLOB_SHA, "content": encoded, "encoding": "base64"})
        if url.endswith("/contents/agent.py"):
            content = BASE_AGENT
            encoded = base64.b64encode(content.encode("utf-8")).decode("ascii")
            return FakeResponse(200, {"sha": BLOB_SHA, "content": encoded, "encoding": "base64"})
        if url.endswith("/branches/main"):
            return FakeResponse(200, {"commit": {"sha": self.branch_head}})
        return FakeResponse(404)

    def post(self, url: str, json=None):
        self.calls.append(("POST", url, json))
        if url.endswith("/git/trees"):
            return FakeResponse(201, {"sha": NEW_TREE})
        if url.endswith("/git/commits"):
            return FakeResponse(201, {"sha": NEW_COMMIT})
        return FakeResponse(404)

    def put(self, url: str, json=None):
        self.calls.append(("PUT", url, json))
        if "/contents/agent.py" in url:
            return FakeResponse(200, {"commit": {"sha": NEW_COMMIT}})
        return FakeResponse(404)

    def patch(self, url: str, json=None):
        self.calls.append(("PATCH", url, json))
        if url.endswith("/git/refs/heads/main"):
            return FakeResponse(200, {"object": {"sha": NEW_COMMIT}})
        return FakeResponse(404)


class PublishPrivateSubmissionTest(unittest.TestCase):
    def _write_bundle(self, tmp: str, *, files: dict[str, str], submission_id: str = "sub-mf"):
        root = Path(tmp) / "private-submissions"
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            base_agent_py=BASE_AGENT,
            submitted_files=files,
            openrouter_judge=lambda payload: PASSING_JUDGE,
        )
        write_private_submission_bundle(
            root=root,
            submission_id=submission_id,
            hotkey=HOTKEY,
            agent_files=files,
            check_result=result,
            signature=SIGNATURE,
            registration_block=100,
        )
        return root, submission_id, result.agent_sha256

    def _submission(self, *, commit_sha: str, submission_id: str) -> ValidatorSubmission:
        return ValidatorSubmission(
            hotkey=HOTKEY,
            uid=198,
            repo_full_name=f"private-submission/{submission_id}",
            repo_url=f"private-submission://{submission_id}",
            commit_sha=commit_sha,
            commitment=f"private-submission:{submission_id}:{commit_sha}",
            commitment_block=123,
            source="private",
        )

    def test_multifile_private_submission_publishes_to_github(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, submission_id, commit_sha = self._write_bundle(tmp, files=MULTI_FILES)
            config = RunConfig(validate_private_submission_root=root)
            submission = self._submission(commit_sha=commit_sha, submission_id=submission_id)
            client = RecordingGitHubClient(
                manifest=["agent.py", "agent/old_helper.py", _GITHUB_AGENT_MANIFEST_FILENAME],
            )

            with patch("validate._fetch_branch_head_sha", return_value=BASE_HEAD):
                published = _publish_promoted_private_submission(
                    github_client=client,
                    config=config,
                    submission=submission,
                )

        self.assertFalse(_is_private_submission(published))
        self.assertEqual(published.repo_full_name, "unarbos/ninja")
        self.assertEqual(published.commit_sha, NEW_COMMIT)
        tree_call = next(call for call in client.calls if call[0] == "POST" and call[1].endswith("/git/trees"))
        tree_entries = tree_call[2]["tree"]
        paths = {entry["path"] for entry in tree_entries if entry.get("content") is not None}
        self.assertEqual(
            paths,
            {"agent.py", "agent/helper.py", _GITHUB_AGENT_MANIFEST_FILENAME},
        )
        deleted = [entry["path"] for entry in tree_entries if "sha" in entry and entry.get("sha") is None]
        self.assertEqual(deleted, ["agent/old_helper.py"])

    def test_single_file_private_submission_removes_stale_manifest_files(self):
        files = {"agent.py": BASE_AGENT}
        with tempfile.TemporaryDirectory() as tmp:
            root, submission_id, commit_sha = self._write_bundle(tmp, files=files, submission_id="sub-single")
            config = RunConfig(validate_private_submission_root=root)
            submission = self._submission(commit_sha=commit_sha, submission_id=submission_id)
            client = RecordingGitHubClient(
                manifest=["agent.py", "agent/old_helper.py", _GITHUB_AGENT_MANIFEST_FILENAME],
            )

            with patch("validate._fetch_branch_head_sha", return_value=BASE_HEAD):
                published = _publish_promoted_private_submission(
                    github_client=client,
                    config=config,
                    submission=submission,
                )

        self.assertFalse(_is_private_submission(published))
        self.assertEqual(published.commit_sha, NEW_COMMIT)
        self.assertFalse(any(call[0] == "PUT" and "/contents/agent.py" in call[1] for call in client.calls))
        tree_call = next(call for call in client.calls if call[0] == "POST" and call[1].endswith("/git/trees"))
        tree_entries = tree_call[2]["tree"]
        written = [entry["path"] for entry in tree_entries if entry.get("content") is not None]
        deleted = sorted(entry["path"] for entry in tree_entries if "sha" in entry and entry.get("sha") is None)
        self.assertEqual(written, ["agent.py"])
        self.assertEqual(deleted, ["agent/old_helper.py", _GITHUB_AGENT_MANIFEST_FILENAME])

    def test_single_file_private_submission_from_single_file_base_writes_no_manifest(self):
        files = {"agent.py": BASE_AGENT}
        with tempfile.TemporaryDirectory() as tmp:
            root, submission_id, commit_sha = self._write_bundle(tmp, files=files, submission_id="sub-single")
            config = RunConfig(validate_private_submission_root=root)
            submission = self._submission(commit_sha=commit_sha, submission_id=submission_id)
            client = RecordingGitHubClient(manifest=None)

            with patch("validate._fetch_branch_head_sha", return_value=BASE_HEAD):
                published = _publish_promoted_private_submission(
                    github_client=client,
                    config=config,
                    submission=submission,
                )

        self.assertFalse(_is_private_submission(published))
        self.assertEqual(published.commit_sha, NEW_COMMIT)
        tree_call = next(call for call in client.calls if call[0] == "POST" and call[1].endswith("/git/trees"))
        tree_entries = tree_call[2]["tree"]
        written = [entry["path"] for entry in tree_entries if entry.get("content") is not None]
        deleted = [entry["path"] for entry in tree_entries if "sha" in entry and entry.get("sha") is None]
        self.assertEqual(written, ["agent.py"])
        self.assertEqual(deleted, [])

    def test_multifile_private_submission_from_single_file_base_adds_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, submission_id, commit_sha = self._write_bundle(tmp, files=MULTI_FILES)
            config = RunConfig(validate_private_submission_root=root)
            submission = self._submission(commit_sha=commit_sha, submission_id=submission_id)
            client = RecordingGitHubClient(manifest=None)

            with patch("validate._fetch_branch_head_sha", return_value=BASE_HEAD):
                published = _publish_promoted_private_submission(
                    github_client=client,
                    config=config,
                    submission=submission,
                )

        self.assertFalse(_is_private_submission(published))
        self.assertEqual(published.commit_sha, NEW_COMMIT)
        tree_call = next(call for call in client.calls if call[0] == "POST" and call[1].endswith("/git/trees"))
        tree_entries = tree_call[2]["tree"]
        written = {entry["path"] for entry in tree_entries if entry.get("content") is not None}
        deleted = [entry["path"] for entry in tree_entries if "sha" in entry and entry.get("sha") is None]
        self.assertEqual(written, {"agent.py", "agent/helper.py", _GITHUB_AGENT_MANIFEST_FILENAME})
        self.assertEqual(deleted, [])


if __name__ == "__main__":
    unittest.main()
