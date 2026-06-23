import hashlib
import io
import json
import subprocess
import tempfile
import unittest
from email.message import Message
from pathlib import Path
from unittest.mock import patch

from config import RunConfig
from docker_solver import _agent_source_sha256, _materialize_agent_source
from private_submission import (
    agent_bundle_sha256,
    agent_file_path_violations,
    agent_files_violations,
    normalize_base_agent_files,
    normalize_agent_files,
    private_submission_bundle_files,
    private_submission_check_passed,
    private_submission_signature_payload,
    record_private_submission_acceptance,
    run_private_submission_checks,
    write_private_submission_bundle,
)
from validate import (
    ValidatorState,
    ValidatorSubmission,
    _build_agent_config,
    _cached_agent_source,
    _fetch_private_api_submissions,
    _materialize_agent_cache,
)

HOTKEY = "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repoTitleHkey"
SIGNATURE = "signed-by-hotkey"
BASE_AGENT = """\
from typing import Optional

def solve(repo_path: str, issue: str, model: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None):
    return {"patch": "", "logs": "", "steps": 0, "cost": None, "success": True}
"""
MULTI_ENTRY = """\
from typing import Optional

from helpers.steps import describe_run

def solve(repo_path: str, issue: str, model: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None):
    logs = describe_run(repo_path, issue)
    return {"patch": "", "logs": logs, "steps": 1, "cost": None, "success": True}
"""
MULTI_HELPER = """\
def describe_run(repo_dir, task):
    return f"running in {repo_dir}: {task[:40]}"
"""
MULTI_FILES = {
    "agent.py": MULTI_ENTRY,
    "helpers/__init__.py": "",
    "helpers/steps.py": MULTI_HELPER,
}
PASSING_JUDGE = {
    "verdict": "pass",
    "overall_score": 92,
    "real_edit_score": 92,
    "summary": "Real multi-file contribution.",
}


class FakeCommitments:
    def __init__(self, commitment: str):
        self.commitment = commitment

    def get_all_revealed_commitments(self, netuid):
        return {}

    def get_all_commitments(self, netuid):
        return {HOTKEY: self.commitment}

    def get_commitment_metadata(self, netuid, hotkey):
        return {"block": 123}


class FakeSubnets:
    def get_uid_for_hotkey_on_subnet(self, hotkey, netuid):
        return 42 if hotkey == HOTKEY else None


class FakeQueryResult:
    def __init__(self, value=100):
        self.value = value


class FakeSubstrate:
    def query(self, **kwargs):
        return FakeQueryResult()


class FakeSubtensor:
    block = 456

    def __init__(self, commitment: str):
        self.commitments = FakeCommitments(commitment)
        self.subnets = FakeSubnets()
        self.substrate = FakeSubstrate()


def multi_signature_verifier(sha256_hex: str):
    expected = private_submission_signature_payload(
        hotkey=HOTKEY,
        submission_id="sub-mf",
        agent_sha256=sha256_hex,
    )

    def verifier(hotkey, payload, signature):
        return hotkey == HOTKEY and payload == expected and signature == SIGNATURE

    return verifier


class AgentFilesValidationTest(unittest.TestCase):
    def test_normalize_merges_legacy_agent_with_extra_files(self):
        files = normalize_agent_files(agent_py="entry", files={"helpers/util.py": "x = 1\n"})
        self.assertEqual(files, {"agent.py": "entry", "helpers/util.py": "x = 1\n"})

    def test_normalize_rejects_conflicting_agent_py(self):
        with self.assertRaises(ValueError):
            normalize_agent_files(agent_py="one", files={"agent.py": "two"})

    def test_violations_require_entrypoint(self):
        violations = agent_files_violations({"helpers/steps.py": MULTI_HELPER})
        self.assertTrue(any("agent.py" in item for item in violations))

    def test_path_violations(self):
        for bad_path in (
            "/abs/agent.py",
            "../escape.py",
            "helpers/../escape.py",
            "helpers\\steps.py",
            "helpers/.hidden.py",
            "notes.txt",
            "a/b/c/d/e/f/g/h/i.py",
        ):
            self.assertTrue(agent_file_path_violations(bad_path), bad_path)
        for good_path in ("agent.py", "helpers/__init__.py", "pkg/sub/module-name.py"):
            self.assertEqual(agent_file_path_violations(good_path), [], good_path)

    def test_bundle_sha256_single_file_matches_legacy_hash(self):
        files = {"agent.py": BASE_AGENT}
        self.assertEqual(
            agent_bundle_sha256(files),
            hashlib.sha256(BASE_AGENT.encode("utf-8")).hexdigest(),
        )

    def test_bundle_sha256_changes_when_module_changes(self):
        first = agent_bundle_sha256(MULTI_FILES)
        second = agent_bundle_sha256({**MULTI_FILES, "helpers/steps.py": MULTI_HELPER + "# x\n"})
        self.assertNotEqual(first, second)


class MultiFileChecksTest(unittest.TestCase):
    def test_multifile_submission_passes_checks(self):
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            base_agent_py=BASE_AGENT,
            submitted_files=MULTI_FILES,
            openrouter_judge=lambda payload: PASSING_JUDGE,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.agent_sha256, agent_bundle_sha256(MULTI_FILES))

    def test_judge_payload_lists_all_files(self):
        captured = {}

        def judge(payload):
            captured.update(payload)
            return PASSING_JUDGE

        run_private_submission_checks(
            hotkey=HOTKEY,
            base_agent_py=BASE_AGENT,
            submitted_files=MULTI_FILES,
            openrouter_judge=judge,
        )
        filenames = [item["filename"] for item in captured["changed_files"]]
        self.assertEqual(filenames, sorted(MULTI_FILES))
        self.assertIn("/dev/null", captured["patch"])
        self.assertIn("b/helpers/steps.py", captured["patch"])
        self.assertEqual(captured["base_files"], {"agent.py": BASE_AGENT})
        self.assertEqual(captured["submitted_files"], MULTI_FILES)

    def test_judge_payload_compares_against_multifile_base(self):
        captured = {}

        def judge(payload):
            captured.update(payload)
            return PASSING_JUDGE

        base_files = normalize_base_agent_files(
            base_agent_py=BASE_AGENT,
            files={
                "agent.py": BASE_AGENT,
                "helpers/__init__.py": "",
                "helpers/steps.py": "def old_helper():\n    return 'old'\n",
            },
        )
        run_private_submission_checks(
            hotkey=HOTKEY,
            base_agent_py=BASE_AGENT,
            base_files=base_files,
            submitted_agent_py=BASE_AGENT,
            openrouter_judge=judge,
        )

        self.assertEqual(captured["base_files"], base_files)
        self.assertIn("a/helpers/steps.py", captured["patch"])
        self.assertIn("/dev/null", captured["patch"])
        statuses = {item["filename"]: item["status"] for item in captured["changed_files"]}
        self.assertEqual(statuses["helpers/steps.py"], "deleted")
        self.assertEqual(statuses["helpers/__init__.py"], "deleted")

    def test_module_with_non_stdlib_import_is_rejected(self):
        files = {**MULTI_FILES, "helpers/steps.py": "import requests\n" + MULTI_HELPER}
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            base_agent_py=BASE_AGENT,
            submitted_files=files,
        )
        self.assertEqual(result.checks["scope_guard"].status, "failed")
        self.assertTrue(
            any("non-stdlib" in item for item in result.checks["scope_guard"].findings)
        )

    def test_module_redefining_contract_is_rejected(self):
        files = {**MULTI_FILES, "helpers/shadow.py": "def solve(a, b):\n    return None\n"}
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            base_agent_py=BASE_AGENT,
            submitted_files=files,
        )
        self.assertEqual(result.checks["scope_guard"].status, "failed")

    def test_module_reading_disallowed_env_is_rejected(self):
        files = {
            **MULTI_FILES,
            "helpers/env.py": 'import os\n\nTOKEN = os.environ.get("SOME_SECRET_TOKEN", "")\n',
        }
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            base_agent_py=BASE_AGENT,
            submitted_files=files,
        )
        self.assertEqual(result.checks["scope_guard"].status, "failed")

    def test_relative_imports_inside_package_are_allowed(self):
        files = {
            "agent.py": MULTI_ENTRY,
            "helpers/__init__.py": "",
            "helpers/steps.py": "from .util import describe_run\n\n__all__ = [\"describe_run\"]\n",
            "helpers/util.py": MULTI_HELPER,
        }
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            base_agent_py=BASE_AGENT,
            submitted_files=files,
            openrouter_judge=lambda payload: PASSING_JUDGE,
        )
        self.assertTrue(result.accepted, result.checks["scope_guard"].findings)

    def test_single_file_legacy_call_still_works(self):
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=BASE_AGENT,
            base_agent_py=BASE_AGENT,
            openrouter_judge=lambda payload: PASSING_JUDGE,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(
            result.agent_sha256,
            hashlib.sha256(BASE_AGENT.encode("utf-8")).hexdigest(),
        )


class MultiFileBundleTest(unittest.TestCase):
    def _accepted_result(self):
        return run_private_submission_checks(
            hotkey=HOTKEY,
            base_agent_py=BASE_AGENT,
            submitted_files=MULTI_FILES,
            openrouter_judge=lambda payload: PASSING_JUDGE,
        )

    def test_bundle_roundtrip_and_verification(self):
        result = self._accepted_result()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = write_private_submission_bundle(
                root=root,
                submission_id="sub-mf",
                hotkey=HOTKEY,
                agent_files=MULTI_FILES,
                check_result=result,
                signature=SIGNATURE,
                registration_block=100,
            )
            self.assertTrue((bundle / "agent.py").is_file())
            self.assertTrue((bundle / "helpers" / "steps.py").is_file())
            saved = json.loads((bundle / "check_result.json").read_text(encoding="utf-8"))
            self.assertEqual(sorted(saved["agent_files"]), sorted(MULTI_FILES))

            files = private_submission_bundle_files(root=root, submission_id="sub-mf")
            self.assertEqual(files, MULTI_FILES)
            self.assertTrue(
                private_submission_check_passed(
                    root,
                    "sub-mf",
                    result.agent_sha256,
                    hotkey=HOTKEY,
                    signature_verifier=multi_signature_verifier(result.agent_sha256),
                )
            )

    def test_tampered_module_fails_verification(self):
        result = self._accepted_result()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = write_private_submission_bundle(
                root=root,
                submission_id="sub-mf",
                hotkey=HOTKEY,
                agent_files=MULTI_FILES,
                check_result=result,
                signature=SIGNATURE,
                registration_block=100,
            )
            (bundle / "helpers" / "steps.py").write_text("def describe_run(a, b):\n    return 'evil'\n")
            self.assertIsNone(private_submission_bundle_files(root=root, submission_id="sub-mf"))
            self.assertFalse(
                private_submission_check_passed(
                    root,
                    "sub-mf",
                    result.agent_sha256,
                    hotkey=HOTKEY,
                    signature_verifier=multi_signature_verifier(result.agent_sha256),
                )
            )

    def test_stray_python_file_fails_verification(self):
        result = self._accepted_result()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = write_private_submission_bundle(
                root=root,
                submission_id="sub-mf",
                hotkey=HOTKEY,
                agent_files=MULTI_FILES,
                check_result=result,
                signature=SIGNATURE,
                registration_block=100,
            )
            (bundle / "helpers" / "smuggled.py").write_text("x = 1\n")
            self.assertIsNone(private_submission_bundle_files(root=root, submission_id="sub-mf"))

    def test_bundle_rejects_traversal_paths(self):
        result = self._accepted_result()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_private_submission_bundle(
                    root=Path(tmp),
                    submission_id="sub-mf",
                    hotkey=HOTKEY,
                    agent_files={**MULTI_FILES, "../escape.py": "x = 1\n"},
                    check_result=result,
                    signature=SIGNATURE,
                    registration_block=100,
                )


class MultiFileSubmissionApiTest(unittest.TestCase):
    def test_api_accepts_multifile_submission(self):
        from submission_api import SubmissionApiConfig, handle_submission_request

        extra_files = {key: value for key, value in MULTI_FILES.items() if key != "agent.py"}
        boundary = "----test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="hotkey"\r\n\r\n'
            f"{HOTKEY}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="submission_id"\r\n\r\n'
            "sub-mf\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="signature"\r\n\r\n'
            f"{SIGNATURE}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="agent"; filename="agent.py"\r\n'
            "Content-Type: text/x-python\r\n\r\n"
            f"{MULTI_ENTRY}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="files"; filename="files.json"\r\n'
            "Content-Type: application/json\r\n\r\n"
            f"{json.dumps(extra_files)}\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        headers = Message()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        headers["Content-Length"] = str(len(body))

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "private-submissions"
            base_agent = Path(tmp) / "base_agent.py"
            base_agent.write_text(BASE_AGENT, encoding="utf-8")
            config = SubmissionApiConfig(
                private_submission_root=root,
                base_agent=base_agent,
                run_config=RunConfig(validate_netuid=66),
                judge=lambda payload: PASSING_JUDGE,
                judge_min_score=70,
            )
            with patch("submission_api._verify_hotkey_signature", return_value=True):
                with patch("submission_api.registration_context", return_value=(100, 42, None)):
                    with patch("submission_api.publish_submissions_api_data"):
                        status, payload = handle_submission_request(
                            headers=headers,
                            rfile=io.BytesIO(body),
                            config=config,
                        )

            expected_sha = agent_bundle_sha256(MULTI_FILES)
            self.assertEqual(status, 200, payload)
            self.assertTrue(payload["accepted"])
            self.assertEqual(payload["agent_sha256"], expected_sha)
            self.assertEqual(payload["commitment"], f"private-submission:sub-mf:{expected_sha}")
            bundle = root / "sub-mf"
            self.assertTrue((bundle / "helpers" / "steps.py").is_file())
            self.assertEqual(
                private_submission_bundle_files(root=root, submission_id="sub-mf"),
                MULTI_FILES,
            )

    def test_api_rejects_invalid_files_json(self):
        from submission_api import SubmissionApiConfig, handle_submission_request

        boundary = "----test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="hotkey"\r\n\r\n'
            f"{HOTKEY}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="signature"\r\n\r\n'
            f"{SIGNATURE}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="agent"; filename="agent.py"\r\n'
            "Content-Type: text/x-python\r\n\r\n"
            f"{MULTI_ENTRY}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="files"; filename="files.json"\r\n'
            "Content-Type: application/json\r\n\r\n"
            "this-is-not-json\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        headers = Message()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        headers["Content-Length"] = str(len(body))

        with tempfile.TemporaryDirectory() as tmp:
            base_agent = Path(tmp) / "base_agent.py"
            base_agent.write_text(BASE_AGENT, encoding="utf-8")
            config = SubmissionApiConfig(
                private_submission_root=Path(tmp) / "private-submissions",
                base_agent=base_agent,
                run_config=RunConfig(validate_netuid=66),
                judge=lambda payload: PASSING_JUDGE,
                judge_min_score=70,
            )
            status, payload = handle_submission_request(
                headers=headers,
                rfile=io.BytesIO(body),
                config=config,
            )

        self.assertEqual(status, 400)
        self.assertFalse(payload["accepted"])
        self.assertIn("invalid agent files", payload["error"])


class MultiFileValidatorPickupTest(unittest.TestCase):
    def test_validator_stages_multifile_private_submission(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "private-submissions"
            result = run_private_submission_checks(
                hotkey=HOTKEY,
                base_agent_py=BASE_AGENT,
                submitted_files=MULTI_FILES,
                openrouter_judge=lambda payload: PASSING_JUDGE,
            )
            write_private_submission_bundle(
                root=root,
                submission_id="sub-mf",
                hotkey=HOTKEY,
                agent_files=MULTI_FILES,
                check_result=result,
                signature=SIGNATURE,
                registration_block=100,
            )
            record_private_submission_acceptance(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-mf",
                agent_sha256=result.agent_sha256,
                registration_block=100,
            )
            commitment = f"private-submission:sub-mf:{result.agent_sha256}"
            config = RunConfig(
                workspace_root=workspace,
                validate_private_submission_watch=True,
                validate_private_submission_root=root,
                validate_hotkey_spent_since_block=None,
            )

            with patch("validate._verify_hotkey_signature", multi_signature_verifier(result.agent_sha256)):
                submissions = _fetch_private_api_submissions(
                    subtensor=FakeSubtensor(commitment),
                    config=config,
                    state=ValidatorState(),
                )
                self.assertEqual(len(submissions), 1)
                self.assertEqual(submissions[0].commit_sha, result.agent_sha256)

                agent_config = _build_agent_config(config, submissions[0])
                source = agent_config.solver_agent_source
                self.assertEqual(source.kind, "local_path")
                staged = Path(source.local_path)
                self.assertEqual(
                    (staged / "agent.py").read_text(encoding="utf-8"),
                    MULTI_ENTRY,
                )
                self.assertEqual(
                    (staged / "helpers" / "steps.py").read_text(encoding="utf-8"),
                    MULTI_HELPER,
                )
                self.assertFalse((staged / "check_result.json").exists())

                agent_root, agent_file = _materialize_agent_source(
                    config=agent_config,
                    target_dir=workspace / "scratch",
                )
                self.assertEqual(agent_root, staged)
                self.assertEqual(agent_file, staged / "agent.py")
                self.assertEqual(
                    _agent_source_sha256(agent_root=agent_root, agent_file=agent_file),
                    result.agent_sha256,
                )

    def test_tampered_staged_module_is_restaged_from_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "private-submissions"
            result = run_private_submission_checks(
                hotkey=HOTKEY,
                base_agent_py=BASE_AGENT,
                submitted_files=MULTI_FILES,
                openrouter_judge=lambda payload: PASSING_JUDGE,
            )
            write_private_submission_bundle(
                root=root,
                submission_id="sub-mf",
                hotkey=HOTKEY,
                agent_files=MULTI_FILES,
                check_result=result,
                signature=SIGNATURE,
                registration_block=100,
            )
            sub = ValidatorSubmission(
                hotkey=HOTKEY,
                uid=42,
                repo_full_name="private-submission/sub-mf",
                repo_url="private-submission://sub-mf",
                commit_sha=result.agent_sha256,
                commitment=f"private-submission:sub-mf:{result.agent_sha256}",
                commitment_block=123,
                source="private",
            )
            config = RunConfig(
                workspace_root=workspace,
                validate_private_submission_watch=True,
                validate_private_submission_root=root,
                validate_hotkey_spent_since_block=None,
            )
            with patch("validate._verify_hotkey_signature", multi_signature_verifier(result.agent_sha256)):
                first = _cached_agent_source(config, sub)
                staged_module = Path(first.local_path) / "helpers" / "steps.py"
                staged_module.write_text("tampered = True\n", encoding="utf-8")
                second = _cached_agent_source(config, sub)
                self.assertEqual(
                    (Path(second.local_path) / "helpers" / "steps.py").read_text(encoding="utf-8"),
                    MULTI_HELPER,
                )


class GithubManifestCacheTest(unittest.TestCase):
    def _make_remote(self, tmp: Path) -> tuple[Path, str]:
        remote = tmp / "remote"
        remote.mkdir()
        (remote / "agent.py").write_text(MULTI_ENTRY, encoding="utf-8")
        (remote / "helpers").mkdir()
        (remote / "helpers" / "__init__.py").write_text("", encoding="utf-8")
        (remote / "helpers" / "steps.py").write_text(MULTI_HELPER, encoding="utf-8")
        (remote / "tau_agent_files.json").write_text(
            json.dumps(sorted(MULTI_FILES)),
            encoding="utf-8",
        )
        subprocess.run(["git", "init", "-q"], cwd=remote, check=True)
        subprocess.run(["git", "add", "-A"], cwd=remote, check=True)
        subprocess.run(
            ["git", "-c", "user.email=a@b.c", "-c", "user.name=test", "commit", "-qm", "init"],
            cwd=remote,
            check=True,
        )
        subprocess.run(
            ["git", "config", "uploadpack.allowAnySHA1InWant", "true"],
            cwd=remote,
            check=True,
        )
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=remote,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        return remote, sha

    def test_manifest_repo_materializes_all_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            remote, sha = self._make_remote(workspace)
            sub = ValidatorSubmission(
                hotkey=HOTKEY,
                uid=42,
                repo_full_name="org/multi-agent",
                repo_url=str(remote),
                commit_sha=sha,
                commitment=f"org/multi-agent@{sha}",
                commitment_block=123,
            )
            config = RunConfig(workspace_root=workspace, validate_netuid=66)

            agent_path, multi_file = _materialize_agent_cache(config, sub)
            self.assertTrue(multi_file)
            self.assertEqual(agent_path.read_text(encoding="utf-8"), MULTI_ENTRY)
            agent_dir = agent_path.parent
            self.assertEqual(
                (agent_dir / "helpers" / "steps.py").read_text(encoding="utf-8"),
                MULTI_HELPER,
            )

            source = _cached_agent_source(config, sub)
            self.assertEqual(source.kind, "local_path")
            self.assertEqual(Path(source.local_path), agent_dir)

            cached_path, cached_multi = _materialize_agent_cache(config, sub)
            self.assertTrue(cached_multi)
            self.assertEqual(cached_path, agent_path)

    def test_repo_without_manifest_keeps_single_file_extraction(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            remote, sha = self._make_remote(workspace)
            (remote / "tau_agent_files.json").unlink()
            subprocess.run(["git", "add", "-A"], cwd=remote, check=True)
            subprocess.run(
                ["git", "-c", "user.email=a@b.c", "-c", "user.name=test", "commit", "-qm", "drop manifest"],
                cwd=remote,
                check=True,
            )
            sha = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=remote,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            sub = ValidatorSubmission(
                hotkey=HOTKEY,
                uid=42,
                repo_full_name="org/single-agent",
                repo_url=str(remote),
                commit_sha=sha,
                commitment=f"org/single-agent@{sha}",
                commitment_block=123,
            )
            config = RunConfig(workspace_root=workspace, validate_netuid=66)

            agent_path, multi_file = _materialize_agent_cache(config, sub)
            self.assertFalse(multi_file)
            self.assertEqual(agent_path.read_text(encoding="utf-8"), MULTI_ENTRY)
            source = _cached_agent_source(config, sub)
            self.assertEqual(source.kind, "local_file")


if __name__ == "__main__":
    unittest.main()
