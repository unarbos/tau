import hashlib
import io
import json
import os
import tempfile
import unittest
from datetime import UTC, datetime
from email.message import Message
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from config import RunConfig
from private_submission import (
    PRIVATE_SUBMISSION_QUEUE_WAKEUP,
    build_public_submissions_api_payload,
    check_and_record_private_submission_attempt,
    private_submission_check_passed,
    private_submission_registration_check,
    private_submission_signature_payload,
    record_private_submission_acceptance,
    run_agent_smoke_checks,
    run_private_submission_checks,
    write_private_submission_bundle,
)
from solve_spend import build_solve_spend_payload
from submission_api import rate_limit_client_ip, solve_spend_payload_for_query
from validate import (
    ValidatorState,
    ValidatorSubmission,
    _agent_cache_entry_valid,
    _build_agent_config,
    _fetch_chain_submissions,
    _fetch_private_api_submissions,
    _is_private_submission,
    _reconcile_state_with_duel_history,
    _refresh_queue,
    _submission_is_eligible,
    _write_agent_cache_metadata,
)

HOTKEY = "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repoTitleHkey"
COLDKEY = "5ColdkeyOwnerForAgentUsernameProof111111111111111111"
SIGNATURE = "signed-by-hotkey"
BASE_AGENT = """\
from typing import Optional

def solve(repo_path: str, issue: str, model: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None):
    return {"patch": "", "logs": "", "steps": 0, "cost": None, "success": True}
"""
GOOD_AGENT = """\
from typing import Optional

def solve(repo_path: str, issue: str, model: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None):
    logs = "private submission"
    return {"patch": "", "logs": logs, "steps": 1, "cost": None, "success": True}
"""
BAD_AGENT = """\
import requests

def solve(repo_path, issue, model=None, api_base=None, api_key=None):
    return {"patch": "", "logs": "", "steps": 0, "cost": None, "success": True}
"""


class FakeResponse:
    status_code = 404

    def json(self):
        return {}


class FakeGithubClient:
    def get(self, path, params=None):
        return FakeResponse()


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
    def __init__(self, owners_by_hotkey=None):
        self.owners_by_hotkey = owners_by_hotkey or {}

    def query(self, **kwargs):
        if kwargs.get("storage_function") == "Owner":
            hotkey = kwargs.get("params", [""])[0]
            return FakeQueryResult(self.owners_by_hotkey.get(str(hotkey)))
        return FakeQueryResult()


class FakeSubtensor:
    block = 456

    def __init__(self, commitment: str, *, coldkey: str | None = None):
        self.commitments = FakeCommitments(commitment)
        self.subnets = FakeSubnets()
        self.substrate = FakeSubstrate({HOTKEY: coldkey} if coldkey is not None else {})


def fake_signature_verifier(hotkey, payload, signature):
    expected = private_submission_signature_payload(
        hotkey=HOTKEY,
        submission_id="sub-1",
        agent_sha256=hashlib.sha256(GOOD_AGENT.encode("utf-8")).hexdigest(),
    )
    return hotkey == HOTKEY and payload == expected and signature == SIGNATURE


class PrivateSubmissionChecksTest(unittest.TestCase):
    def test_rate_limit_client_ip_uses_forwarded_for_from_local_proxy(self):
        headers = Message()
        headers["X-Forwarded-For"] = "203.0.113.7, 127.0.0.1"

        client_ip = rate_limit_client_ip(headers=headers, client_address=("127.0.0.1", 39196))

        self.assertEqual(client_ip, "203.0.113.7")

    def test_rate_limit_client_ip_uses_real_ip_from_local_proxy(self):
        headers = Message()
        headers["X-Real-IP"] = "2001:db8::66"

        client_ip = rate_limit_client_ip(headers=headers, client_address=("::1", 39196))

        self.assertEqual(client_ip, "2001:db8::66")

    def test_rate_limit_client_ip_does_not_trust_forwarded_for_from_remote_peer(self):
        headers = Message()
        headers["X-Forwarded-For"] = "203.0.113.7"

        client_ip = rate_limit_client_ip(headers=headers, client_address=("198.51.100.8", 39196))

        self.assertEqual(client_ip, "198.51.100.8")

    def test_rate_limit_client_ip_falls_back_to_local_peer_without_proxy_header(self):
        headers = Message()

        client_ip = rate_limit_client_ip(headers=headers, client_address=("127.0.0.1", 39196))

        self.assertEqual(client_ip, "127.0.0.1")

    def test_scope_guard_failure_skips_judge_and_rejects(self):
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=BAD_AGENT,
            base_agent_py=BASE_AGENT,
            openrouter_judge=lambda payload: {"verdict": "pass", "overall_score": 99},
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.checks["scope_guard"].status, "failed")
        self.assertEqual(result.checks["openrouter_judge"].status, "skipped")
        self.assertTrue(
            any("non-stdlib module `requests`" in item for item in result.checks["scope_guard"].findings)
        )

    def test_smoke_check_rejects_pyflakes_regressions(self):
        result = run_agent_smoke_checks(agent_py=GOOD_AGENT + "\nimport os\n")

        self.assertEqual(result.status, "failed")
        self.assertTrue(any("os" in item for item in result.findings))

    def test_scope_guard_rejects_validator_owned_contract_edits(self):
        changed_agent = GOOD_AGENT.replace(
            "def solve(repo_path: str, issue: str, model: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None):",
            "def solve(repo_path: str, issue: str, temperature: float = 0.1, model: Optional[str] = None, api_base: Optional[str] = None, api_key: Optional[str] = None):",
        )
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=changed_agent,
            base_agent_py=GOOD_AGENT,
            openrouter_judge=lambda payload: {"verdict": "pass", "overall_score": 90},
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.checks["scope_guard"].status, "failed")
        self.assertTrue(any("solve()" in item or "validator-owned" in item for item in result.checks["scope_guard"].findings))

    def test_passing_checks_can_be_written_as_bundle(self):
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=GOOD_AGENT,
            base_agent_py=BASE_AGENT,
            openrouter_judge=lambda payload: {
                "verdict": "pass",
                "overall_score": 88,
                "summary": "Looks like a real local submission.",
                "reasons": ["changes runtime logs"],
            },
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.checks["scope_guard"].status, "passed")
        self.assertEqual(result.checks["openrouter_judge"].status, "passed")

        with tempfile.TemporaryDirectory() as tmp:
            bundle = write_private_submission_bundle(
                root=Path(tmp),
                submission_id="sub-1",
                hotkey=HOTKEY,
                agent_py=GOOD_AGENT,
                check_result=result,
                signature=SIGNATURE,
            )

            self.assertTrue((bundle / "agent.py").is_file())
            self.assertTrue((bundle / "check_result.json").is_file())
            saved = json.loads((bundle / "check_result.json").read_text())
            self.assertEqual(saved["checks"]["openrouter_judge"]["metadata"]["judgment"]["overall_score"], 88)
            self.assertEqual(saved["ci_checks"]["openrouter_judge"]["score"], 88)
            self.assertEqual(saved["llm_judge"]["metadata"]["judgment"]["summary"], "Looks like a real local submission.")

    def test_registration_gate_allows_one_submission_per_registration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = private_submission_registration_check(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-1",
                agent_sha256="a" * 64,
                registration_block=100,
            )
            self.assertEqual(first.status, "passed")

            record_private_submission_acceptance(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-1",
                agent_sha256="a" * 64,
                registration_block=100,
            )
            second = private_submission_registration_check(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-2",
                agent_sha256="b" * 64,
                registration_block=100,
            )
            self.assertEqual(second.status, "failed")
            self.assertTrue(any("re-register" in finding for finding in second.findings))

            after_reregistration = private_submission_registration_check(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-2",
                agent_sha256="b" * 64,
                registration_block=101,
            )
            self.assertEqual(after_reregistration.status, "passed")

    def test_registration_gate_requires_registration_block(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = private_submission_registration_check(
                root=Path(tmp),
                hotkey=HOTKEY,
                submission_id="sub-1",
                agent_sha256="a" * 64,
                registration_block=None,
            )

        self.assertEqual(result.status, "failed")
        self.assertTrue(any("Registration block is required" in item for item in result.findings))

    def test_hotkey_submission_attempt_limit_prunes_old_attempts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old_attempt = check_and_record_private_submission_attempt(
                root=root,
                hotkey=HOTKEY,
                submission_id="old",
                agent_sha256="a" * 64,
                max_attempts=1,
                window_seconds=60,
                now=datetime.fromtimestamp(0, UTC),
            )
            fresh_attempt = check_and_record_private_submission_attempt(
                root=root,
                hotkey=HOTKEY,
                submission_id="fresh",
                agent_sha256="b" * 64,
                max_attempts=1,
                window_seconds=60,
                now=datetime.fromtimestamp(61, UTC),
            )
            blocked = check_and_record_private_submission_attempt(
                root=root,
                hotkey=HOTKEY,
                submission_id="blocked",
                agent_sha256="c" * 64,
                max_attempts=1,
                window_seconds=60,
                now=datetime.fromtimestamp(62, UTC),
            )

        self.assertTrue(old_attempt["allowed"])
        self.assertTrue(fresh_attempt["allowed"])
        self.assertFalse(blocked["allowed"])
        self.assertEqual(blocked["attempts"], 1)

    def test_public_submissions_api_payload_excludes_private_code_and_signature(self):
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=GOOD_AGENT,
            base_agent_py=BASE_AGENT,
            openrouter_judge=lambda payload: {
                "verdict": "pass",
                "overall_score": 91,
                "summary": "Accepted.",
                "reasons": ["local improvement"],
            },
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_private_submission_bundle(
                root=root,
                submission_id="sub-1",
                hotkey=HOTKEY,
                agent_py=GOOD_AGENT,
                check_result=result,
                signature=SIGNATURE,
                registration_block=100,
            )
            record_private_submission_acceptance(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-1",
                agent_sha256=result.agent_sha256,
                registration_block=100,
            )

            payload = build_public_submissions_api_payload(root=root)

        self.assertEqual(payload["version"], 1)
        self.assertEqual(len(payload["submissions"]), 1)
        public_submission = payload["submissions"][0]
        self.assertEqual(public_submission["submission_id"], "sub-1")
        self.assertEqual(public_submission["hotkey"], HOTKEY)
        self.assertEqual(public_submission["registration_block"], 100)
        self.assertEqual(
            public_submission["commitment"],
            f"private-submission:sub-1:{result.agent_sha256}",
        )
        self.assertEqual(public_submission["llm_judge"]["judgment"]["overall_score"], 91)
        encoded = json.dumps(payload)
        self.assertNotIn(GOOD_AGENT, encoded)
        self.assertNotIn(SIGNATURE, encoded)
        self.assertNotIn("signature_payload", encoded)

    def test_bundle_requires_matching_hotkey_signature(self):
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=GOOD_AGENT,
            base_agent_py=BASE_AGENT,
            openrouter_judge=lambda payload: {"verdict": "pass", "overall_score": 90},
        )

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_private_submission_bundle(
                root=root,
                submission_id="sub-1",
                hotkey=HOTKEY,
                agent_py=GOOD_AGENT,
                check_result=result,
                signature=SIGNATURE,
            )

            self.assertTrue(
                private_submission_check_passed(
                    root,
                    "sub-1",
                    result.agent_sha256,
                    hotkey=HOTKEY,
                    signature_verifier=fake_signature_verifier,
                )
            )
            self.assertFalse(
                private_submission_check_passed(
                    root,
                    "sub-1",
                    result.agent_sha256,
                    hotkey="5GspoofedHotkeyCannotClaimThisSubmission",
                    signature_verifier=fake_signature_verifier,
                )
            )
            self.assertFalse(
                private_submission_check_passed(
                    root,
                    "sub-1",
                    result.agent_sha256,
                    hotkey=HOTKEY,
                    signature_verifier=lambda hotkey, payload, signature: False,
                )
            )


class PrivateSubmissionValidatorTest(unittest.TestCase):
    def test_public_agent_cache_requires_matching_metadata_and_file_hash(self):
        submission = ValidatorSubmission(
            hotkey=HOTKEY,
            uid=7,
            repo_full_name="miner/ninja",
            repo_url="https://github.com/miner/ninja.git",
            commit_sha="a" * 40,
            commitment="miner/ninja@" + "a" * 40,
            commitment_block=100,
            source="chain",
        )

        with tempfile.TemporaryDirectory() as tmp:
            cache_dir = Path(tmp)
            agent_path = cache_dir / "agent.py"
            agent_path.write_text(GOOD_AGENT, encoding="utf-8")

            self.assertFalse(
                _agent_cache_entry_valid(cache_dir=cache_dir, agent_path=agent_path, sub=submission)
            )

            _write_agent_cache_metadata(cache_dir=cache_dir, agent_path=agent_path, sub=submission)
            self.assertTrue(
                _agent_cache_entry_valid(cache_dir=cache_dir, agent_path=agent_path, sub=submission)
            )

            agent_path.write_text(GOOD_AGENT + "\n# changed\n", encoding="utf-8")
            self.assertFalse(
                _agent_cache_entry_valid(cache_dir=cache_dir, agent_path=agent_path, sub=submission)
            )

    def test_fetch_private_api_submissions_accepts_checked_private_bundle(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_private_submission_checks(
                hotkey=HOTKEY,
                submitted_agent_py=GOOD_AGENT,
                base_agent_py=BASE_AGENT,
                openrouter_judge=lambda payload: {"verdict": "pass", "overall_score": 90},
            )
            write_private_submission_bundle(
                root=root,
                submission_id="sub-1",
                hotkey=HOTKEY,
                agent_py=GOOD_AGENT,
                check_result=result,
                signature=SIGNATURE,
                registration_block=100,
            )
            record_private_submission_acceptance(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-1",
                agent_sha256=result.agent_sha256,
                registration_block=100,
            )
            commitment = f"private-submission:sub-1:{result.agent_sha256}"
            config = RunConfig(
                validate_private_submission_watch=True,
                validate_private_submission_root=root,
                validate_hotkey_spent_since_block=None,
            )

            with patch("validate._verify_hotkey_signature", fake_signature_verifier):
                submissions = _fetch_private_api_submissions(
                    subtensor=FakeSubtensor(commitment),
                    config=config,
                    state=ValidatorState(),
                )

                self.assertEqual(len(submissions), 1)
                self.assertEqual(submissions[0].source, "private")
                self.assertEqual(submissions[0].commit_sha, result.agent_sha256)
                self.assertTrue(
                    _submission_is_eligible(
                        subtensor=FakeSubtensor(commitment),
                        github_client=FakeGithubClient(),
                        config=config,
                        submission=submissions[0],
                    )
                )
                agent_config = _build_agent_config(config, submissions[0])
                self.assertEqual(agent_config.solver_agent_source.kind, "local_file")
                self.assertEqual(Path(agent_config.solver_agent_source.local_path).read_text(), GOOD_AGENT)

    def test_fetch_private_api_submissions_stores_verified_agent_username(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = run_private_submission_checks(
                hotkey=HOTKEY,
                submitted_agent_py=GOOD_AGENT,
                base_agent_py=BASE_AGENT,
                openrouter_judge=lambda payload: {"verdict": "pass", "overall_score": 90},
            )
            write_private_submission_bundle(
                root=root,
                submission_id="sub-1",
                hotkey=HOTKEY,
                agent_py=GOOD_AGENT,
                check_result=result,
                signature=SIGNATURE,
                registration_block=100,
            )
            record_private_submission_acceptance(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-1",
                agent_sha256=result.agent_sha256,
                registration_block=100,
            )
            ledger_path = root / "_accepted_submissions.json"
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
            ledger["hotkeys"][HOTKEY].update(
                {
                    "agent_username": "alice",
                    "coldkey": COLDKEY,
                    "coldkey_signature": "signed-by-coldkey",
                }
            )
            ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
            commitment = f"private-submission:sub-1:{result.agent_sha256}"
            config = RunConfig(
                validate_private_submission_watch=True,
                validate_private_submission_root=root,
                validate_hotkey_spent_since_block=None,
            )

            def verifier(hotkey, payload, signature):
                if hotkey == COLDKEY:
                    return payload == b"tau-agent-submission-username:alice" and signature == "signed-by-coldkey"
                return fake_signature_verifier(hotkey, payload, signature)

            with patch("validate._verify_hotkey_signature", verifier):
                submissions = _fetch_private_api_submissions(
                    subtensor=FakeSubtensor(commitment, coldkey=COLDKEY),
                    config=config,
                    state=ValidatorState(),
                )

        self.assertEqual(len(submissions), 1)
        self.assertEqual(submissions[0].agent_username, "alice")
        self.assertEqual(submissions[0].coldkey, COLDKEY)
        self.assertEqual(submissions[0].coldkey_signature, "signed-by-coldkey")

    def test_github_pr_commitments_are_not_submission_method(self):
        config = RunConfig(
            validate_hotkey_spent_since_block=None,
        )

        submissions = _fetch_private_api_submissions(
            subtensor=FakeSubtensor(f"github-pr:unarbos/ninja#7@{'a' * 40}"),
            config=config,
            state=ValidatorState(),
        )

        self.assertEqual(submissions, [])

    def test_refresh_queue_spends_hotkey_only_after_checked_acceptance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            passing = run_private_submission_checks(
                hotkey=HOTKEY,
                submitted_agent_py=GOOD_AGENT,
                base_agent_py=BASE_AGENT,
                openrouter_judge=lambda payload: {"verdict": "pass", "overall_score": 90},
            )
            write_private_submission_bundle(
                root=root,
                submission_id="sub-1",
                hotkey=HOTKEY,
                agent_py=GOOD_AGENT,
                check_result=passing,
                signature=SIGNATURE,
                registration_block=100,
            )
            record_private_submission_acceptance(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-1",
                agent_sha256=passing.agent_sha256,
                registration_block=100,
            )
            failed_sha = hashlib.sha256(BAD_AGENT.encode("utf-8")).hexdigest()
            config = RunConfig(
                validate_private_submission_watch=True,
                validate_private_submission_root=root,
                validate_hotkey_spent_since_block=None,
            )

            with patch("validate._verify_hotkey_signature", fake_signature_verifier):
                rejected = _fetch_private_api_submissions(
                    subtensor=FakeSubtensor(f"private-submission:missing:{failed_sha}"),
                    config=config,
                    state=ValidatorState(),
                )
                self.assertEqual(len(rejected), 1)

                accepted = _fetch_private_api_submissions(
                    subtensor=FakeSubtensor(f"private-submission:sub-1:{passing.agent_sha256}"),
                    config=config,
                    state=ValidatorState(),
                )
            state = ValidatorState()
            _refresh_queue(chain_submissions=accepted, config=config, state=state, subtensor=FakeSubtensor(""))

            self.assertEqual([item.hotkey for item in state.queue], [HOTKEY])
            self.assertEqual(state.locked_commitments[HOTKEY], f"private-submission:sub-1:{passing.agent_sha256}")

    def test_private_api_submission_queue_ignores_legacy_min_commitment_cutoff(self):
        submission = ValidatorSubmission(
            hotkey=HOTKEY,
            uid=42,
            repo_full_name="private-submission/sub-1",
            repo_url="private-submission://sub-1",
            commit_sha="a" * 64,
            commitment=f"private-submission:sub-1:{'a' * 64}",
            commitment_block=100,
            source="private",
        )
        config = RunConfig(
            validate_min_commitment_block=999,
            validate_hotkey_spent_since_block=None,
        )
        state = ValidatorState()

        _refresh_queue(chain_submissions=[submission], config=config, state=state, subtensor=None)

        self.assertEqual([item.commitment for item in state.queue], [submission.commitment])

    def test_acceptance_record_touches_validator_queue_wakeup(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)

            record_private_submission_acceptance(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-1",
                agent_sha256="a" * 64,
                registration_block=100,
            )

            self.assertTrue((root / PRIVATE_SUBMISSION_QUEUE_WAKEUP).is_file())

    def test_private_api_submission_queue_ignores_prior_public_hotkey_spend(self):
        submission = ValidatorSubmission(
            hotkey=HOTKEY,
            uid=42,
            repo_full_name="private-submission/sub-1",
            repo_url="private-submission://sub-1",
            commit_sha="a" * 64,
            commitment=f"private-submission:sub-1:{'a' * 64}",
            commitment_block=100,
            source="private",
        )
        config = RunConfig(validate_hotkey_spent_since_block=None)
        state = ValidatorState(
            seen_hotkeys=[HOTKEY],
            locked_commitments={HOTKEY: f"github-pr:unarbos/ninja#1@{'b' * 40}"},
            commitment_blocks_by_hotkey={HOTKEY: 50},
        )

        _refresh_queue(chain_submissions=[submission], config=config, state=state, subtensor=None)

        self.assertEqual([item.commitment for item in state.queue], [submission.commitment])
        self.assertEqual(state.locked_commitments[HOTKEY], submission.commitment)

    def test_chain_private_submission_commitments_are_ignored(self):
        commitment = f"private-submission:sub-1:{'a' * 64}"
        config = RunConfig(
            validate_private_submission_watch=True,
            validate_private_submission_root=Path("/tmp/private-submissions-test"),
            validate_hotkey_spent_since_block=None,
        )

        with patch("validate.private_submission_check_passed", return_value=True) as check_passed:
            submissions = _fetch_chain_submissions(
                subtensor=FakeSubtensor(commitment),
                github_client=FakeGithubClient(),
                config=config,
                state=ValidatorState(),
            )

        self.assertEqual(submissions, [])
        check_passed.assert_not_called()

    def test_revealed_chain_private_submission_does_not_mark_hotkey_spent(self):
        commitment = f"private-submission:sub-1:{'a' * 64}"
        public_commitment = "github-pr:unarbos/ninja#1@" + "b" * 40

        class RevealedPrivateCommitments(FakeCommitments):
            def get_all_revealed_commitments(self, netuid):
                return {HOTKEY: [{"block": 100, "commitment": commitment}]}

        class RevealedPrivateSubtensor(FakeSubtensor):
            def __init__(self):
                super().__init__(public_commitment)
                self.commitments = RevealedPrivateCommitments(public_commitment)

        config = RunConfig(
            validate_private_submission_watch=True,
            validate_private_submission_root=Path("/tmp/private-submissions-test"),
            validate_hotkey_spent_since_block=None,
        )
        state = ValidatorState()

        submissions = _fetch_chain_submissions(
            subtensor=RevealedPrivateSubtensor(),
            github_client=FakeGithubClient(),
            config=config,
            state=state,
        )

        self.assertEqual(submissions, [])
        self.assertNotIn(HOTKEY, state.locked_commitments)
        self.assertNotIn(HOTKEY, state.seen_hotkeys)

    def test_reconcile_keeps_private_queue_when_only_prior_public_commitment_completed(self):
        queued = ValidatorSubmission(
            hotkey=HOTKEY,
            uid=42,
            repo_full_name="private-submission/sub-1",
            repo_url="private-submission://sub-1",
            commit_sha="a" * 64,
            commitment=f"private-submission:sub-1:{'a' * 64}",
            commitment_block=100,
            source="private",
        )
        public_commitment = f"github-pr:unarbos/ninja#1@{'b' * 40}"
        state = ValidatorState(queue=[queued])

        with tempfile.TemporaryDirectory() as tmp:
            duels_dir = Path(tmp)
            (duels_dir / "000001.json").write_text(json.dumps({
                "duel_id": 1,
                "challenger": {
                    "hotkey": HOTKEY,
                    "commitment": public_commitment,
                    "commitment_block": 50,
                },
            }))

            _reconcile_state_with_duel_history(state, duels_dir)

        self.assertEqual([item.commitment for item in state.queue], [queued.commitment])

    def test_reconcile_removes_private_queue_when_same_private_commitment_completed(self):
        queued = ValidatorSubmission(
            hotkey=HOTKEY,
            uid=42,
            repo_full_name="private-submission/sub-1",
            repo_url="private-submission://sub-1",
            commit_sha="a" * 64,
            commitment=f"private-submission:sub-1:{'a' * 64}",
            commitment_block=100,
            source="private",
        )
        state = ValidatorState(queue=[queued])

        with tempfile.TemporaryDirectory() as tmp:
            duels_dir = Path(tmp)
            (duels_dir / "000001.json").write_text(json.dumps({
                "duel_id": 1,
                "challenger": {
                    "hotkey": HOTKEY,
                    "commitment": queued.commitment,
                    "commitment_block": queued.commitment_block,
                },
            }))

            _reconcile_state_with_duel_history(state, duels_dir)

        self.assertEqual(state.queue, [])

    def test_private_api_queue_is_fifo_by_acceptance_time(self):
        older = ValidatorSubmission(
            hotkey="5OlderPrivateSubmissionHotkey",
            uid=22,
            repo_full_name="private-submission/older",
            repo_url="private-submission://older",
            commit_sha="a" * 64,
            commitment=f"private-submission:older:{'a' * 64}",
            commitment_block=900,
            source="private",
            accepted_at="2026-05-15T01:00:00+00:00",
        )
        newer = ValidatorSubmission(
            hotkey="5NewerPrivateSubmissionHotkey",
            uid=11,
            repo_full_name="private-submission/newer",
            repo_url="private-submission://newer",
            commit_sha="b" * 64,
            commitment=f"private-submission:newer:{'b' * 64}",
            commitment_block=100,
            source="private",
            accepted_at="2026-05-15T02:00:00+00:00",
        )
        config = RunConfig(validate_hotkey_spent_since_block=None)
        state = ValidatorState()

        _refresh_queue(chain_submissions=[newer, older], config=config, state=state, subtensor=None)

        self.assertEqual([item.hotkey for item in state.queue], [older.hotkey, newer.hotkey])

    def test_existing_private_queue_is_repaired_to_fifo_when_acceptance_time_arrives(self):
        older = ValidatorSubmission(
            hotkey="5OlderPrivateSubmissionHotkey",
            uid=22,
            repo_full_name="private-submission/older",
            repo_url="private-submission://older",
            commit_sha="a" * 64,
            commitment=f"private-submission:older:{'a' * 64}",
            commitment_block=900,
            source="private",
            accepted_at="2026-05-15T01:00:00+00:00",
        )
        newer = ValidatorSubmission(
            hotkey="5NewerPrivateSubmissionHotkey",
            uid=11,
            repo_full_name="private-submission/newer",
            repo_url="private-submission://newer",
            commit_sha="b" * 64,
            commitment=f"private-submission:newer:{'b' * 64}",
            commitment_block=100,
            source="private",
            accepted_at="2026-05-15T02:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "_accepted_submissions.json").write_text(json.dumps({
                "version": 1,
                "hotkeys": {
                    older.hotkey: {
                        "registration_block": older.commitment_block,
                        "submission_id": "older",
                        "agent_sha256": older.commit_sha,
                        "accepted_at": older.accepted_at,
                    },
                    newer.hotkey: {
                        "registration_block": newer.commitment_block,
                        "submission_id": "newer",
                        "agent_sha256": newer.commit_sha,
                        "accepted_at": newer.accepted_at,
                    },
                },
            }))
            config = RunConfig(
                validate_hotkey_spent_since_block=None,
                validate_private_submission_root=root,
            )
            state = ValidatorState(queue=[
                ValidatorSubmission.from_dict({**newer.to_dict(), "accepted_at": None}),
                ValidatorSubmission.from_dict({**older.to_dict(), "accepted_at": None}),
            ])

            _refresh_queue(chain_submissions=[], config=config, state=state, subtensor=None)

        self.assertEqual([item.hotkey for item in state.queue], [older.hotkey, newer.hotkey])
        self.assertEqual([item.accepted_at for item in state.queue], [older.accepted_at, newer.accepted_at])

    def test_refresh_queue_removes_private_submission_from_prior_registration(self):
        stale = ValidatorSubmission(
            hotkey=HOTKEY,
            uid=42,
            repo_full_name="private-submission/stale",
            repo_url="private-submission://stale",
            commit_sha="a" * 64,
            commitment=f"private-submission:stale:{'a' * 64}",
            commitment_block=100,
            source="private",
            accepted_at="2026-05-24T21:51:16.885693+00:00",
        )
        current = ValidatorSubmission(
            hotkey="5CurrentPrivateSubmissionHotkey",
            uid=24,
            repo_full_name="private-submission/current",
            repo_url="private-submission://current",
            commit_sha="b" * 64,
            commitment=f"private-submission:current:{'b' * 64}",
            commitment_block=150,
            source="private",
            accepted_at="2026-05-28T14:53:28.941563+00:00",
        )
        state = ValidatorState(queue=[stale, current])
        config = RunConfig(validate_hotkey_spent_since_block=None)

        def registration_block(*, subtensor, config, hotkey, uid=None):
            return 125 if hotkey == HOTKEY else 150

        def current_uid(*, subtensor, hotkey, netuid):
            return 42 if hotkey == HOTKEY else 24

        with (
            patch("validate._uid_for_hotkey_on_subnet", current_uid),
            patch("validate._current_registration_block", registration_block),
        ):
            _refresh_queue(
                chain_submissions=[],
                config=config,
                state=state,
                subtensor=FakeSubtensor(""),
            )

        self.assertEqual([item.hotkey for item in state.queue], [current.hotkey])

    def test_refresh_queue_removes_unregistered_private_submission(self):
        stale = ValidatorSubmission(
            hotkey="5UnregisteredPrivateSubmissionHotkey",
            uid=42,
            repo_full_name="private-submission/stale",
            repo_url="private-submission://stale",
            commit_sha="a" * 64,
            commitment=f"private-submission:stale:{'a' * 64}",
            commitment_block=100,
            source="private",
            accepted_at="2026-05-24T21:51:16.885693+00:00",
        )
        current = ValidatorSubmission(
            hotkey=HOTKEY,
            uid=42,
            repo_full_name="private-submission/current",
            repo_url="private-submission://current",
            commit_sha="b" * 64,
            commitment=f"private-submission:current:{'b' * 64}",
            commitment_block=150,
            source="private",
            accepted_at="2026-05-28T14:53:28.941563+00:00",
        )
        state = ValidatorState(queue=[stale, current])
        config = RunConfig(validate_hotkey_spent_since_block=None)

        _refresh_queue(
            chain_submissions=[],
            config=config,
            state=state,
            subtensor=FakeSubtensor(""),
        )

        self.assertEqual([item.hotkey for item in state.queue], [current.hotkey])

    def test_published_private_submission_is_no_longer_runtime_private(self):
        submission = ValidatorSubmission(
            hotkey=HOTKEY,
            uid=42,
            repo_full_name="unarbos/ninja",
            repo_url="https://github.com/unarbos/ninja.git",
            commit_sha="f" * 40,
            commitment=f"private-submission:sub-1:{'a' * 64}",
            commitment_block=123,
            source="private_published",
        )

        self.assertFalse(_is_private_submission(submission))


class PrivateSubmissionApiTest(unittest.TestCase):
    def test_invalid_signature_fails_before_ci_checks(self):
        from submission_api import SubmissionApiConfig, handle_submission_request

        boundary = "----test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="hotkey"\r\n\r\n'
            f"{HOTKEY}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="submission_id"\r\n\r\n'
            "sub-1\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="signature"\r\n\r\n'
            "bad-signature\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="agent"; filename="agent.py"\r\n'
            "Content-Type: text/x-python\r\n\r\n"
            f"{BAD_AGENT}\r\n"
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
                judge=lambda payload: (_ for _ in ()).throw(AssertionError("judge should not run")),
                judge_min_score=70,
            )

            with patch("submission_api._verify_hotkey_signature", return_value=False):
                status, payload = handle_submission_request(
                    headers=headers,
                    rfile=io.BytesIO(body),
                    config=config,
                )

        self.assertEqual(status, 401)
        self.assertFalse(payload["accepted"])
        self.assertFalse(payload["signature_valid"])
        self.assertEqual(list(payload["ci_checks"].keys()), ["hotkey_signature"])

    def test_registration_failure_fails_before_ci_checks(self):
        from submission_api import SubmissionApiConfig, handle_submission_request

        boundary = "----test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="hotkey"\r\n\r\n'
            f"{HOTKEY}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="submission_id"\r\n\r\n'
            "sub-1\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="signature"\r\n\r\n'
            f"{SIGNATURE}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="agent"; filename="agent.py"\r\n'
            "Content-Type: text/x-python\r\n\r\n"
            f"{BAD_AGENT}\r\n"
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
                judge=lambda payload: (_ for _ in ()).throw(AssertionError("judge should not run")),
                judge_min_score=70,
            )

            with patch("submission_api._verify_hotkey_signature", return_value=True):
                with patch("submission_api.registration_context", return_value=(None, None, "not registered")):
                    status, payload = handle_submission_request(
                        headers=headers,
                        rfile=io.BytesIO(body),
                        config=config,
                    )

        self.assertEqual(status, 422)
        self.assertFalse(payload["accepted"])
        self.assertTrue(payload["signature_valid"])
        self.assertEqual(list(payload["ci_checks"].keys()), ["registration_gate"])

    def test_exact_resubmission_returns_existing_acceptance_without_ci_checks(self):
        from submission_api import SubmissionApiConfig, handle_submission_request

        boundary = "----test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="hotkey"\r\n\r\n'
            f"{HOTKEY}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="submission_id"\r\n\r\n'
            "sub-1\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="signature"\r\n\r\n'
            f"{SIGNATURE}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="agent"; filename="agent.py"\r\n'
            "Content-Type: text/x-python\r\n\r\n"
            f"{GOOD_AGENT}\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        headers = Message()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        headers["Content-Length"] = str(len(body))
        judge_calls = 0

        def judge(payload):
            nonlocal judge_calls
            judge_calls += 1
            if judge_calls > 1:
                raise AssertionError("judge should not rerun for exact resubmission")
            return {"verdict": "pass", "overall_score": 90}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "private-submissions"
            base_agent = Path(tmp) / "base_agent.py"
            base_agent.write_text(BASE_AGENT, encoding="utf-8")
            config = SubmissionApiConfig(
                private_submission_root=root,
                base_agent=base_agent,
                run_config=RunConfig(validate_netuid=66),
                judge=judge,
                judge_min_score=70,
            )

            with patch("submission_api._verify_hotkey_signature", return_value=True):
                with patch("submission_api.registration_context", return_value=(100, 42, None)):
                    with patch("submission_api.publish_submissions_api_data") as publish:
                        first_status, first_payload = handle_submission_request(
                            headers=headers,
                            rfile=io.BytesIO(body),
                            config=config,
                        )
                        second_status, second_payload = handle_submission_request(
                            headers=headers,
                            rfile=io.BytesIO(body),
                            config=config,
                        )

        self.assertEqual(first_status, 200)
        self.assertTrue(first_payload["accepted"])
        self.assertEqual(second_status, 200)
        self.assertTrue(second_payload["accepted"])
        self.assertTrue(second_payload["already_accepted"])
        self.assertEqual(
            second_payload["message"],
            "This exact private submission was already accepted; no CI or LLM checks were rerun.",
        )
        self.assertIsNone(second_payload["agent_username"])
        self.assertIsNone(second_payload["coldkey"])
        self.assertEqual(judge_calls, 1)
        self.assertEqual(publish.call_count, 2)
        self.assertEqual(list(second_payload["ci_checks"].keys()), ["registration_gate"])
        self.assertIsNone(second_payload["llm_judge"])
        self.assertEqual(
            second_payload["ci_checks"]["registration_gate"]["summary"],
            "This exact private submission is already accepted for the current registration.",
        )

    def test_hotkey_rate_limit_blocks_before_ci_checks(self):
        from submission_api import SubmissionApiConfig, handle_submission_request

        boundary = "----test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="hotkey"\r\n\r\n'
            f"{HOTKEY}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="submission_id"\r\n\r\n'
            "sub-1\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="signature"\r\n\r\n'
            f"{SIGNATURE}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="agent"; filename="agent.py"\r\n'
            "Content-Type: text/x-python\r\n\r\n"
            f"{GOOD_AGENT}\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        headers = Message()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        headers["Content-Length"] = str(len(body))
        judge_calls = 0

        def judge(payload):
            nonlocal judge_calls
            judge_calls += 1
            return {"verdict": "fail", "overall_score": 10}

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "private-submissions"
            base_agent = Path(tmp) / "base_agent.py"
            base_agent.write_text(BASE_AGENT, encoding="utf-8")
            config = SubmissionApiConfig(
                private_submission_root=root,
                base_agent=base_agent,
                run_config=RunConfig(validate_netuid=66),
                judge=judge,
                judge_min_score=70,
                hotkey_rate_limit_max_attempts=2,
                hotkey_rate_limit_window_seconds=86_400,
            )

            with patch("submission_api._verify_hotkey_signature", return_value=True):
                with patch("submission_api.registration_context", return_value=(100, 42, None)):
                    first_status, first_payload = handle_submission_request(
                        headers=headers,
                        rfile=io.BytesIO(body),
                        config=config,
                    )
                    second_status, second_payload = handle_submission_request(
                        headers=headers,
                        rfile=io.BytesIO(body),
                        config=config,
                    )
                    third_status, third_payload = handle_submission_request(
                        headers=headers,
                        rfile=io.BytesIO(body),
                        config=config,
                    )

        self.assertEqual(first_status, 422)
        self.assertEqual(second_status, 422)
        self.assertEqual(third_status, 429)
        self.assertFalse(first_payload["accepted"])
        self.assertFalse(second_payload["accepted"])
        self.assertEqual(third_payload["error"], "hotkey_rate_limited")
        self.assertEqual(third_payload["rate_limit"]["attempts"], 2)
        self.assertEqual(list(third_payload["ci_checks"].keys()), ["hotkey_rate_limit"])
        self.assertEqual(judge_calls, 2)

    def test_exact_resubmission_does_not_fall_through_when_bundle_is_missing(self):
        from submission_api import SubmissionApiConfig, handle_submission_request

        boundary = "----test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="hotkey"\r\n\r\n'
            f"{HOTKEY}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="submission_id"\r\n\r\n'
            "sub-1\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="signature"\r\n\r\n'
            f"{SIGNATURE}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="agent"; filename="agent.py"\r\n'
            "Content-Type: text/x-python\r\n\r\n"
            f"{GOOD_AGENT}\r\n"
            f"--{boundary}--\r\n"
        ).encode()
        headers = Message()
        headers["Content-Type"] = f"multipart/form-data; boundary={boundary}"
        headers["Content-Length"] = str(len(body))
        agent_sha256 = hashlib.sha256(GOOD_AGENT.encode("utf-8")).hexdigest()

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "private-submissions"
            record_private_submission_acceptance(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-1",
                agent_sha256=agent_sha256,
                registration_block=100,
            )
            config = SubmissionApiConfig(
                private_submission_root=root,
                base_agent=Path(tmp) / "missing-base-agent.py",
                run_config=RunConfig(validate_netuid=66),
                judge=lambda payload: (_ for _ in ()).throw(AssertionError("judge should not run")),
                judge_min_score=70,
            )

            with patch("submission_api._verify_hotkey_signature", return_value=True):
                with patch("submission_api.registration_context", return_value=(100, 42, None)):
                    with patch("submission_api.publish_submissions_api_data") as publish:
                        status, payload = handle_submission_request(
                            headers=headers,
                            rfile=io.BytesIO(body),
                            config=config,
                        )

        self.assertEqual(status, 200)
        self.assertTrue(payload["accepted"])
        self.assertTrue(payload["already_accepted"])
        self.assertEqual(payload["agent_sha256"], agent_sha256)
        self.assertEqual(list(payload["ci_checks"].keys()), ["registration_gate"])
        self.assertIsNone(payload["llm_judge"])
        publish.assert_called_once()

    def test_cli_private_submit_registration_failure_skips_ci_checks(self):
        from cli import _run_private_submit

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "private-submissions"
            record_private_submission_acceptance(
                root=root,
                hotkey=HOTKEY,
                submission_id="prior",
                agent_sha256="a" * 64,
                registration_block=100,
            )
            agent_path = Path(tmp) / "agent.py"
            agent_path.write_text(GOOD_AGENT, encoding="utf-8")
            args = SimpleNamespace(
                agent=agent_path,
                base_agent=Path(tmp) / "missing-base-agent.py",
                hotkey=HOTKEY,
                signature=SIGNATURE,
                submission_id="sub-2",
                private_submission_root=root,
                workspace_root=Path(tmp),
                netuid=66,
                network=None,
                subtensor_endpoint=None,
                registration_block=None,
                agent_username=None,
                coldkey=None,
                coldkey_signature=None,
                overwrite=False,
                skip_openrouter_judge=False,
                judge_min_score=70,
                judge_model=None,
            )

            with patch("validate._verify_hotkey_signature", return_value=True):
                with patch("cli._private_submit_registration_context", return_value=(100, 42, None)):
                    with patch(
                        "cli._build_private_submission_openrouter_judge",
                        side_effect=AssertionError("judge should not be built"),
                    ):
                        with self.assertRaises(SystemExit) as raised:
                            _run_private_submit(args)

        self.assertEqual(raised.exception.code, 1)

    def test_cli_private_submit_exact_resubmission_skips_ci_checks_from_ledger(self):
        from cli import _run_private_submit

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "private-submissions"
            agent_sha256 = hashlib.sha256(GOOD_AGENT.encode("utf-8")).hexdigest()
            record_private_submission_acceptance(
                root=root,
                hotkey=HOTKEY,
                submission_id="sub-1",
                agent_sha256=agent_sha256,
                registration_block=100,
            )
            agent_path = Path(tmp) / "agent.py"
            agent_path.write_text(GOOD_AGENT, encoding="utf-8")
            args = SimpleNamespace(
                agent=agent_path,
                base_agent=Path(tmp) / "missing-base-agent.py",
                hotkey=HOTKEY,
                signature=SIGNATURE,
                submission_id="sub-1",
                private_submission_root=root,
                workspace_root=Path(tmp),
                netuid=66,
                network=None,
                subtensor_endpoint=None,
                registration_block=None,
                agent_username=None,
                coldkey=None,
                coldkey_signature=None,
                overwrite=False,
                skip_openrouter_judge=False,
                judge_min_score=70,
                judge_model=None,
            )

            with patch("validate._verify_hotkey_signature", return_value=True):
                with patch("cli._private_submit_registration_context", return_value=(100, 42, None)):
                    with patch(
                        "cli._build_private_submission_openrouter_judge",
                        side_effect=AssertionError("judge should not be built"),
                    ):
                        with patch("r2.publish_submissions_api_data") as publish:
                            with patch("builtins.print") as printed:
                                _run_private_submit(args)

        publish.assert_called_once()
        payload = json.loads(printed.call_args.args[0])
        self.assertTrue(payload["already_accepted"])
        self.assertEqual(payload["agent_sha256"], agent_sha256)
        self.assertEqual(list(payload["ci_checks"].keys()), ["registration_gate"])
        self.assertIsNone(payload["llm_judge"])

    def test_unverified_agent_username_is_not_stored_or_published(self):
        from submission_api import SubmissionApiConfig, handle_submission_request

        boundary = "----test-boundary"
        body = (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="hotkey"\r\n\r\n'
            f"{HOTKEY}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="submission_id"\r\n\r\n'
            "sub-1\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="signature"\r\n\r\n'
            f"{SIGNATURE}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="agent_username"\r\n\r\n'
            "not-alice\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="coldkey"\r\n\r\n'
            f"{COLDKEY}\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="coldkey_signature"\r\n\r\n'
            "bad-coldkey-signature\r\n"
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="agent"; filename="agent.py"\r\n'
            "Content-Type: text/x-python\r\n\r\n"
            f"{GOOD_AGENT}\r\n"
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
                judge=lambda payload: {"verdict": "pass", "overall_score": 90},
                judge_min_score=70,
            )

            with patch("submission_api._verify_hotkey_signature", return_value=True):
                with patch("submission_api.registration_context", return_value=(100, 42, None)):
                    with patch("submission_api._verified_submission_identity_from_config", return_value=None):
                        with patch("submission_api.publish_submissions_api_data"):
                            status, payload = handle_submission_request(
                                headers=headers,
                                rfile=io.BytesIO(body),
                                config=config,
                            )

            ledger = json.loads((root / "_accepted_submissions.json").read_text(encoding="utf-8"))
            public_payload = build_public_submissions_api_payload(root=root)

        self.assertEqual(status, 200)
        self.assertTrue(payload["accepted"])
        self.assertIsNone(payload["agent_username"])
        self.assertIsNone(payload["coldkey"])
        self.assertNotIn("agent_username", ledger["hotkeys"][HOTKEY])
        self.assertNotIn("coldkey", ledger["hotkeys"][HOTKEY])
        self.assertNotIn("agent_username", public_payload["submissions"][0])
        self.assertNotIn("coldkey", public_payload["submissions"][0])

    def test_private_submission_judge_rejects_weak_component_scores(self):
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=GOOD_AGENT,
            base_agent_py=BASE_AGENT,
            min_score=70,
            openrouter_judge=lambda payload: {
                "verdict": "pass",
                "overall_score": 72,
                "real_edit_score": 69,
                "safety_score": 95,
                "scope_score": 90,
                "contract_score": 95,
                "summary": "Mostly reorders existing gates.",
                "reasons": ["moves syntax fix before hail mary"],
            },
        )

        self.assertFalse(result.accepted)
        check = result.checks["openrouter_judge"]
        self.assertEqual(check.status, "failed")
        self.assertTrue(any("real_edit_score=69" in item for item in check.findings))


    def test_private_submission_judge_warn_does_not_auto_accept(self):
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=GOOD_AGENT,
            base_agent_py=BASE_AGENT,
            min_score=70,
            openrouter_judge=lambda payload: {
                "verdict": "warn",
                "overall_score": 85,
                "real_edit_score": 82,
                "safety_score": 95,
                "scope_score": 90,
                "contract_score": 95,
                "summary": "Possibly useful but needs review.",
                "reasons": ["uncertain signal"],
            },
        )

        self.assertFalse(result.accepted)
        self.assertEqual(result.checks["openrouter_judge"].status, "warn")

    def test_private_submission_judge_rejects_borderline_cosmetic_goodhart_risk(self):
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=GOOD_AGENT,
            base_agent_py=BASE_AGENT,
            min_score=70,
            openrouter_judge=lambda payload: {
                "verdict": "pass",
                "overall_score": 70,
                "real_edit_score": 72,
                "safety_score": 95,
                "scope_score": 90,
                "contract_score": 95,
                "summary": "Mostly comment cleanup with a tiny scoring tweak.",
                "reasons": ["changes comments and normalizes newlines"],
                "risks": ["cosmetic-copy: significant comment churn and bulk unchanged file"],
            },
        )

        self.assertFalse(result.accepted)
        check = result.checks["openrouter_judge"]
        self.assertEqual(check.status, "failed")
        self.assertTrue(any("cosmetic-copy/goodhart" in item for item in check.findings))

    def test_private_submission_judge_allows_positive_goodhart_mention(self):
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=GOOD_AGENT,
            base_agent_py=BASE_AGENT,
            min_score=70,
            openrouter_judge=lambda payload: {
                "verdict": "pass",
                "overall_score": 70,
                "real_edit_score": 70,
                "safety_score": 95,
                "scope_score": 90,
                "contract_score": 95,
                "summary": "Functional fix; no Goodhart risk in the scoring path.",
                "reasons": ["avoids goodhart incentives by preserving real behavior"],
                "risks": [],
            },
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.checks["openrouter_judge"].status, "passed")

    def test_private_submission_judge_allows_positive_non_contribution_mentions(self):
        positive_mentions = (
            "Functional change; no cosmetic-copy risk remains.",
            "Functional change; no comment churn risk remains.",
            "Functional change; newline normalization is not present.",
            "Functional change; this is not merely normalizes newlines.",
            "Functional change; parameter-only is not present.",
        )
        for summary in positive_mentions:
            with self.subTest(summary=summary):
                result = run_private_submission_checks(
                    hotkey=HOTKEY,
                    submitted_agent_py=GOOD_AGENT,
                    base_agent_py=BASE_AGENT,
                    min_score=70,
                    openrouter_judge=lambda payload: {
                        "verdict": "pass",
                        "overall_score": 70,
                        "real_edit_score": 70,
                        "safety_score": 95,
                        "scope_score": 90,
                        "contract_score": 95,
                        "summary": summary,
                        "reasons": [],
                        "risks": [],
                    },
                )

                self.assertTrue(result.accepted)
                self.assertEqual(result.checks["openrouter_judge"].status, "passed")

    def test_private_submission_judge_ignores_unstructured_risk_prose(self):
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            submitted_agent_py=GOOD_AGENT,
            base_agent_py=BASE_AGENT,
            min_score=70,
            openrouter_judge=lambda payload: {
                "verdict": "pass",
                "overall_score": 70,
                "real_edit_score": 70,
                "safety_score": 95,
                "scope_score": 90,
                "contract_score": 95,
                "summary": "Avoids Goodhart incentives but the patch is mostly comment churn.",
                "reasons": [],
                "risks": [],
            },
        )

        self.assertTrue(result.accepted)
        self.assertEqual(result.checks["openrouter_judge"].status, "passed")

    def test_private_submission_judge_rejects_structured_non_contribution_risks(self):
        risk_payloads = (
            ["comment-churn"],
            ["comment-churn: comments make up most of the patch"],
            ["newline-normalization"],
            ["newline-normalization - mostly line ending churn"],
            ["parameter-only"],
            ["parameter-only: only changes a threshold"],
            [{"category": "goodhart", "evidence": "score-targeted behavior"}],
        )
        for risks in risk_payloads:
            with self.subTest(risks=risks):
                result = run_private_submission_checks(
                    hotkey=HOTKEY,
                    submitted_agent_py=GOOD_AGENT,
                    base_agent_py=BASE_AGENT,
                    min_score=70,
                    openrouter_judge=lambda payload: {
                        "verdict": "pass",
                        "overall_score": 74,
                        "real_edit_score": 74,
                        "safety_score": 95,
                        "scope_score": 90,
                        "contract_score": 95,
                        "summary": "Functional-looking but judge reported a concrete risk.",
                        "reasons": [],
                        "risks": risks,
                    },
                )

                self.assertFalse(result.accepted)
                self.assertEqual(result.checks["openrouter_judge"].status, "failed")

    def test_private_submission_judge_prompt_excludes_non_contributions(self):
        from cli import _PRIVATE_SUBMISSION_JUDGE_SYSTEM_PROMPT

        prompt = _PRIVATE_SUBMISSION_JUDGE_SYSTEM_PROMPT

        self.assertIn("actual contribution to the real function of the agent", prompt)
        self.assertIn("newline normalization", prompt)
        self.assertIn("parameter/constant/cap changes", prompt)
        self.assertIn("count as zero contribution", prompt)


    def test_private_submission_judge_uses_private_claude_prompt(self):
        from cli import _build_private_submission_openrouter_judge

        calls = []

        def fake_complete_text(**kwargs):
            calls.append(kwargs)
            return json.dumps(
                {
                    "verdict": "warn",
                    "overall_score": 55,
                    "real_edit_score": 50,
                    "safety_score": 90,
                    "scope_score": 80,
                    "contract_score": 90,
                    "summary": "Looks suspicious.",
                    "reasons": ["reason"],
                    "risks": ["risk"],
                    "required_changes": ["change"],
                }
            )

        args = SimpleNamespace(agent_timeout=123, judge_model=None)
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key"}, clear=False):
            with patch("openrouter_client.complete_text", side_effect=fake_complete_text):
                judge = _build_private_submission_openrouter_judge(args)
                result = judge(
                    {
                        "patch": "diff",
                        "base_agent_py": BASE_AGENT,
                        "submitted_agent_py": GOOD_AGENT,
                    }
                )

        self.assertEqual(result["verdict"], "warn")
        self.assertEqual(len(calls), 1)
        call = calls[0]
        self.assertEqual(call["model"], "google/gemini-3.1-flash-lite")
        self.assertEqual(call["temperature"], 0)
        self.assertIsNone(call["reasoning"])
        self.assertIn("CI gatekeeping judge", call["system_prompt"])
        self.assertIn("private Subnet 66 ninja submission API", call["system_prompt"])
        self.assertIn("Reorder-only / gate-order changes", call["system_prompt"])
        self.assertIn("hail-mary", call["system_prompt"])
        self.assertIn("real_edit_score", call["system_prompt"])
        self.assertIn("Deletion is not itself a", call["system_prompt"])
        self.assertIn("small amount of positive credit", call["system_prompt"])
        self.assertIn("credit modest", call["system_prompt"])
        self.assertIn("<submission_data>", call["prompt"])
        self.assertIn('"patch": "diff"', call["prompt"])
        self.assertIn("base_files", call["prompt"])
        self.assertIn("base_agent_py", call["prompt"])
        self.assertIn("submitted_agent_py", call["prompt"])
        self.assertIn("current public base harness files", call["system_prompt"])
        self.assertIn("full harness", call["system_prompt"])
        self.assertNotIn("<pr_data>", call["prompt"])

    def test_solve_spend_payload_sums_recent_solve_costs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tasks_root = Path(tmp) / "tasks"
            recent = tasks_root / "task-a" / "solutions" / "challenger-1" / "solve.json"
            old = tasks_root / "task-a" / "solutions" / "king-1" / "solve.json"
            recent.parent.mkdir(parents=True)
            old.parent.mkdir(parents=True)
            recent.write_text(
                json.dumps(
                    {
                        "solution_name": "challenger-1",
                        "result": {"model": "minimax/minimax-m2.7", "cost": 1.25},
                    }
                ),
                encoding="utf-8",
            )
            old.write_text(
                json.dumps(
                    {
                        "solution_name": "king-1",
                        "result": {"model": "minimax/minimax-m2.7", "cost": 9.0},
                    }
                ),
                encoding="utf-8",
            )
            os.utime(recent, (1_000, 1_000))
            os.utime(old, (100, 100))

            payload = build_solve_spend_payload(tasks_root=tasks_root, now=1_000, window_seconds=60)

        self.assertEqual(payload["solve_count"], 1)
        self.assertEqual(payload["total_cost_usd"], 1.25)
        self.assertEqual(payload["by_solution_prefix_usd"], {"challenger": 1.25})

    def test_solve_spend_endpoint_query_clamps_window(self):
        config = SimpleNamespace(
            run_config=SimpleNamespace(tasks_root=Path("/tmp/does-not-exist")),
        )

        payload = solve_spend_payload_for_query(config=config, query="window_seconds=0")

        self.assertEqual(payload["window_seconds"], 1)



if __name__ == "__main__":
    unittest.main()
