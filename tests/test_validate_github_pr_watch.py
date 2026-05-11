import base64
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from config import RunConfig
from validate import (
    _BURN_KING_HOTKEY,
    _BURN_KING_UID,
    ValidatorState,
    ValidatorSubmission,
    _build_burn_king,
    _cleanup_stale_github_prs,
    _dashboard_submission_dict,
    _enforce_submission_mode_on_state,
    _ensure_king,
    _fetch_chain_submissions,
    _github_pr_required_checks_passed,
    _hotkey_spent_since_block,
    _is_burn_king,
    _maybe_set_weights,
    _merge_promoted_github_pr,
    _refresh_queue,
    _submission_is_eligible,
)


SHA = "a" * 40
BASE_SHA = "b" * 40
RESOLVED_SHA = "c" * 40
MERGE_SHA = "d" * 40
ANCESTOR_SHA = "e" * 40
MINER_HOTKEY = "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repoTitleHkey"
OTHER_HOTKEY = "5G3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1otherMinerHkey"
PR_COMMITMENT = f"github-pr:unarbos/ninja#7@{SHA}"
PR_HEAD_COMMITMENT = f"github-pr-head:unarbos/ninja@{SHA}"


class FakeResponse:
    def __init__(self, status_code, payload, *, text: str | None = None):
        self.status_code = status_code
        self._payload = payload
        self.text = text if text is not None else str(payload)

    def json(self):
        return self._payload


class FakeGithubClient:
    def __init__(self, *, title: str | None = None):
        self.calls = []
        self.title = title or f"{MINER_HOTKEY} Improve harness"

    def get(self, path, params=None):
        self.calls.append((path, params))
        if path == "/repos/unarbos/ninja/pulls/7":
            return FakeResponse(
                200,
                {
                    "number": 7,
                    "state": "open",
                    "draft": False,
                    "title": self.title,
                    "html_url": "https://github.com/unarbos/ninja/pull/7",
                    "base": {
                        "ref": "main",
                        "repo": {"full_name": "unarbos/ninja"},
                    },
                    "head": {
                        "sha": SHA,
                        "repo": {
                            "full_name": "miner/ninja",
                            "clone_url": "https://github.com/miner/ninja.git",
                        },
                    },
                },
            )
        if path in {
            f"/repos/unarbos/ninja/commits/{SHA}/check-runs",
            f"/repos/miner/ninja/commits/{SHA}/check-runs",
        }:
            return FakeResponse(
                200,
                {
                    "check_runs": [
                        {"name": "PR Scope Guard", "status": "completed", "conclusion": "success"},
                        {"name": "OpenRouter PR Judge", "status": "completed", "conclusion": "success"},
                    ]
                },
            )
        if path == "/repos/miner/ninja":
            return FakeResponse(200, {"private": False})
        if path == f"/repos/miner/ninja/commits/{SHA}":
            return FakeResponse(200, {"sha": SHA})
        if path == "/repos/unarbos/ninja":
            return FakeResponse(200, {"private": False})
        if path == f"/repos/unarbos/ninja/commits/{BASE_SHA}":
            return FakeResponse(200, {"sha": BASE_SHA})
        if path == "/repos/unarbos/ninja/branches/main":
            return FakeResponse(200, {"commit": {"sha": BASE_SHA}})
        if path == f"/repos/unarbos/ninja/compare/{BASE_SHA}...main":
            return FakeResponse(200, {"status": "identical"})
        raise AssertionError(f"unexpected GitHub path: {path}")

    def put(self, path, json=None):
        raise AssertionError(f"unexpected GitHub PUT path: {path}")

    def post(self, path, json=None):
        raise AssertionError(f"unexpected GitHub POST path: {path}")

    def patch(self, path, json=None):
        raise AssertionError(f"unexpected GitHub PATCH path: {path}")

    def delete(self, path):
        raise AssertionError(f"unexpected GitHub DELETE path: {path}")


class ContextualChecksGithubClient(FakeGithubClient):
    def get(self, path, params=None):
        self.calls.append((path, params))
        if path.endswith("/check-runs"):
            return FakeResponse(
                200,
                {
                    "check_runs": [
                        {
                            "name": "PR Scope Guard",
                            "status": "completed",
                            "conclusion": "success",
                            "started_at": "2026-05-05T16:51:40Z",
                            "html_url": "https://github.com/unarbos/ninja/actions/runs/1001/job/1",
                        },
                        {
                            "name": "OpenRouter PR Judge",
                            "status": "completed",
                            "conclusion": "success",
                            "started_at": "2026-05-05T16:51:48Z",
                            "html_url": "https://github.com/unarbos/ninja/actions/runs/1001/job/2",
                        },
                        {
                            "name": "PR Scope Guard",
                            "status": "completed",
                            "conclusion": "failure",
                            "started_at": "2026-05-06T00:28:22Z",
                            "html_url": "https://github.com/unarbos/ninja/actions/runs/2002/job/1",
                        },
                        {
                            "name": "OpenRouter PR Judge",
                            "status": "completed",
                            "conclusion": "skipped",
                            "started_at": "2026-05-06T00:28:29Z",
                            "html_url": "https://github.com/unarbos/ninja/actions/runs/2002/job/2",
                        },
                    ]
                },
            )
        if path == "/repos/unarbos/ninja/actions/runs/1001":
            return FakeResponse(
                200,
                {
                    "head_sha": SHA,
                    "head_branch": "feature/pr-7",
                    "head_repository": {"full_name": "miner/ninja"},
                },
            )
        if path == "/repos/unarbos/ninja/actions/runs/2002":
            return FakeResponse(
                200,
                {
                    "head_sha": SHA,
                    "head_branch": "main",
                    "head_repository": {"full_name": "goUp9/ninja"},
                },
            )
        return super().get(path, params=params)


class HeadCommitmentGithubClient(FakeGithubClient):
    def get(self, path, params=None):
        if path == "/repos/unarbos/ninja/pulls":
            return FakeResponse(
                200,
                [
                    _open_pr(number=5, title=f"{OTHER_HOTKEY} wrong hotkey", sha=SHA),
                    _open_pr(number=7, title=f"{MINER_HOTKEY} improve harness", sha=SHA),
                ],
            )
        return super().get(path, params=params)


class EmptyPullsGithubClient(FakeGithubClient):
    def get(self, path, params=None):
        if path == "/repos/unarbos/ninja/pulls":
            return FakeResponse(200, [])
        return super().get(path, params=params)


class ConflictResolvingGithubClient(FakeGithubClient):
    def __init__(self, *, temp_merge_failures=None):
        super().__init__()
        self.head_sha = SHA
        self.merge_attempts = 0
        self.temp_merge_attempts = 0
        self.temp_merge_failures = list(temp_merge_failures or [])
        self.updates = []
        self.created_refs = []
        self.deleted_refs = []
        self.temp_branch = None

    def get(self, path, params=None):
        if path == "/repos/unarbos/ninja/pulls/7":
            return FakeResponse(
                200,
                {
                    "number": 7,
                    "state": "open",
                    "draft": False,
                    "title": f"{MINER_HOTKEY} Improve harness",
                    "html_url": "https://github.com/unarbos/ninja/pull/7",
                    "base": {
                        "ref": "main",
                        "sha": BASE_SHA,
                        "repo": {"full_name": "unarbos/ninja"},
                    },
                    "head": {
                        "ref": "feature/conflict",
                        "sha": self.head_sha,
                        "repo": {
                            "full_name": "miner/ninja",
                            "clone_url": "https://github.com/miner/ninja.git",
                        },
                    },
                },
            )
        if path == f"/repos/unarbos/ninja/compare/{BASE_SHA}...{SHA}":
            return FakeResponse(200, {"merge_base_commit": {"sha": ANCESTOR_SHA}})
        if path == "/repos/unarbos/ninja/contents/agent.py":
            ref = (params or {}).get("ref")
            if ref == BASE_SHA:
                return _github_content("def solve(repo_path, issue):\n    return {'patch': ''}\n", "baseblob")
            if ref == ANCESTOR_SHA:
                return _github_content("def solve(repo_path, issue):\n    return {}\n", "ancestorblob")
        if path == "/repos/miner/ninja/contents/agent.py":
            ref = (params or {}).get("ref")
            if ref == SHA:
                return _github_content("def solve(repo_path, issue):\n    return {'patch': 'winner'}\n", "headblob")
        return super().get(path, params=params)

    def put(self, path, json=None):
        if path == "/repos/unarbos/ninja/pulls/7/merge":
            self.merge_attempts += 1
            if self.merge_attempts != 1:
                raise AssertionError("original PR branch must not be retried after conflict resolution")
            return FakeResponse(
                405,
                {"message": "Pull Request has merge conflicts"},
                text='{"message":"Pull Request has merge conflicts","status":"405"}',
            )
        if path == "/repos/unarbos/ninja/contents/agent.py":
            if self.temp_branch is None:
                raise AssertionError("content update happened before temp branch creation")
            content = base64.b64decode(json["content"]).decode("utf-8")
            self.updates.append({"payload": json, "content": content})
            if json["branch"] != self.temp_branch:
                raise AssertionError(f"expected update on temp branch {self.temp_branch}, got {json['branch']}")
            if json["sha"] != "baseblob":
                raise AssertionError(f"expected base blob sha, got {json['sha']}")
            return FakeResponse(200, {"commit": {"sha": RESOLVED_SHA}})
        raise AssertionError(f"unexpected GitHub PUT path: {path}")

    def post(self, path, json=None):
        if path == "/repos/unarbos/ninja/git/refs":
            ref = json["ref"]
            if not ref.startswith(f"refs/heads/validator/resolve-pr-7-{SHA[:12]}-"):
                raise AssertionError(f"unexpected temp ref {ref}")
            if json["sha"] != BASE_SHA:
                raise AssertionError(f"expected temp branch from base head {BASE_SHA}, got {json['sha']}")
            self.temp_branch = ref.removeprefix("refs/heads/")
            self.created_refs.append(json)
            return FakeResponse(201, {"ref": ref, "object": {"sha": BASE_SHA}})
        if path == "/repos/unarbos/ninja/merges":
            self.temp_merge_attempts += 1
            if json["base"] != "main":
                raise AssertionError(f"expected base main, got {json['base']}")
            if json["head"] != self.temp_branch:
                raise AssertionError(f"expected temp branch head {self.temp_branch}, got {json['head']}")
            if self.temp_merge_failures:
                failure = self.temp_merge_failures.pop(0)
                return FakeResponse(
                    failure.get("status_code", 409),
                    failure.get("payload", {"message": "merge failed"}),
                    text=failure.get("text"),
                )
            return FakeResponse(201, {"sha": MERGE_SHA})
        raise AssertionError(f"unexpected GitHub POST path: {path}")

    def delete(self, path):
        if path.startswith("/repos/unarbos/ninja/git/refs/heads/validator/resolve-pr-7-"):
            self.deleted_refs.append(path)
            return FakeResponse(204, {})
        raise AssertionError(f"unexpected GitHub DELETE path: {path}")


class CleanupGithubClient(FakeGithubClient):
    def __init__(self, pulls, *, check_runs=None):
        super().__init__()
        self.pulls = pulls
        self.check_runs = check_runs or {}
        self.labels = []
        self.comments = []
        self.closed = []

    def get(self, path, params=None):
        if path == "/repos/unarbos/ninja/pulls":
            return FakeResponse(200, self.pulls)
        if path.endswith("/check-runs"):
            return FakeResponse(200, {"check_runs": self.check_runs.get(path, [])})
        return super().get(path, params=params)

    def post(self, path, json=None):
        if path == "/repos/unarbos/ninja/labels":
            self.labels.append(json["name"])
            return FakeResponse(422, {"message": "already_exists"})
        if path.startswith("/repos/unarbos/ninja/issues/") and path.endswith("/labels"):
            self.labels.extend(json["labels"])
            return FakeResponse(200, [{"name": label} for label in json["labels"]])
        if path.startswith("/repos/unarbos/ninja/issues/") and path.endswith("/comments"):
            issue_number = int(path.split("/")[5])
            self.comments.append((issue_number, json["body"]))
            return FakeResponse(201, {"id": len(self.comments)})
        return super().post(path, json=json)

    def patch(self, path, json=None):
        if path.startswith("/repos/unarbos/ninja/pulls/"):
            issue_number = int(path.rsplit("/", 1)[1])
            if json != {"state": "closed"}:
                raise AssertionError(f"unexpected close payload: {json}")
            self.closed.append(issue_number)
            return FakeResponse(200, {"number": issue_number, "state": "closed"})
        return super().patch(path, json=json)


def _github_content(text: str, sha: str) -> FakeResponse:
    return FakeResponse(
        200,
        {
            "sha": sha,
            "encoding": "base64",
            "content": base64.b64encode(text.encode("utf-8")).decode("ascii"),
        },
    )


class FakeCommitments:
    def __init__(
        self,
        commitment: str = PR_COMMITMENT,
        *,
        metadata_block: int = 123,
        revealed=None,
    ):
        self.commitment = commitment
        self.metadata_block = metadata_block
        self.revealed = revealed or {}

    def get_all_revealed_commitments(self, netuid):
        return self.revealed

    def get_all_commitments(self, netuid):
        return {MINER_HOTKEY: self.commitment}

    def get_commitment_metadata(self, netuid, hotkey):
        return {"block": self.metadata_block}


class FakeQueryResult:
    def __init__(self, value):
        self.value = value


class FakeSubstrate:
    def __init__(self, registration_blocks_by_uid=None):
        self.registration_blocks_by_uid = registration_blocks_by_uid or {}
        self.queries = []

    def query(self, *, module, storage_function, params, block_hash=None):
        self.queries.append(
            {
                "module": module,
                "storage_function": storage_function,
                "params": params,
                "block_hash": block_hash,
            }
        )
        if module == "SubtensorModule" and storage_function == "BlockAtRegistration":
            _netuid, uid = params
            return FakeQueryResult(self.registration_blocks_by_uid.get(int(uid)))
        raise AssertionError(f"unexpected substrate query: {module}.{storage_function}")


class FakeSubnets:
    def __init__(self, uids_by_hotkey=None):
        self.uids_by_hotkey = uids_by_hotkey or {MINER_HOTKEY: 42}

    def get_uid_for_hotkey_on_subnet(self, hotkey, netuid):
        return self.uids_by_hotkey.get(hotkey)


class FakeSubtensor:
    def __init__(
        self,
        commitment: str = PR_COMMITMENT,
        *,
        metadata_block: int = 123,
        revealed=None,
        block: int = 456,
        registration_block: int | None = None,
    ):
        self.block = block
        self.commitments = FakeCommitments(
            commitment,
            metadata_block=metadata_block,
            revealed=revealed,
        )
        self.subnets = FakeSubnets()
        self.substrate = FakeSubstrate(
            {42: registration_block} if registration_block is not None else {}
        )

    def determine_block_hash(self, block=None):
        return None


class FakeWeightSubtensor:
    def __init__(self, *, allow_hotkey_lookup: bool = False, successes: list[bool] | None = None):
        self.neurons = SimpleNamespace(
            neurons_lite=lambda netuid: [
                SimpleNamespace(uid=0),
                SimpleNamespace(uid=42),
            ],
        )
        self.subnets = SimpleNamespace(
            get_uid_for_hotkey_on_subnet=(
                self._hotkey_lookup if allow_hotkey_lookup else self._unexpected_hotkey_lookup
            ),
        )
        self.extrinsics = SimpleNamespace(set_weights=self._set_weights)
        self.calls = []
        self.successes = list(successes) if successes is not None else [True]

    def _unexpected_hotkey_lookup(self, hotkey, netuid):
        raise AssertionError("burn king weights must not resolve a hotkey")

    def _hotkey_lookup(self, hotkey, netuid):
        if hotkey == MINER_HOTKEY:
            return 42
        return None

    def _set_weights(self, **kwargs):
        self.calls.append(kwargs)
        success = self.successes.pop(0) if self.successes else True
        return SimpleNamespace(success=success)


class GithubPrWatchTest(unittest.TestCase):
    def test_hotkey_spent_since_block_defaults_to_hardcoded_cutoff(self):
        config = RunConfig()

        block = _hotkey_spent_since_block(config)

        self.assertEqual(block, 8_104_340)

    def test_required_checks_pass_with_expected_ci_names(self):
        client = FakeGithubClient()

        self.assertTrue(
            _github_pr_required_checks_passed(
                client,
                base_repo="unarbos/ninja",
                head_repo="miner/ninja",
                sha=SHA,
            )
        )

    def test_required_checks_ignore_same_sha_runs_from_different_pr_head(self):
        client = ContextualChecksGithubClient()

        self.assertTrue(
            _github_pr_required_checks_passed(
                client,
                base_repo="unarbos/ninja",
                head_repo="miner/ninja",
                head_ref="feature/pr-7",
                sha=SHA,
            )
        )

    def test_fetches_pr_commitment_as_real_miner_submission(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_hotkey_spent_since_block=0,
        )

        submissions = _fetch_chain_submissions(
            subtensor=FakeSubtensor(),
            github_client=client,
            config=config,
        )

        self.assertEqual(len(submissions), 1)
        sub = submissions[0]
        self.assertEqual(sub.source, "github_pr")
        self.assertEqual(sub.hotkey, MINER_HOTKEY)
        self.assertEqual(sub.uid, 42)
        self.assertEqual(sub.repo_full_name, "miner/ninja")
        self.assertEqual(sub.repo_url, "https://github.com/miner/ninja.git")
        self.assertEqual(sub.commitment, PR_COMMITMENT)
        self.assertEqual(sub.commitment_block, 123)
        self.assertEqual(sub.pr_number, 7)

    def test_fetches_pre_pr_head_commitment_after_matching_pr_opens(self):
        client = HeadCommitmentGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_hotkey_spent_since_block=0,
        )

        submissions = _fetch_chain_submissions(
            subtensor=FakeSubtensor(PR_HEAD_COMMITMENT),
            github_client=client,
            config=config,
        )

        self.assertEqual(len(submissions), 1)
        sub = submissions[0]
        self.assertEqual(sub.source, "github_pr")
        self.assertEqual(sub.hotkey, MINER_HOTKEY)
        self.assertEqual(sub.uid, 42)
        self.assertEqual(sub.repo_full_name, "miner/ninja")
        self.assertEqual(sub.commit_sha, SHA)
        self.assertEqual(sub.commitment, PR_HEAD_COMMITMENT)
        self.assertEqual(sub.pr_number, 7)
        self.assertEqual(sub.pr_url, "https://github.com/unarbos/ninja/pull/7")

    def test_pre_pr_head_commitment_waits_until_matching_pr_exists(self):
        client = EmptyPullsGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_hotkey_spent_since_block=0,
        )

        submissions = _fetch_chain_submissions(
            subtensor=FakeSubtensor(PR_HEAD_COMMITMENT),
            github_client=client,
            config=config,
        )

        self.assertEqual(submissions, [])

    def test_empty_king_stays_unset_without_consuming_queue(self):
        client = FakeGithubClient()
        config = RunConfig(validate_github_pr_repo="unarbos/ninja", validate_github_pr_base="main")
        state = ValidatorState(queue=[_submission(commitment=PR_COMMITMENT, sha=SHA, block=123)])

        _ensure_king(state=state, github_client=client, config=config)

        self.assertIsNone(state.current_king)
        self.assertEqual(len(state.queue), 1)

    def test_burn_king_weights_target_uid_zero_directly(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_wallet_name="wallet",
            validate_wallet_hotkey="hotkey",
        )
        state = ValidatorState(current_king=_build_burn_king(github_client=client, config=config))
        subtensor = FakeWeightSubtensor()

        with patch("validate.bt.Wallet", return_value=object()):
            _maybe_set_weights(subtensor=subtensor, config=config, state=state, current_block=100)

        self.assertEqual(len(subtensor.calls), 1)
        self.assertEqual(subtensor.calls[0]["uids"], [0, 42])
        self.assertEqual(subtensor.calls[0]["weights"], [1.0, 0.0])
        self.assertEqual(state.last_weight_block, 100)

    def test_weight_force_bypasses_interval_after_new_king(self):
        config = RunConfig(
            validate_wallet_name="wallet",
            validate_wallet_hotkey="hotkey",
        )
        king = _submission(commitment=PR_COMMITMENT, sha=SHA, block=123)
        state = ValidatorState(
            current_king=king,
            recent_kings=[king],
            last_weight_block=95,
        )
        subtensor = FakeWeightSubtensor(allow_hotkey_lookup=True)

        with patch("validate.bt.Wallet", return_value=object()):
            _maybe_set_weights(
                subtensor=subtensor,
                config=config,
                state=state,
                current_block=100,
                force=True,
            )

        self.assertEqual(len(subtensor.calls), 1)
        self.assertEqual(subtensor.calls[0]["uids"], [0, 42])
        self.assertEqual(subtensor.calls[0]["weights"], [0.8, 0.2])
        self.assertEqual(state.last_weight_block, 100)

    def test_weight_failure_retries_immediately_and_only_advances_on_success(self):
        config = RunConfig(
            validate_wallet_name="wallet",
            validate_wallet_hotkey="hotkey",
        )
        king = _submission(commitment=PR_COMMITMENT, sha=SHA, block=123)
        state = ValidatorState(
            current_king=king,
            recent_kings=[king],
        )
        subtensor = FakeWeightSubtensor(allow_hotkey_lookup=True, successes=[False, False, True])

        with patch("validate.bt.Wallet", return_value=object()):
            _maybe_set_weights(
                subtensor=subtensor,
                config=config,
                state=state,
                current_block=100,
                force=True,
            )

        self.assertEqual(len(subtensor.calls), 3)
        self.assertEqual(state.last_weight_block, 100)

    def test_weight_total_failure_does_not_advance_last_weight_block(self):
        config = RunConfig(
            validate_wallet_name="wallet",
            validate_wallet_hotkey="hotkey",
        )
        king = _submission(commitment=PR_COMMITMENT, sha=SHA, block=123)
        state = ValidatorState(
            current_king=king,
            recent_kings=[king],
            last_weight_block=95,
        )
        subtensor = FakeWeightSubtensor(allow_hotkey_lookup=True, successes=[False, False, False])

        with patch("validate.bt.Wallet", return_value=object()):
            _maybe_set_weights(
                subtensor=subtensor,
                config=config,
                state=state,
                current_block=100,
                force=True,
            )

        self.assertEqual(len(subtensor.calls), 3)
        self.assertEqual(state.last_weight_block, 95)

    def test_dashboard_displays_winning_pr_repo_for_merged_king(self):
        submission = ValidatorSubmission(
            hotkey=MINER_HOTKEY,
            uid=42,
            repo_full_name="unarbos/ninja",
            repo_url="https://github.com/unarbos/ninja.git",
            commit_sha=MERGE_SHA,
            commitment=PR_COMMITMENT,
            commitment_block=123,
            source="github_pr_merged",
            pr_number=7,
            pr_url=None,
        )
        history = [
            {
                "king_replaced": True,
                "challenger_hotkey": MINER_HOTKEY,
                "challenger_uid": 42,
                "challenger_repo": "miner/ninja",
                "challenger_repo_url": "https://github.com/miner/ninja",
                "challenger_pr_url": "https://github.com/unarbos/ninja/pull/7",
                "challenger_commit_sha": SHA,
            }
        ]

        payload = _dashboard_submission_dict(submission, history=history)

        self.assertEqual(payload["repo"], "miner/ninja")
        self.assertEqual(payload["repo_full_name"], "miner/ninja")
        self.assertEqual(payload["repo_url"], "https://github.com/miner/ninja")
        self.assertEqual(payload["pr_url"], "https://github.com/unarbos/ninja/pull/7")
        self.assertEqual(payload["commit_sha"], SHA)
        self.assertEqual(payload["runtime_repo_full_name"], "unarbos/ninja")
        self.assertEqual(payload["runtime_commit_sha"], MERGE_SHA)

    def test_pr_title_must_start_with_committing_miner_hotkey_before_ci_checks(self):
        client = FakeGithubClient(title=f"Improve harness {MINER_HOTKEY}")
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_hotkey_spent_since_block=0,
        )

        submissions = _fetch_chain_submissions(
            subtensor=FakeSubtensor(),
            github_client=client,
            config=config,
        )

        self.assertEqual(submissions, [])
        check_run_calls = [path for path, _ in client.calls if path.endswith("/check-runs")]
        self.assertEqual(check_run_calls, [])

    def test_pr_only_mode_skips_normal_commitments(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_only=True,
            validate_hotkey_spent_since_block=0,
        )

        submissions = _fetch_chain_submissions(
            subtensor=FakeSubtensor("unarbos/ninja@" + SHA),
            github_client=client,
            config=config,
        )

        self.assertEqual(submissions, [])

    def test_pr_only_mode_makes_raw_submission_ineligible_even_from_ninja_repo(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_only=True,
            validate_hotkey_spent_since_block=0,
        )
        submission = ValidatorSubmission(
            hotkey=MINER_HOTKEY,
            uid=42,
            repo_full_name="unarbos/ninja",
            repo_url="https://github.com/unarbos/ninja.git",
            commit_sha=SHA,
            commitment=f"unarbos/ninja@{SHA}",
            commitment_block=123,
        )

        self.assertFalse(
            _submission_is_eligible(
                subtensor=FakeSubtensor(),
                github_client=client,
                config=config,
                submission=submission,
            )
        )

    def test_pr_only_mode_removes_restored_raw_kings_queue_and_window(self):
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_only=True,
            validate_hotkey_spent_since_block=0,
        )
        raw_king = ValidatorSubmission(
            hotkey=OTHER_HOTKEY,
            uid=43,
            repo_full_name="unarbos/ninja",
            repo_url="https://github.com/unarbos/ninja.git",
            commit_sha=SHA,
            commitment=f"unarbos/ninja@{SHA}",
            commitment_block=123,
        )
        pr_king = _github_pr_submission()
        raw_queue_hotkey = "5HrawQueueHotkeyForGithubPrOnlyModeTest111111111"
        raw_queue = ValidatorSubmission(
            hotkey=raw_queue_hotkey,
            uid=44,
            repo_full_name="unarbos/ninja",
            repo_url="https://github.com/unarbos/ninja.git",
            commit_sha=BASE_SHA,
            commitment=f"unarbos/ninja@{BASE_SHA}",
            commitment_block=124,
        )
        state = ValidatorState(
            current_king=raw_king,
            queue=[raw_queue, pr_king],
            recent_kings=[raw_king, pr_king],
        )

        changed = _enforce_submission_mode_on_state(config, state)

        self.assertTrue(changed)
        self.assertIsNone(state.current_king)
        self.assertEqual(state.queue, [pr_king])
        self.assertEqual(state.recent_kings, [pr_king])
        self.assertIn(OTHER_HOTKEY, state.disqualified_hotkeys)
        self.assertIn(raw_queue_hotkey, state.disqualified_hotkeys)

    def test_pr_only_mode_keeps_promoted_github_pr_king(self):
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_only=True,
            validate_hotkey_spent_since_block=0,
        )
        promoted = _github_pr_submission()
        promoted.source = "github_pr_merged"
        promoted.repo_full_name = "unarbos/ninja"
        promoted.commit_sha = MERGE_SHA
        state = ValidatorState(current_king=promoted, recent_kings=[promoted])

        changed = _enforce_submission_mode_on_state(config, state)

        self.assertFalse(changed)
        self.assertEqual(state.current_king, promoted)
        self.assertEqual(state.recent_kings, [promoted])
        self.assertEqual(state.disqualified_hotkeys, [])

    def test_pr_only_mode_validates_promoted_pr_king_as_base_repo_commit(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_only=True,
            validate_hotkey_spent_since_block=0,
        )
        promoted = _github_pr_submission()
        promoted.source = "github_pr_merged"
        promoted.repo_full_name = "unarbos/ninja"
        promoted.repo_url = "https://github.com/unarbos/ninja.git"
        promoted.commit_sha = BASE_SHA

        self.assertTrue(
            _submission_is_eligible(
                subtensor=FakeSubtensor(),
                github_client=client,
                config=config,
                submission=promoted,
            )
        )

    def test_pr_only_mode_does_not_weight_raw_recent_king(self):
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_only=True,
            validate_king_window_size=1,
            validate_hotkey_spent_since_block=0,
        )
        raw_king = ValidatorSubmission(
            hotkey=MINER_HOTKEY,
            uid=42,
            repo_full_name="unarbos/ninja",
            repo_url="https://github.com/unarbos/ninja.git",
            commit_sha=SHA,
            commitment=f"unarbos/ninja@{SHA}",
            commitment_block=123,
        )
        state = ValidatorState(recent_kings=[raw_king])
        subtensor = FakeWeightSubtensor(allow_hotkey_lookup=True)

        _maybe_set_weights(
            subtensor=subtensor,
            config=config,
            state=state,
            current_block=100,
            force=True,
        )

        self.assertEqual(len(subtensor.calls), 1)
        self.assertEqual(subtensor.calls[0]["uids"], [0, 42])
        self.assertEqual(subtensor.calls[0]["weights"], [1.0, 0.0])

    def test_pr_submission_eligibility_rechecks_chain_uid_and_title(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_hotkey_spent_since_block=0,
        )
        submission = _fetch_chain_submissions(
            subtensor=FakeSubtensor(),
            github_client=client,
            config=config,
        )[0]

        self.assertTrue(
            _submission_is_eligible(
                subtensor=FakeSubtensor(),
                github_client=client,
                config=config,
                submission=submission,
            )
        )

    def test_pr_submission_eligibility_rejects_pre_registration_commitment(self):
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_hotkey_spent_since_block=0,
        )
        submission = _github_pr_submission()
        submission.commitment_block = 100

        self.assertFalse(
            _submission_is_eligible(
                subtensor=FakeSubtensor(registration_block=200),
                github_client=FakeGithubClient(),
                config=config,
                submission=submission,
            )
        )

    def test_refresh_queue_rejects_second_commitment_from_seen_hotkey(self):
        config = RunConfig(validate_hotkey_spent_since_block=0)
        state = ValidatorState()
        first = _submission(commitment="repo@a", sha="a" * 40, block=100)
        second = _submission(commitment="repo@b", sha="b" * 40, block=101)

        _refresh_queue(chain_submissions=[first], config=config, state=state)
        state.queue.clear()

        _refresh_queue(chain_submissions=[second], config=config, state=state)

        self.assertEqual(state.queue, [])
        self.assertEqual(state.locked_commitments[MINER_HOTKEY], first.commitment)
        self.assertEqual(state.commitment_blocks_by_hotkey[MINER_HOTKEY], first.commitment_block)
        self.assertEqual(state.seen_hotkeys, [MINER_HOTKEY])

    def test_refresh_queue_rejects_second_commitment_even_at_later_block(self):
        config = RunConfig(validate_hotkey_spent_since_block=0)
        state = ValidatorState()
        first = _submission(commitment="repo@a", sha="a" * 40, block=100)
        second = _submission(commitment="repo@b", sha="b" * 40, block=100_000)

        _refresh_queue(chain_submissions=[first], config=config, state=state)
        state.queue.clear()

        _refresh_queue(chain_submissions=[second], config=config, state=state)

        self.assertEqual(state.queue, [])
        self.assertEqual(state.locked_commitments[MINER_HOTKEY], first.commitment)
        self.assertEqual(state.commitment_blocks_by_hotkey[MINER_HOTKEY], first.commitment_block)

    def test_refresh_queue_allows_same_hotkey_after_new_registration_block(self):
        config = RunConfig(validate_hotkey_spent_since_block=0)
        state = ValidatorState()
        first = _submission(commitment="repo@a", sha="a" * 40, block=100)
        second = _submission(commitment="repo@b", sha="b" * 40, block=300)

        _refresh_queue(chain_submissions=[first], config=config, state=state)
        state.queue.clear()

        _refresh_queue(
            chain_submissions=[second],
            config=config,
            state=state,
            subtensor=FakeSubtensor(registration_block=200),
        )

        self.assertEqual(state.queue, [second])
        self.assertEqual(state.locked_commitments[MINER_HOTKEY], second.commitment)
        self.assertEqual(state.commitment_blocks_by_hotkey[MINER_HOTKEY], second.commitment_block)
        self.assertEqual(state.seen_hotkeys, [MINER_HOTKEY])

    def test_new_eligible_commitment_does_not_replace_queued_candidate(self):
        config = RunConfig(validate_hotkey_spent_since_block=0)
        state = ValidatorState()
        first = _submission(commitment="repo@a", sha="a" * 40, block=100)
        second = _submission(commitment="repo@b", sha="b" * 40, block=100_000)

        _refresh_queue(chain_submissions=[first], config=config, state=state)
        _refresh_queue(chain_submissions=[second], config=config, state=state)

        self.assertEqual(state.queue, [first])
        self.assertEqual(state.locked_commitments[MINER_HOTKEY], first.commitment)

    def test_fetch_chain_submissions_skips_hotkey_that_already_submitted(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_hotkey_spent_since_block=0,
        )
        state = ValidatorState()
        _refresh_queue(
            chain_submissions=[_submission(commitment=PR_COMMITMENT, sha=SHA, block=123)],
            config=config,
            state=state,
        )

        submissions = _fetch_chain_submissions(
            subtensor=FakeSubtensor(f"github-pr:unarbos/ninja#8@{'b' * 40}"),
            github_client=client,
            config=config,
            state=state,
        )

        self.assertEqual(submissions, [])

    def test_fetch_chain_submissions_allows_hotkey_spent_only_before_cutoff(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_hotkey_spent_since_block=200,
        )
        state = ValidatorState(
            seen_hotkeys=[MINER_HOTKEY],
            locked_commitments={MINER_HOTKEY: "unarbos/ninja@" + "b" * 40},
            commitment_blocks_by_hotkey={MINER_HOTKEY: 100},
        )

        submissions = _fetch_chain_submissions(
            subtensor=FakeSubtensor(PR_COMMITMENT, metadata_block=300),
            github_client=client,
            config=config,
            state=state,
        )

        self.assertEqual(len(submissions), 1)
        self.assertEqual(submissions[0].commitment, PR_COMMITMENT)
        self.assertEqual(submissions[0].commitment_block, 300)

    def test_fetch_chain_submissions_allows_hotkey_spent_before_registration_block(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_hotkey_spent_since_block=0,
        )
        state = ValidatorState(
            seen_hotkeys=[MINER_HOTKEY],
            locked_commitments={MINER_HOTKEY: "unarbos/ninja@" + "b" * 40},
            commitment_blocks_by_hotkey={MINER_HOTKEY: 100},
        )

        submissions = _fetch_chain_submissions(
            subtensor=FakeSubtensor(PR_COMMITMENT, metadata_block=300, registration_block=200),
            github_client=client,
            config=config,
            state=state,
        )

        self.assertEqual(len(submissions), 1)
        self.assertEqual(submissions[0].commitment, PR_COMMITMENT)
        self.assertEqual(submissions[0].commitment_block, 300)

    def test_fetch_chain_submissions_ignores_revealed_history_before_registration_block(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_hotkey_spent_since_block=0,
        )
        state = ValidatorState(
            seen_hotkeys=[MINER_HOTKEY],
            locked_commitments={MINER_HOTKEY: "unarbos/ninja@" + "b" * 40},
            commitment_blocks_by_hotkey={MINER_HOTKEY: 250},
        )

        submissions = _fetch_chain_submissions(
            subtensor=FakeSubtensor(
                PR_COMMITMENT,
                metadata_block=350,
                revealed={MINER_HOTKEY: [(250, "unarbos/ninja@" + "b" * 40)]},
                registration_block=300,
            ),
            github_client=client,
            config=config,
            state=state,
        )

        self.assertEqual(len(submissions), 1)
        self.assertEqual(submissions[0].commitment, PR_COMMITMENT)
        self.assertEqual(state.locked_commitments[MINER_HOTKEY], "unarbos/ninja@" + "b" * 40)

    def test_fetch_chain_submissions_treats_prior_chain_commitment_since_cutoff_as_spent(self):
        client = FakeGithubClient()
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_only=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_hotkey_spent_since_block=200,
        )
        state = ValidatorState()

        submissions = _fetch_chain_submissions(
            subtensor=FakeSubtensor(
                PR_COMMITMENT,
                metadata_block=300,
                revealed={MINER_HOTKEY: [(250, "unarbos/ninja@" + "b" * 40)]},
            ),
            github_client=client,
            config=config,
            state=state,
        )

        self.assertEqual(submissions, [])
        self.assertEqual(client.calls, [])
        self.assertEqual(state.locked_commitments[MINER_HOTKEY], "unarbos/ninja@" + "b" * 40)
        self.assertEqual(state.commitment_blocks_by_hotkey[MINER_HOTKEY], 250)

    def test_state_load_seeds_seen_hotkeys_from_locked_commitments(self):
        state = ValidatorState.from_dict(
            {
                "locked_commitments": {MINER_HOTKEY: PR_COMMITMENT},
                "commitment_blocks_by_hotkey": {MINER_HOTKEY: 123},
            }
        )

        self.assertEqual(state.seen_hotkeys, [MINER_HOTKEY])

    def test_state_load_seeds_seen_hotkeys_from_recent_kings(self):
        state = ValidatorState.from_dict(
            {"recent_kings": [_submission(commitment=PR_COMMITMENT, sha=SHA, block=123).to_dict()]}
        )

        self.assertEqual(state.seen_hotkeys, [MINER_HOTKEY])

    def test_cleanup_closes_pr_from_spent_hotkey_with_reason_label(self):
        state = ValidatorState(locked_commitments={MINER_HOTKEY: PR_COMMITMENT})
        client = CleanupGithubClient([
            _open_pr(number=8, title=f"{MINER_HOTKEY} another attempt", sha="b" * 40),
        ])
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_cleanup=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_hotkey_spent_since_block=0,
        )

        closed = _cleanup_stale_github_prs(github_client=client, config=config, state=state)

        self.assertEqual(closed, 1)
        self.assertEqual(client.closed, [8])
        self.assertIn("close: hotkey-spent", client.labels)
        self.assertIn("one lifetime", client.comments[0][1])

    def test_cleanup_does_not_close_hotkey_spent_for_pre_cutoff_state(self):
        state = ValidatorState(
            locked_commitments={MINER_HOTKEY: PR_COMMITMENT},
            commitment_blocks_by_hotkey={MINER_HOTKEY: 100},
        )
        client = CleanupGithubClient([
            _open_pr(number=8, title=f"{MINER_HOTKEY} another attempt", sha="b" * 40),
        ])
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_cleanup=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_require_checks=False,
            validate_github_pr_cleanup_stale_after_hours=-1,
            validate_hotkey_spent_since_block=200,
        )

        closed = _cleanup_stale_github_prs(github_client=client, config=config, state=state)

        self.assertEqual(closed, 0)
        self.assertEqual(client.closed, [])

    def test_cleanup_does_not_close_hotkey_spent_for_pre_registration_state(self):
        state = ValidatorState(
            locked_commitments={MINER_HOTKEY: PR_COMMITMENT},
            commitment_blocks_by_hotkey={MINER_HOTKEY: 100},
        )
        client = CleanupGithubClient([
            _open_pr(number=8, title=f"{MINER_HOTKEY} another attempt", sha="b" * 40),
        ])
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_cleanup=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_require_checks=False,
            validate_github_pr_cleanup_stale_after_hours=-1,
            validate_hotkey_spent_since_block=0,
        )

        closed = _cleanup_stale_github_prs(
            github_client=client,
            config=config,
            state=state,
            subtensor=FakeSubtensor(registration_block=200),
        )

        self.assertEqual(closed, 0)
        self.assertEqual(client.closed, [])

    def test_cleanup_keeps_live_queued_pr_open(self):
        state = ValidatorState(queue=[_github_pr_submission()])
        client = CleanupGithubClient([
            _open_pr(number=7, title=f"{MINER_HOTKEY} improve harness", sha=SHA),
        ])
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_cleanup=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_cleanup_stale_after_hours=0,
            validate_hotkey_spent_since_block=0,
        )

        closed = _cleanup_stale_github_prs(github_client=client, config=config, state=state)

        self.assertEqual(closed, 0)
        self.assertEqual(client.closed, [])

    def test_cleanup_closes_open_pr_that_was_already_promoted(self):
        promoted = _github_pr_submission()
        promoted.source = "github_pr_merged"
        promoted.repo_full_name = "unarbos/ninja"
        promoted.commit_sha = MERGE_SHA
        state = ValidatorState(current_king=promoted)
        client = CleanupGithubClient([
            _open_pr(number=7, title=f"{MINER_HOTKEY} promoted", sha="b" * 40),
        ])
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_cleanup=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_hotkey_spent_since_block=0,
        )

        closed = _cleanup_stale_github_prs(github_client=client, config=config, state=state)

        self.assertEqual(closed, 1)
        self.assertEqual(client.closed, [7])
        self.assertIn("close: promoted-king", client.labels)

    def test_cleanup_closes_old_unqueued_pr_as_stale_submission(self):
        state = ValidatorState()
        client = CleanupGithubClient([
            _open_pr(number=9, title=f"{OTHER_HOTKEY} old attempt", sha="b" * 40),
        ])
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_cleanup=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_cleanup_stale_after_hours=1,
        )

        closed = _cleanup_stale_github_prs(github_client=client, config=config, state=state)

        self.assertEqual(closed, 1)
        self.assertEqual(client.closed, [9])
        self.assertIn("close: stale-submission", client.labels)

    def test_cleanup_comments_on_old_pr_without_matching_commitment(self):
        sha = "b" * 40
        state = ValidatorState()
        client = CleanupGithubClient([
            _open_pr(number=9, title=f"{OTHER_HOTKEY} old attempt", sha=sha),
        ])
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_cleanup=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_require_checks=False,
            validate_github_pr_cleanup_stale_after_hours=999999,
            validate_github_pr_missing_commitment_notice_after_minutes=30,
        )

        closed = _cleanup_stale_github_prs(github_client=client, config=config, state=state)

        self.assertEqual(closed, 0)
        self.assertEqual(client.closed, [])
        self.assertIn("notice: missing-commitment", client.labels)
        self.assertEqual(client.comments[0][0], 9)
        self.assertIn("No posted commitment with the hotkey in the title", client.comments[0][1])
        self.assertIn(f"github-pr:unarbos/ninja#9@{sha}", client.comments[0][1])

    def test_cleanup_does_not_repeat_missing_commitment_notice(self):
        state = ValidatorState()
        client = CleanupGithubClient([
            _open_pr(
                number=9,
                title=f"{OTHER_HOTKEY} old attempt",
                sha="b" * 40,
                labels=[{"name": "notice: missing-commitment"}],
            ),
        ])
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_cleanup=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_require_checks=False,
            validate_github_pr_cleanup_stale_after_hours=999999,
            validate_github_pr_missing_commitment_notice_after_minutes=30,
        )

        closed = _cleanup_stale_github_prs(github_client=client, config=config, state=state)

        self.assertEqual(closed, 0)
        self.assertEqual(client.closed, [])
        self.assertEqual(client.comments, [])

    def test_cleanup_does_not_comment_when_title_hotkey_committed_pr_head(self):
        sha = "b" * 40
        state = ValidatorState(locked_commitments={OTHER_HOTKEY: f"github-pr:unarbos/ninja#9@{sha[:12]}"})
        client = CleanupGithubClient([
            _open_pr(number=9, title=f"{OTHER_HOTKEY} old attempt", sha=sha),
        ])
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_cleanup=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_require_checks=False,
            validate_github_pr_cleanup_stale_after_hours=999999,
            validate_github_pr_missing_commitment_notice_after_minutes=30,
            validate_hotkey_spent_since_block=200,
        )

        closed = _cleanup_stale_github_prs(github_client=client, config=config, state=state)

        self.assertEqual(closed, 0)
        self.assertEqual(client.closed, [])
        self.assertEqual(client.comments, [])

    def test_cleanup_does_not_comment_when_title_hotkey_precommitted_head(self):
        sha = "b" * 40
        state = ValidatorState(locked_commitments={OTHER_HOTKEY: f"github-pr-head:unarbos/ninja@{sha[:12]}"})
        client = CleanupGithubClient([
            _open_pr(number=9, title=f"{OTHER_HOTKEY} old attempt", sha=sha),
        ])
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_cleanup=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_require_checks=False,
            validate_github_pr_cleanup_stale_after_hours=999999,
            validate_github_pr_missing_commitment_notice_after_minutes=30,
            validate_hotkey_spent_since_block=200,
        )

        closed = _cleanup_stale_github_prs(github_client=client, config=config, state=state)

        self.assertEqual(closed, 0)
        self.assertEqual(client.closed, [])
        self.assertEqual(client.comments, [])

    def test_cleanup_closes_failed_scope_guard_pr(self):
        sha = "f" * 40
        check_runs = [
            {"name": "PR Scope Guard", "status": "completed", "conclusion": "failure"},
            {"name": "OpenRouter PR Judge", "status": "completed", "conclusion": "success"},
        ]
        client = CleanupGithubClient(
            [_open_pr(number=10, title=f"{OTHER_HOTKEY} invalid edit", sha=sha, created_at="2026-05-05T00:00:00Z")],
            check_runs={
                f"/repos/unarbos/ninja/commits/{sha}/check-runs": check_runs,
                f"/repos/miner/ninja/commits/{sha}/check-runs": check_runs,
            },
        )
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_cleanup=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_cleanup_stale_after_hours=999,
        )

        closed = _cleanup_stale_github_prs(github_client=client, config=config, state=ValidatorState())

        self.assertEqual(closed, 1)
        self.assertEqual(client.closed, [10])
        self.assertIn("close: failed-test", client.labels)

    def test_cleanup_closes_failed_judge_pr_as_inadequate(self):
        sha = "e" * 40
        check_runs = [
            {"name": "PR Scope Guard", "status": "completed", "conclusion": "success"},
            {"name": "OpenRouter PR Judge", "status": "completed", "conclusion": "failure"},
        ]
        client = CleanupGithubClient(
            [_open_pr(number=11, title=f"{OTHER_HOTKEY} low score", sha=sha, created_at="2026-05-05T00:00:00Z")],
            check_runs={
                f"/repos/unarbos/ninja/commits/{sha}/check-runs": check_runs,
                f"/repos/miner/ninja/commits/{sha}/check-runs": check_runs,
            },
        )
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_cleanup=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_github_pr_cleanup_stale_after_hours=999,
        )

        closed = _cleanup_stale_github_prs(github_client=client, config=config, state=ValidatorState())

        self.assertEqual(closed, 1)
        self.assertEqual(client.closed, [11])
        self.assertIn("close: passed-test-inadequate", client.labels)

    def test_cleanup_closes_stale_head_for_committed_pr(self):
        state = ValidatorState(locked_commitments={MINER_HOTKEY: PR_COMMITMENT})
        client = CleanupGithubClient([
            _open_pr(number=7, title=f"{MINER_HOTKEY} moved head", sha="b" * 40),
        ])
        config = RunConfig(
            validate_github_pr_watch=True,
            validate_github_pr_cleanup=True,
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
            validate_hotkey_spent_since_block=0,
        )

        closed = _cleanup_stale_github_prs(github_client=client, config=config, state=state)

        self.assertEqual(closed, 1)
        self.assertEqual(client.closed, [7])
        self.assertIn("close: stale-head", client.labels)

    def test_promoted_pr_merge_conflict_uses_llm_resolver_then_retries(self):
        client = ConflictResolvingGithubClient(
            temp_merge_failures=[
                {
                    "status_code": 409,
                    "payload": {"message": "merge conflict in agent.py near solve()"},
                    "text": '{"message":"merge conflict in agent.py near solve()"}',
                }
            ]
        )
        config = RunConfig(
            openrouter_api_key="or-key",
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
        )
        first_resolved_agent = (
            "def solve(repo_path, issue, model=None, api_base=None, api_key=None):\n"
            "    return {'patch': 'first', 'logs': '', 'steps': 0, 'cost': None, 'success': True}\n"
        )
        second_resolved_agent = (
            "def solve(repo_path, issue, model=None, api_base=None, api_key=None):\n"
            "    return {'patch': 'second', 'logs': '', 'steps': 1, 'cost': None, 'success': True}\n"
        )

        with patch(
            "validate.complete_text",
            side_effect=[
                f"<resolved_agent_py>\n{first_resolved_agent}</resolved_agent_py>",
                f"<resolved_agent_py>\n{second_resolved_agent}</resolved_agent_py>",
            ],
        ) as complete:
            merged = _merge_promoted_github_pr(
                github_client=client,
                config=config,
                submission=_github_pr_submission(),
            )

        self.assertEqual(merged.source, "github_pr_merged")
        self.assertEqual(merged.repo_full_name, "unarbos/ninja")
        self.assertEqual(merged.commit_sha, MERGE_SHA)
        self.assertEqual(client.merge_attempts, 1)
        self.assertEqual(client.temp_merge_attempts, 2)
        self.assertEqual(len(client.created_refs), 2)
        self.assertEqual(len(client.deleted_refs), 2)
        self.assertEqual(len(client.updates), 2)
        self.assertEqual(client.updates[0]["content"], first_resolved_agent)
        self.assertEqual(client.updates[1]["content"], second_resolved_agent)
        self.assertEqual(complete.call_count, 2)
        self.assertEqual(complete.call_args_list[0].kwargs["model"], "anthropic/claude-opus-4.7")
        self.assertIn("merge conflict in agent.py near solve()", complete.call_args_list[1].kwargs["prompt"])

    def test_promoted_pr_merge_conflict_rejects_invalid_llm_resolution(self):
        client = ConflictResolvingGithubClient()
        config = RunConfig(
            openrouter_api_key="or-key",
            validate_github_pr_repo="unarbos/ninja",
            validate_github_pr_base="main",
        )
        original = _github_pr_submission()

        with patch("validate.complete_text", return_value="<resolved_agent_py>\ndef nope(\n</resolved_agent_py>"):
            merged = _merge_promoted_github_pr(
                github_client=client,
                config=config,
                submission=original,
            )

        self.assertEqual(merged.source, "github_pr")
        self.assertEqual(merged.commit_sha, SHA)
        self.assertEqual(client.merge_attempts, 1)
        self.assertEqual(client.temp_merge_attempts, 0)
        self.assertEqual(client.created_refs, [])
        self.assertEqual(client.updates, [])


def _submission(*, commitment: str, sha: str, block: int) -> ValidatorSubmission:
    return ValidatorSubmission(
        hotkey=MINER_HOTKEY,
        uid=42,
        repo_full_name="miner/ninja",
        repo_url="https://github.com/miner/ninja.git",
        commit_sha=sha,
        commitment=commitment,
        commitment_block=block,
    )


def _github_pr_submission() -> ValidatorSubmission:
    return ValidatorSubmission(
        hotkey=MINER_HOTKEY,
        uid=42,
        repo_full_name="miner/ninja",
        repo_url="https://github.com/miner/ninja.git",
        commit_sha=SHA,
        commitment=PR_COMMITMENT,
        commitment_block=123,
        source="github_pr",
        pr_number=7,
        pr_url="https://github.com/unarbos/ninja/pull/7",
        base_repo_full_name="unarbos/ninja",
        base_ref="main",
    )


def _open_pr(
    *,
    number: int,
    title: str,
    sha: str,
    created_at: str = "2020-01-01T00:00:00Z",
    base_ref: str = "main",
    base_repo: str = "unarbos/ninja",
    head_repo: str = "miner/ninja",
    labels: list[dict] | None = None,
) -> dict:
    return {
        "number": number,
        "state": "open",
        "draft": False,
        "title": title,
        "created_at": created_at,
        "labels": labels or [],
        "html_url": f"https://github.com/{base_repo}/pull/{number}",
        "base": {
            "ref": base_ref,
            "repo": {"full_name": base_repo},
        },
        "head": {
            "sha": sha,
            "repo": {
                "full_name": head_repo,
                "clone_url": f"https://github.com/{head_repo}.git",
            },
        },
    }


if __name__ == "__main__":
    unittest.main()
