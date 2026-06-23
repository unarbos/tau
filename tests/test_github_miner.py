import random
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from github_miner import (
    CommitRejectCache,
    GitHubMiner,
    GitHubTokenRotator,
    _commit_search_query,
    _extend_date_commit_buffer,
    _pop_date_commit,
    clear_recent_events_cache,
    first_symlink_tree_path,
    reset_commit_search_buffer_for_tests,
    stop_commit_search_refiller_for_tests,
)


class FakeGitHubClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers_seen = []

    def close(self) -> None:
        return None

    def get(self, path, params=None, headers=None):
        self.headers_seen.append(dict(headers or {}))
        response = self.responses.pop(0)
        response.request = httpx.Request("GET", "https://api.github.com" + path)
        return response


class GitHubTokenRotatorTest(unittest.TestCase):
    def tearDown(self):
        clear_recent_events_cache()
        reset_commit_search_buffer_for_tests()
        stop_commit_search_refiller_for_tests()

    def test_401_disables_token_and_retries_next_token(self):
        rotator = GitHubTokenRotator(["bad-token", "good-token"])
        miner = GitHubMiner(
            token_rotator=rotator,
            rng=random.Random(1),
            start_search_refiller=False,
        )
        miner._client = FakeGitHubClient([
            httpx.Response(401, json={"message": "Bad credentials"}),
            httpx.Response(200, json={"ok": True}),
        ])

        self.assertEqual(miner._get_json("/events"), {"ok": True})
        self.assertEqual(rotator.active_count, 1)
        self.assertEqual(
            miner._client.headers_seen,
            [
                {"Authorization": "Bearer bad-token"},
                {"Authorization": "Bearer good-token"},
            ],
        )
        miner.close()

    def test_all_401_tokens_fall_back_to_unauthenticated_request(self):
        rotator = GitHubTokenRotator(["bad-token"])
        miner = GitHubMiner(
            token_rotator=rotator,
            rng=random.Random(1),
            start_search_refiller=False,
        )
        miner._client = FakeGitHubClient([
            httpx.Response(401, json={"message": "Bad credentials"}),
            httpx.Response(200, json={"ok": True}),
        ])

        self.assertEqual(miner._get_json("/events"), {"ok": True})
        self.assertEqual(rotator.active_count, 0)
        self.assertEqual(
            miner._client.headers_seen,
            [
                {"Authorization": "Bearer bad-token"},
                {},
            ],
        )
        miner.close()

    def test_commit_search_query_includes_merge_false_by_default(self):
        self.assertEqual(_commit_search_query("2026-01-02"), "committer-date:2026-01-02 merge:false")

    def test_commit_search_buffer_is_pop_only_for_workers(self):
        _extend_date_commit_buffer([("owner/a", "sha-a"), ("owner/b", "sha-b")])
        miner = GitHubMiner(rng=random.Random(1), start_search_refiller=False)
        search_calls = {"count": 0}

        def fake_search():
            search_calls["count"] += 1
            return []

        miner._search_commits_for_random_day = fake_search  # type: ignore[method-assign]
        self.assertEqual(miner._sample_commit_by_random_day(), ("owner/b", "sha-b"))
        self.assertEqual(miner._sample_commit_by_random_day(), ("owner/a", "sha-a"))
        self.assertIsNone(miner._sample_commit_by_random_day())
        self.assertEqual(search_calls["count"], 0)
        miner.close()

    def test_reject_cache_skips_cached_commits_without_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "rejects.jsonl"
            cache = CommitRejectCache(cache_path)
            cache.add("owner/repo", "bad-sha", "too small")
            _extend_date_commit_buffer([("owner/repo", "bad-sha"), ("owner/repo", "good-sha")])
            miner = GitHubMiner(
                rng=random.Random(1),
                reject_cache_path=cache_path,
                start_search_refiller=False,
            )
            fetch_calls = {"count": 0}

            def fake_fetch(**kwargs):
                fetch_calls["count"] += 1
                from github_miner import CommitCandidate, CommitFile

                return CommitCandidate(
                    repo_full_name=kwargs["repo_full_name"],
                    repo_clone_url="https://github.com/owner/repo.git",
                    commit_sha=kwargs["commit_sha"],
                    parent_sha="parent",
                    message="ok",
                    html_url="",
                    author_name=None,
                    event_id="",
                    files=[
                        CommitFile(
                            filename="app.py",
                            status="modified",
                            additions=100,
                            deletions=0,
                            changes=100,
                            patch="@@ -1 +1 @@\n-old\n+new",
                        ),
                    ],
                )

            miner._fetch_commit_candidate = fake_fetch  # type: ignore[method-assign]
            candidate = miner.sample_commit(max_attempts=2)
            self.assertEqual(candidate.commit_sha, "good-sha")
            self.assertEqual(fetch_calls["count"], 1)
            miner.close()

    def test_pop_date_commit_skips_cached_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "rejects.jsonl"
            cache = CommitRejectCache(cache_path)
            cache.add("owner/repo", "bad-sha", "cached")
            _extend_date_commit_buffer([("owner/repo", "bad-sha"), ("owner/repo", "good-sha")])
            self.assertEqual(_pop_date_commit(cache), ("owner/repo", "good-sha"))
            self.assertIsNone(_pop_date_commit(cache))

    def test_sample_commit_does_not_retry_same_event_commit(self):
        miner = GitHubMiner(rng=random.Random(1), start_search_refiller=False)
        events = [
            {
                "type": "PushEvent",
                "id": "event-1",
                "repo": {"name": "owner/repo"},
                "payload": {"commits": [{"sha": "bad-sha"}]},
            }
        ]
        calls = {"events": 0, "commit": 0}

        def fake_recent_events():
            calls["events"] += 1
            return events

        def fake_fetch_commit_candidate(**_kwargs):
            calls["commit"] += 1
            raise ValueError("bad commit")

        miner._recent_push_events = fake_recent_events  # type: ignore[method-assign]
        miner._fetch_commit_candidate = fake_fetch_commit_candidate  # type: ignore[method-assign]
        miner._sample_commit_by_random_day = lambda: None  # type: ignore[method-assign]
        miner._reject_cache.add = lambda *args, **kwargs: None  # type: ignore[method-assign]

        with self.assertRaisesRegex(RuntimeError, "No commits from date search and no usable recent push events"):
            miner.sample_commit(max_attempts=3)

        self.assertEqual(calls, {"events": 1, "commit": 1})
        miner.close()

    def test_event_fallback_skips_cached_rejects_before_fetch(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache_path = Path(tmp) / "rejects.jsonl"
            cache = CommitRejectCache(cache_path)
            cache.add("owner/repo", "bad-sha", "cached")

            miner = GitHubMiner(
                rng=random.Random(1),
                reject_cache_path=cache_path,
                start_search_refiller=False,
            )
            events = [
                {
                    "type": "PushEvent",
                    "id": "event-1",
                    "repo": {"name": "owner/repo"},
                    "payload": {
                        "commits": [
                            {"sha": "bad-sha", "modified": ["src/bad.py"]},
                            {"sha": "good-sha", "modified": ["src/good.py"]},
                        ],
                    },
                }
            ]

            candidates = miner._fallback_event_commit_candidates(events)

            self.assertEqual(candidates, [("owner/repo", "good-sha", "event-1")])
            miner.close()

    def test_event_fallback_candidates_are_process_wide_unique(self):
        events = [
            {
                "type": "PushEvent",
                "id": "event-1",
                "repo": {"name": "owner/repo"},
                "payload": {"commits": [{"sha": "bad-sha"}]},
            }
        ]
        miner_a = GitHubMiner(rng=random.Random(1), start_search_refiller=False)
        miner_b = GitHubMiner(rng=random.Random(2), start_search_refiller=False)
        calls = {"commit": 0}

        def fake_fetch_commit_candidate(**_kwargs):
            calls["commit"] += 1
            raise ValueError("bad commit")

        for miner in (miner_a, miner_b):
            miner._recent_push_events = lambda: events  # type: ignore[method-assign]
            miner._fetch_commit_candidate = fake_fetch_commit_candidate  # type: ignore[method-assign]
            miner._sample_commit_by_random_day = lambda: None  # type: ignore[method-assign]
            miner._reject_cache.add = lambda *args, **kwargs: None  # type: ignore[method-assign]

        with self.assertRaisesRegex(RuntimeError, "bad commit"):
            miner_a.sample_commit(max_attempts=1)
        with self.assertRaisesRegex(RuntimeError, "No commits from date search and no usable recent push events"):
            miner_b.sample_commit(max_attempts=1)

        self.assertEqual(calls, {"commit": 1})
        miner_a.close()
        miner_b.close()

    def test_pick_random_commit_sha_prefers_event_commits_with_code_hints(self):
        miner = GitHubMiner(rng=random.Random(1), start_search_refiller=False)
        event = {
            "payload": {
                "commits": [
                    {"sha": "docs", "modified": ["README.md"]},
                    {"sha": "code", "modified": ["src/app.py"]},
                ],
            },
        }

        self.assertEqual(miner._pick_random_commit_sha(event), "code")
        miner.close()

    def test_pick_random_commit_sha_falls_back_without_file_hints(self):
        miner = GitHubMiner(rng=random.Random(1), start_search_refiller=False)
        event = {
            "payload": {
                "commits": [
                    {"sha": "first"},
                    {"sha": "second"},
                ],
            },
        }

        self.assertIn(miner._pick_random_commit_sha(event), {"first", "second"})
        miner.close()

    def test_search_refiller_extends_buffer_when_low(self):
        rotator = GitHubTokenRotator(["token-a"])
        with patch.object(
            GitHubMiner,
            "_search_commits_for_random_day",
            return_value=[("owner/a", "sha-a"), ("owner/b", "sha-b")],
        ):
            from github_miner import _commit_search_refiller_loop

            with patch("github_miner._SEARCH_REFILLER_STOP") as mock_stop:
                mock_stop.is_set.return_value = False
                mock_stop.wait.return_value = True
                _commit_search_refiller_loop(token_rotator=rotator, timeout=5.0)
        self.assertEqual(_pop_date_commit(None), ("owner/b", "sha-b"))
        self.assertEqual(_pop_date_commit(None), ("owner/a", "sha-a"))

    def test_first_symlink_tree_path_returns_sorted_symlink_path(self):
        payload = {
            "tree": [
                {"path": "z-link", "type": "blob", "mode": "120000"},
                {"path": "regular.py", "type": "blob", "mode": "100644"},
                {"path": ".antigravitycli/file.json", "type": "blob", "mode": "120000"},
            ],
        }

        self.assertEqual(first_symlink_tree_path(payload), ".antigravitycli/file.json")

    def test_recent_push_events_cache_is_shared_between_miners(self):
        event_payload = [
            {
                "type": "PushEvent",
                "id": "event-1",
                "repo": {"name": "owner/repo"},
                "payload": {"commits": [{"sha": "abc"}]},
            }
        ]
        response = httpx.Response(200, json=event_payload)
        response.headers["link"] = ""
        miner_a = GitHubMiner(rng=random.Random(1), start_search_refiller=False)
        miner_b = GitHubMiner(rng=random.Random(2), start_search_refiller=False)
        miner_a._client = FakeGitHubClient([response])
        miner_b._client = FakeGitHubClient([])

        self.assertEqual(miner_a._recent_push_events(), event_payload)
        self.assertEqual(miner_b._recent_push_events(), event_payload)
        self.assertEqual(len(miner_a._client.headers_seen), 1)
        self.assertEqual(len(miner_b._client.headers_seen), 0)
        miner_a.close()
        miner_b.close()


if __name__ == "__main__":
    unittest.main()
