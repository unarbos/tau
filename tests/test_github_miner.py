import random
import unittest

import httpx

from github_miner import GitHubMiner, GitHubTokenRotator


class FakeGitHubClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.headers_seen = []

    def get(self, path, params=None, headers=None):
        self.headers_seen.append(dict(headers or {}))
        response = self.responses.pop(0)
        response.request = httpx.Request("GET", "https://api.github.com" + path)
        return response


class GitHubTokenRotatorTest(unittest.TestCase):
    def test_401_disables_token_and_retries_next_token(self):
        rotator = GitHubTokenRotator(["bad-token", "good-token"])
        miner = GitHubMiner(token_rotator=rotator, rng=random.Random(1))
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

    def test_all_401_tokens_fall_back_to_unauthenticated_request(self):
        rotator = GitHubTokenRotator(["bad-token"])
        miner = GitHubMiner(token_rotator=rotator, rng=random.Random(1))
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


if __name__ == "__main__":
    unittest.main()
