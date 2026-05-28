from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from config import RunConfig


class RunConfigGitHubTokenTest(unittest.TestCase):
    def test_reserves_unarbos_token_for_merge_only(self):
        with patch.dict(
            os.environ,
            {
                "GITHUB_MERGE_TOKEN": "",
                "GITHUB_TOKEN_UNARBOS": "write-token",
                "GITHUB_TASK_TOKEN": "",
                "GITHUB_READ_TOKEN": "",
                "GITHUB_TOKEN": "write-token",
                "GH_TOKEN": "",
                "GITHUB_TOKENS": "read-a,write-token,read-b",
            },
            clear=False,
        ):
            config = RunConfig()

        self.assertEqual(config.github_merge_token, "write-token")
        self.assertIsNone(config.github_token)
        self.assertEqual(config.github_tokens, "read-a,read-b")

    def test_prefers_task_read_token_for_general_github_work(self):
        with patch.dict(
            os.environ,
            {
                "GITHUB_MERGE_TOKEN": "",
                "GITHUB_TOKEN_UNARBOS": "write-token",
                "GITHUB_TASK_TOKEN": "read-token",
                "GITHUB_READ_TOKEN": "",
                "GITHUB_TOKEN": "write-token",
                "GH_TOKEN": "",
                "GITHUB_TOKENS": "",
            },
            clear=False,
        ):
            config = RunConfig()

        self.assertEqual(config.github_merge_token, "write-token")
        self.assertEqual(config.github_token, "read-token")


if __name__ == "__main__":
    unittest.main()
