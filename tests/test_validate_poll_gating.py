import unittest
from unittest.mock import Mock

from validate import (
    _POOL_FILLER_RATE_LIMIT_BACKOFF_BUFFER_SECONDS,
    _POOL_FILLER_RATE_LIMIT_BACKOFF_SECONDS,
    _github_rate_limit_backoff_seconds,
    _remaining_poll_sleep_seconds,
    _should_refresh_chain_submissions,
)


class ValidatePollGatingTest(unittest.TestCase):
    def test_force_always_refreshes(self):
        self.assertTrue(
            _should_refresh_chain_submissions(
                force=True,
                current_block=100,
                last_refresh_block=99,
                interval_blocks=360,
            )
        )

    def test_first_refresh_without_history_runs_immediately(self):
        self.assertTrue(
            _should_refresh_chain_submissions(
                force=False,
                current_block=100,
                last_refresh_block=None,
                interval_blocks=360,
            )
        )

    def test_regular_refresh_ignores_interval_blocks(self):
        self.assertTrue(
            _should_refresh_chain_submissions(
                force=False,
                current_block=150,
                last_refresh_block=100,
                interval_blocks=360,
            )
        )
        self.assertTrue(
            _should_refresh_chain_submissions(
                force=False,
                current_block=460,
                last_refresh_block=100,
                interval_blocks=360,
            )
        )

    def test_poll_sleep_counts_work_done_during_loop(self):
        self.assertEqual(
            _remaining_poll_sleep_seconds(started_at=100.0, interval_seconds=600, now=250.0),
            450.0,
        )
        self.assertEqual(
            _remaining_poll_sleep_seconds(started_at=100.0, interval_seconds=600, now=700.0),
            0.0,
        )

    def test_github_rate_limit_backoff_prefers_retry_after_with_floor(self):
        response = Mock(headers={"retry-after": "45"})

        self.assertEqual(
            _github_rate_limit_backoff_seconds(response),
            _POOL_FILLER_RATE_LIMIT_BACKOFF_SECONDS,
        )

    def test_github_rate_limit_backoff_uses_core_reset_when_remaining_zero(self):
        response = Mock(headers={"x-ratelimit-remaining": "0", "x-ratelimit-reset": "1700000900"})

        self.assertEqual(
            _github_rate_limit_backoff_seconds(response, now=1700000000),
            900 + _POOL_FILLER_RATE_LIMIT_BACKOFF_BUFFER_SECONDS,
        )


if __name__ == "__main__":
    unittest.main()
