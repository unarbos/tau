import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from docker_solver import (
    _DockerSolverCommandResult,
    _proxy_request_is_provider_account_error,
    _proxy_request_is_provider_endpoint_error,
    _resolve_exit_reason,
)
from openrouter_proxy import OpenRouterProxy, _upstream_base_url
from solver_runner import COMPLETED_EXIT_REASON, PROVIDER_ACCOUNT_ERROR_EXIT_REASON


class OpenRouterProxyModelEnforcementTest(unittest.TestCase):
    def test_upstream_base_url_reads_env_at_request_time(self):
        with patch.dict(
            "openrouter_proxy.os.environ",
            {"OPENROUTER_BASE_URL": "https://example.test/custom/v1"},
            clear=False,
        ):
            self.assertEqual(_upstream_base_url(), "https://example.test/custom")

    def test_rewrites_requested_model_to_validator_model(self):
        proxy = OpenRouterProxy(openrouter_api_key="upstream-key", enforced_model="validator/model")
        body = json.dumps(
            {
                "model": "miner/chosen-model",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 12,
            }
        ).encode("utf-8")

        prepared_body, rejection_reason = proxy._prepare_request_body(
            body=body,
            request_payload=json.loads(body.decode("utf-8")),
        )

        self.assertIsNone(rejection_reason)
        self.assertIsNotNone(prepared_body)
        prepared = json.loads(prepared_body.decode("utf-8"))
        self.assertEqual(prepared["model"], "validator/model")

    def test_adds_validator_model_when_request_omits_model(self):
        proxy = OpenRouterProxy(openrouter_api_key="upstream-key", enforced_model="validator/model")
        body = json.dumps({"messages": [{"role": "user", "content": "hi"}]}).encode("utf-8")

        prepared_body, rejection_reason = proxy._prepare_request_body(
            body=body,
            request_payload=json.loads(body.decode("utf-8")),
        )

        self.assertIsNone(rejection_reason)
        self.assertIsNotNone(prepared_body)
        prepared = json.loads(prepared_body.decode("utf-8"))
        self.assertEqual(prepared["model"], "validator/model")

    def test_rewrites_sampling_params_to_validator_policy(self):
        proxy = OpenRouterProxy(openrouter_api_key="upstream-key", enforced_model="validator/model")
        body = json.dumps(
            {
                "model": "miner/chosen-model",
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 1.0,
                "top_p": 0.2,
                "top_k": 7,
                "seed": 123,
                "presence_penalty": 1.5,
            }
        ).encode("utf-8")

        prepared_body, rejection_reason = proxy._prepare_request_body(
            body=body,
            request_payload=json.loads(body.decode("utf-8")),
        )

        self.assertIsNone(rejection_reason)
        self.assertIsNotNone(prepared_body)
        prepared = json.loads(prepared_body.decode("utf-8"))
        self.assertEqual(prepared["temperature"], 0.0)
        self.assertEqual(prepared["top_p"], 1.0)
        self.assertNotIn("top_k", prepared)
        self.assertNotIn("seed", prepared)
        self.assertNotIn("presence_penalty", prepared)

    def test_rewrites_provider_to_validator_policy(self):
        proxy = OpenRouterProxy(
            openrouter_api_key="upstream-key",
            enforced_model="validator/model",
            enforced_provider={
                "sort": "throughput",
                "only": ["validator/highspeed"],
                "allow_fallbacks": False,
                "preferred_min_throughput": {"p90": 50},
            },
        )
        body = json.dumps(
            {
                "model": "miner/chosen-model",
                "messages": [{"role": "user", "content": "hi"}],
                "provider": {"only": ["slow-provider"]},
            }
        ).encode("utf-8")

        prepared_body, rejection_reason = proxy._prepare_request_body(
            body=body,
            request_payload=json.loads(body.decode("utf-8")),
        )

        self.assertIsNone(rejection_reason)
        self.assertIsNotNone(prepared_body)
        prepared = json.loads(prepared_body.decode("utf-8"))
        self.assertEqual(
            prepared["provider"],
            {
                "sort": "throughput",
                "only": ["validator/highspeed"],
                "allow_fallbacks": False,
                "preferred_min_throughput": {"p90": 50},
            },
        )

    def test_provider_endpoint_error_detection_matches_upstream_failures(self):
        self.assertTrue(
            _proxy_request_is_provider_endpoint_error(
                SimpleNamespace(status_code=429, error="rate limited by upstream provider")
            )
        )
        self.assertTrue(
            _proxy_request_is_provider_endpoint_error(
                SimpleNamespace(status_code=502, error="bad gateway")
            )
        )
        self.assertTrue(
            _proxy_request_is_provider_endpoint_error(
                SimpleNamespace(status_code=400, error="Provider returned error: no endpoints available")
            )
        )
        self.assertFalse(
            _proxy_request_is_provider_endpoint_error(
                SimpleNamespace(status_code=400, error="Request body must include messages")
            )
        )



    def test_provider_account_error_detection_matches_billing_and_auth_failures(self):
        for status_code, error in (
            (401, "unauthorized"),
            (402, "insufficient credits"),
            (403, "invalid api key"),
            (400, "billing quota exceeded"),
        ):
            self.assertTrue(
                _proxy_request_is_provider_account_error(
                    SimpleNamespace(status_code=status_code, error=error)
                )
            )
        self.assertFalse(
            _proxy_request_is_provider_account_error(
                SimpleNamespace(status_code=502, error="bad gateway")
            )
        )

    def test_failed_solve_with_account_error_uses_account_exit_reason(self):
        proxy = SimpleNamespace(
            budget_exceeded_reason=None,
            usage_snapshot=lambda: SimpleNamespace(
                requests=[SimpleNamespace(status_code=402, error="insufficient credits")]
            ),
        )

        exit_reason = _resolve_exit_reason(
            solver_run=_DockerSolverCommandResult(returncode=1, stdout="", stderr=""),
            proxy=proxy,
        )

        self.assertEqual(exit_reason, PROVIDER_ACCOUNT_ERROR_EXIT_REASON)

    def test_recovered_provider_error_does_not_override_successful_solve(self):
        proxy = SimpleNamespace(
            budget_exceeded_reason=None,
            usage_snapshot=lambda: SimpleNamespace(
                requests=[SimpleNamespace(status_code=429, error="rate limited by upstream provider")]
            ),
        )

        exit_reason = _resolve_exit_reason(
            solver_run=_DockerSolverCommandResult(returncode=0, stdout="", stderr=""),
            proxy=proxy,
        )

        self.assertEqual(exit_reason, COMPLETED_EXIT_REASON)



if __name__ == "__main__":
    unittest.main()
