import json
import unittest
import unittest.mock
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from docker_solver import (
    _DockerSolverCommandResult,
    _proxy_request_is_provider_account_error,
    _proxy_request_is_provider_endpoint_error,
    _resolve_exit_reason,
)
from openrouter_proxy import (
    OpenRouterProxy,
    UpstreamResponse,
    _upstream_base_url,
    select_solver_upstream_base_url,
    solver_upstream_base_urls_from_env,
)
from sampling_seed import VALIDATOR_TOP_P
from solver_runner import COMPLETED_EXIT_REASON, PROVIDER_ACCOUNT_ERROR_EXIT_REASON
from tau.io.upstream_request_policy import UpstreamRequestPolicy

_SAMPLE_PAYLOAD = {
    "model": "test/model",
    "messages": [{"role": "user", "content": "hello"}],
    "temperature": 0.0,
    "top_p": VALIDATOR_TOP_P,
}
_SAMPLE_UPSTREAM_RESPONSE = UpstreamResponse(
    body=json.dumps(
        {
            "id": "chatcmpl-abc",
            "choices": [{"message": {"content": "hi there"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        }
    ).encode("utf-8"),
    payload={
        "id": "chatcmpl-abc",
        "choices": [{"message": {"content": "hi there"}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
    },
    status=200,
    headers=httpx.Headers({"Content-Type": "application/json"}),
    first_token_latency_ms=None,
)


class OpenRouterProxyModelEnforcementTest(unittest.TestCase):
    def test_upstream_base_url_reads_env_at_request_time(self):
        with patch.dict(
            "openrouter_proxy.os.environ",
            {"OPENROUTER_BASE_URL": "https://example.test/custom/v1"},
            clear=False,
        ):
            self.assertEqual(_upstream_base_url(), "https://example.test/custom")

    def test_solver_upstream_base_urls_reads_comma_list(self):
        with patch.dict(
            "openrouter_proxy.os.environ",
            {
                "SOLVER_UPSTREAM_BASE_URLS": (
                    "https://gpu-a.example/v1, https://gpu-b.example/v1/chat/completions"
                )
            },
            clear=False,
        ):
            self.assertEqual(
                solver_upstream_base_urls_from_env(),
                ["https://gpu-a.example", "https://gpu-b.example"],
            )

    def test_select_solver_upstream_base_url_is_stable_for_key(self):
        with patch.dict(
            "openrouter_proxy.os.environ",
            {
                "SOLVER_UPSTREAM_BASE_URLS": (
                    "https://gpu-a.example/v1, https://gpu-b.example/v1"
                )
            },
            clear=False,
        ):
            first = select_solver_upstream_base_url("task-a\0solution-a")
            second = select_solver_upstream_base_url("task-a\0solution-a")
            self.assertEqual(first, second)
            self.assertIn(first, {"https://gpu-a.example", "https://gpu-b.example"})

    def test_proxy_can_pin_upstream_base_url_for_session(self):
        with patch.dict(
            "openrouter_proxy.os.environ",
            {"SOLVER_UPSTREAM_BASE_URL": "https://global.example/v1"},
            clear=False,
        ):
            proxy = OpenRouterProxy(
                openrouter_api_key="upstream-key",
                upstream_base_url="https://pinned.example/v1/chat/completions",
            )
            self.assertEqual(proxy._upstream_base_url(), "https://pinned.example")

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
        self.assertEqual(prepared["top_p"], VALIDATOR_TOP_P)
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

    def test_shell_tools_policy_is_applied_when_configured(self):
        proxy = OpenRouterProxy(
            openrouter_api_key="upstream-key",
            enforced_model="provider/model",
            upstream_request_policy=UpstreamRequestPolicy(shell_tools=True),
        )
        body = json.dumps(
            {
                "model": "miner/chosen-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "miner_tool"}}],
            }
        ).encode("utf-8")

        prepared_body, rejection_reason = proxy._prepare_request_body(
            body=body,
            request_payload=json.loads(body.decode("utf-8")),
        )

        self.assertIsNone(rejection_reason)
        prepared = json.loads(prepared_body.decode("utf-8"))
        self.assertEqual(prepared["tool_choice"], "auto")
        self.assertEqual(len(prepared["tools"]), 1)
        self.assertEqual(prepared["tools"][0]["function"]["name"], "bash")

    def test_text_only_policy_is_applied_when_configured(self):
        proxy = OpenRouterProxy(
            openrouter_api_key="upstream-key",
            enforced_model="provider/model",
            upstream_request_policy=UpstreamRequestPolicy(text_only=True),
        )
        body = json.dumps(
            {
                "model": "miner/chosen-model",
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": ""},
                ],
                "tools": [{"type": "function", "function": {"name": "bash"}}],
            }
        ).encode("utf-8")

        prepared_body, rejection_reason = proxy._prepare_request_body(
            body=body,
            request_payload=json.loads(body.decode("utf-8")),
        )

        self.assertIsNone(rejection_reason)
        self.assertIsNotNone(prepared_body)
        prepared = json.loads(prepared_body.decode("utf-8"))
        self.assertEqual(prepared["model"], "provider/model")
        self.assertEqual(prepared["tool_choice"], "none")
        self.assertNotIn("tools", prepared)
        self.assertFalse(prepared["parallel_tool_calls"])
        self.assertEqual(len(prepared["messages"]), 1)

    def test_requests_without_policy_keep_tools(self):
        proxy = OpenRouterProxy(
            openrouter_api_key="upstream-key",
            enforced_model="provider/model",
        )
        body = json.dumps(
            {
                "model": "miner/chosen-model",
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [{"type": "function", "function": {"name": "bash"}}],
            }
        ).encode("utf-8")

        prepared_body, rejection_reason = proxy._prepare_request_body(
            body=body,
            request_payload=json.loads(body.decode("utf-8")),
        )

        self.assertIsNone(rejection_reason)
        prepared = json.loads(prepared_body.decode("utf-8"))
        self.assertIn("tools", prepared)
        self.assertNotIn("tool_choice", prepared)

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


    def test_rejected_request_emits_rollout_event(self):
        events = []
        proxy = OpenRouterProxy(openrouter_api_key="upstream-key", rollout_event_sink=events.append)

        class WFile:
            def __init__(self):
                self.body = b""
            def write(self, body):
                self.body += body
            def flush(self):
                pass

        class Handler:
            def __init__(self):
                self.wfile = WFile()
                self.headers = []
                self.status = None
                self.close_connection = False
            def send_response(self, status):
                self.status = status
            def send_header(self, key, value):
                self.headers.append((key, value))
            def end_headers(self):
                pass

        handler = Handler()
        proxy._reject_request(
            handler,
            reason="proxy_error",
            status=403,
            error_type="proxy_policy_violation",
            message="Endpoint not allowed",
            method="POST",
            path="/v1/not-allowed",
            request_model="model/a",
        )

        self.assertEqual(handler.status, 403)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "llm_call")
        self.assertEqual(events[0]["status_code"], 403)
        self.assertEqual(events[0]["response"]["error"]["code"], "proxy_error")


class OpenRouterRateLimitHandlingTest(unittest.TestCase):
    def test_upstream_response_is_retryable_rate_limit(self):
        from openrouter_proxy import _upstream_response_is_retryable_rate_limit

        retryable = UpstreamResponse(
            body=b'{"error":{"message":"Rate limit exceeded"}}',
            payload={"error": {"message": "Rate limit exceeded"}},
            status=429,
            headers=httpx.Headers({}),
            first_token_latency_ms=None,
        )
        budget = UpstreamResponse(
            body=b'{"error":{"message":"Solve budget exceeded"}}',
            payload={"error": {"message": "Solve budget exceeded"}},
            status=429,
            headers=httpx.Headers({}),
            first_token_latency_ms=None,
        )
        self.assertTrue(_upstream_response_is_retryable_rate_limit(retryable))
        self.assertFalse(_upstream_response_is_retryable_rate_limit(budget))

    def test_fetch_retries_rate_limited_upstream_response(self):
        from openrouter_proxy import HttpxUpstreamClient

        responses = [
            UpstreamResponse(
                body=b'{"error":{"message":"Rate limit exceeded"}}',
                payload={"error": {"message": "Rate limit exceeded"}},
                status=429,
                headers=httpx.Headers({}),
                first_token_latency_ms=None,
            ),
            _SAMPLE_UPSTREAM_RESPONSE,
        ]
        client = HttpxUpstreamClient()

        with unittest.mock.patch("openrouter_proxy._should_stream_chat_completion", return_value=False):
            with unittest.mock.patch.object(client, "_fetch_direct", side_effect=responses) as fetch_direct:
                with unittest.mock.patch("openrouter_proxy.time.sleep"):
                    result = client.fetch(
                        command="POST",
                        url="http://unused",
                        headers={},
                        body=b"{}",
                        prepared_payload=_SAMPLE_PAYLOAD,
                        request_path="/v1/chat/completions",
                        start=0.0,
                    )

        self.assertEqual(result.status, 200)
        self.assertEqual(fetch_direct.call_count, 2)



if __name__ == "__main__":
    unittest.main()
