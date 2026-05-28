import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "openrouter_solver_backend_probe.py"
spec = importlib.util.spec_from_file_location("openrouter_solver_backend_probe", SCRIPT)
probe = importlib.util.module_from_spec(spec)
sys.modules["openrouter_solver_backend_probe"] = probe
assert spec.loader is not None
spec.loader.exec_module(probe)

from config import RunConfig


class OpenRouterSolverBackendProbeTest(unittest.TestCase):
    def test_payload_matches_validator_owned_solver_controls(self):
        config = RunConfig(
            solver_model="openrouter/validator/model",
            solver_provider_sort="throughput",
            solver_provider_only="fast-a, fast-b",
            solver_provider_allow_fallbacks=False,
            solver_provider_min_throughput_p90=42.0,
        )

        payload = probe.build_validator_equivalent_payload(
            config=config,
            task_name="validate-example",
            task_prompt="Fix the parser.",
            max_tokens=32,
        )

        self.assertEqual(payload["model"], "validator/model")
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["top_p"], 1.0)
        self.assertEqual(payload["max_tokens"], 32)
        self.assertEqual(
            payload["provider"],
            {
                "sort": "throughput",
                "only": ["fast-a", "fast-b"],
                "allow_fallbacks": False,
                "preferred_min_throughput": {"p90": 42.0},
            },
        )
        self.assertIn("Fix the parser.", payload["messages"][1]["content"])

    def test_chat_url_uses_openrouter_v1_endpoint(self):
        self.assertEqual(
            probe.openrouter_chat_url("https://openrouter.ai/api"),
            "https://openrouter.ai/api/v1/chat/completions",
        )

    def test_post_sends_payload_to_openrouter_without_mutating_it(self):
        payload = {"model": "validator/model", "messages": []}
        response_payload = {"id": "gen-1", "model": "validator/model", "choices": []}

        class FakeResponse:
            def raise_for_status(self):
                return None

            def json(self):
                return response_payload

        class FakeClient:
            def __init__(self, *, timeout):
                self.timeout = timeout

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def post(self, url, *, headers, json):
                self.url = url
                self.headers = headers
                self.json = json
                return FakeResponse()

        with patch.object(probe.httpx, "Client", FakeClient):
            response = probe.post_openrouter_probe(
                payload=payload,
                api_key="key",
                base_url="https://example.test/api/v1",
                timeout=12,
            )

        self.assertEqual(response, response_payload)
        self.assertEqual(payload, {"model": "validator/model", "messages": []})


if __name__ == "__main__":
    unittest.main()
