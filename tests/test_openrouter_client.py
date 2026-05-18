import unittest
from unittest.mock import patch

from openrouter_client import complete_text


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class _FakeClient:
    def __init__(self, payload):
        self.payload = payload
        self.request_json = None
        self.request_url = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, headers, json):
        self.request_url = url
        self.request_json = json
        return _FakeResponse(self.payload)


class OpenRouterClientTest(unittest.TestCase):
    def test_complete_text_passes_reasoning_config(self):
        client = _FakeClient(
            {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"},
                ],
            },
        )

        with patch("openrouter_client.httpx.Client", return_value=client):
            text = complete_text(
                prompt="judge",
                model="deepseek/deepseek-v4-flash",
                timeout=10,
                openrouter_api_key="key",
                reasoning={"effort": "medium", "exclude": True},
            )

        self.assertEqual(text, "ok")
        self.assertEqual(client.request_json["reasoning"], {"effort": "medium", "exclude": True})

    def test_empty_content_error_includes_reasoning_metadata(self):
        client = _FakeClient(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "native_finish_reason": "length",
                        "message": {"content": None, "reasoning": "thinking"},
                    },
                ],
                "usage": {
                    "completion_tokens": 1200,
                    "completion_tokens_details": {"reasoning_tokens": 1200},
                },
            },
        )

        with patch("openrouter_client.httpx.Client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "reasoning_tokens=1200"):
                complete_text(
                    prompt="judge",
                    model="deepseek/deepseek-v4-flash",
                    timeout=10,
                    openrouter_api_key="key",
                )

    def test_no_choices_error_includes_openrouter_error_payload(self):
        client = _FakeClient(
            {
                "error": {
                    "code": 429,
                    "message": "rate limited by upstream provider",
                },
            },
        )

        with patch("openrouter_client.httpx.Client", return_value=client):
            with self.assertRaisesRegex(RuntimeError, "error_code=429"):
                complete_text(
                    prompt="judge",
                    model="deepseek/deepseek-v4-flash",
                    timeout=10,
                    openrouter_api_key="key",
                )

        with self.assertRaisesRegex(RuntimeError, "rate limited by upstream provider"):
            with patch("openrouter_client.httpx.Client", return_value=client):
                complete_text(
                    prompt="judge",
                    model="deepseek/deepseek-v4-flash",
                    timeout=10,
                    openrouter_api_key="key",
                )

    def test_complete_text_reads_base_url_from_env_at_call_time(self):
        client = _FakeClient(
            {
                "choices": [
                    {"message": {"content": "ok"}, "finish_reason": "stop"},
                ],
            },
        )

        with patch.dict(
            "openrouter_client.os.environ",
            {"OPENROUTER_BASE_URL": "https://example.test/custom"},
            clear=False,
        ):
            with patch("openrouter_client.httpx.Client", return_value=client):
                complete_text(
                    prompt="judge",
                    model="deepseek/deepseek-v4-flash",
                    timeout=10,
                    openrouter_api_key="key",
                )

        self.assertEqual(
            client.request_url,
            "https://example.test/custom/v1/chat/completions",
        )


if __name__ == "__main__":
    unittest.main()
