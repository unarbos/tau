from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import socketserver
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, cast

import httpx

import tau.utils
from sampling_seed import VALIDATOR_TOP_P
from tau.io.chat_completion import (
    append_stream_text_part,
    merge_stream_tool_call_delta,
    normalize_chat_completion_payload,
    payload_has_retryable_empty_content,
)
from tau.io.openrouter import CacheMissError, normalize_base_url
from tau.io.upstream_request_policy import (
    DEFAULT_EMPTY_RESPONSE_RETRIES,
    DEFAULT_RATE_LIMIT_RETRIES,
    UpstreamRequestPolicy,
    apply_upstream_request_policy,
)
from tau.rollouts.schema import build_llm_event, utc_now
from tau.utils import DiskCache, json_sha256

log = logging.getLogger("swe-eval.openrouter_proxy")

_UPSTREAM_TIMEOUT = httpx.Timeout(connect=30.0, read=600.0, write=30.0, pool=30.0)
REQUEST_LIMIT_EXIT_REASON = "request_limit_exceeded"
TOKEN_LIMIT_EXIT_REASON = "token_limit_exceeded"
COST_LIMIT_EXIT_REASON = "cost_limit_exceeded"
PROXY_ERROR_EXIT_REASON = "proxy_error"
_MAX_REQUEST_BODY_BYTES = 2 * 1024 * 1024
_ALLOWED_METHODS = {"POST", "HEAD"}
_ALLOWED_PATHS = {"/v1/chat/completions", "/v1/messages"}
_VALIDATOR_SAMPLING_PARAMS = {
    "temperature": 0.0,
    "top_p": VALIDATOR_TOP_P,
    # seed is injected per solve via enforced_sampling_params in docker_solver.
}
_MINER_CONTROLLED_SAMPLING_PARAMS = {
    "top_k",
    "min_p",
    "top_a",
    "frequency_penalty",
    "presence_penalty",
    "repetition_penalty",
    "seed",
    "logit_bias",
    "logprobs",
    "top_logprobs",
}
_ESTIMATED_CHARS_PER_TOKEN = 3
_ESTIMATED_MESSAGE_OVERHEAD_TOKENS = 8
_ESTIMATED_TOOL_OVERHEAD_TOKENS = 24
_HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "host",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


def _upstream_base_url() -> str:
    # SOLVER_UPSTREAM_BASE_URL lets the solver proxy target a different backend
    # (e.g. a self-hosted vLLM endpoint) than the judge, which keeps using
    # OPENROUTER_UPSTREAM_BASE_URL/OPENROUTER_BASE_URL via its separate client.
    return normalize_base_url(
        os.environ.get("SOLVER_UPSTREAM_BASE_URL")
        or os.environ.get("OPENROUTER_UPSTREAM_BASE_URL")
        or os.environ.get("OPENROUTER_BASE_URL"),
    )


def _split_upstream_base_urls(raw: str | None) -> list[str]:
    if not raw:
        return []
    urls: list[str] = []
    for item in raw.replace("\n", ",").split(","):
        value = item.strip()
        if value:
            urls.append(normalize_base_url(value))
    return urls


def solver_upstream_base_urls_from_env() -> list[str]:
    urls = _split_upstream_base_urls(os.environ.get("SOLVER_UPSTREAM_BASE_URLS"))
    if urls:
        return urls
    return [_upstream_base_url()]


def select_solver_upstream_base_url(shard_key: str) -> str:
    urls = solver_upstream_base_urls_from_env()
    if len(urls) == 1:
        return urls[0]
    digest = hashlib.sha256(shard_key.encode("utf-8")).digest()
    index = int.from_bytes(digest[:8], "big") % len(urls)
    return urls[index]


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class _ReusableThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    allow_reuse_address = True
    daemon_threads = True


@dataclass(slots=True)
class SolveBudget:
    max_requests: int | None = None
    max_total_tokens: int | None = None
    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_cost: float | None = None
    max_tokens_per_request: int | None = None

    def enabled(self) -> bool:
        return any(
            value is not None
            for value in (
                self.max_requests,
                self.max_total_tokens,
                self.max_prompt_tokens,
                self.max_completion_tokens,
                self.max_cost,
                self.max_tokens_per_request,
            )
        )

    @classmethod
    def from_config(cls, config: Any | None) -> SolveBudget | None:
        if config is None:
            return None
        budget = cls(
            max_requests=getattr(config, "solver_max_requests", None),
            max_total_tokens=getattr(config, "solver_max_total_tokens", None),
            max_prompt_tokens=getattr(config, "solver_max_prompt_tokens", None),
            max_completion_tokens=getattr(config, "solver_max_completion_tokens", None),
            max_cost=getattr(config, "solver_max_cost", None),
            max_tokens_per_request=getattr(config, "solver_max_tokens_per_request", None),
        )
        return budget if budget.enabled() else None


@dataclass(slots=True)
class ProxyRequestRecord:
    method: str
    path: str
    status_code: int | None
    latency_ms: int
    request_model: str | None = None
    response_model: str | None = None
    generation_id: str | None = None
    first_token_latency_ms: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    cost: float | None = None
    rejected: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "path": self.path,
            "status_code": self.status_code,
            "latency_ms": self.latency_ms,
            "request_model": self.request_model,
            "response_model": self.response_model,
            "generation_id": self.generation_id,
            "first_token_latency_ms": self.first_token_latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost": self.cost,
            "rejected": self.rejected,
            "error": self.error,
        }


@dataclass(slots=True)
class SolveUsageSummary:
    request_count: int = 0
    rejected_request_count: int = 0
    first_token_count: int = 0
    success_count: int = 0
    error_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cached_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    cost: float = 0.0
    budget_exceeded_reason: str | None = None
    requests: list[ProxyRequestRecord] = field(default_factory=list)

    def snapshot(self) -> SolveUsageSummary:
        return SolveUsageSummary(
            request_count=self.request_count,
            rejected_request_count=self.rejected_request_count,
            first_token_count=self.first_token_count,
            success_count=self.success_count,
            error_count=self.error_count,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            total_tokens=self.total_tokens,
            cached_tokens=self.cached_tokens,
            cache_write_tokens=self.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens,
            cost=self.cost,
            budget_exceeded_reason=self.budget_exceeded_reason,
            requests=[ProxyRequestRecord(**request.to_dict()) for request in self.requests],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_count": self.request_count,
            "rejected_request_count": self.rejected_request_count,
            "first_token_count": self.first_token_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cost": self.cost,
            "budget_exceeded_reason": self.budget_exceeded_reason,
            "requests": [request.to_dict() for request in self.requests],
        }


@dataclass(slots=True)
class UpstreamResponse:
    body: bytes
    payload: Any
    status: int
    headers: httpx.Headers
    first_token_latency_ms: int | None


class UpstreamClient(ABC):
    @abstractmethod
    def fetch(
        self,
        *,
        command: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        prepared_payload: Any,
        request_path: str,
        start: float,
        upstream_request_policy: UpstreamRequestPolicy | None = None,
    ) -> UpstreamResponse: ...


class HttpxUpstreamClient(UpstreamClient):
    def __init__(self, on_first_token: Callable[[], None] | None = None) -> None:
        self._on_first_token = on_first_token

    def fetch(
        self,
        *,
        command: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        prepared_payload: Any,
        request_path: str,
        start: float,
        upstream_request_policy: UpstreamRequestPolicy | None = None,
    ) -> UpstreamResponse:
        last_response: UpstreamResponse | None = None
        max_empty_retries = (
            upstream_request_policy.empty_response_retries
            if upstream_request_policy is not None
            else DEFAULT_EMPTY_RESPONSE_RETRIES
        )
        max_rate_limit_retries = (
            upstream_request_policy.rate_limit_retries
            if upstream_request_policy is not None
            else DEFAULT_RATE_LIMIT_RETRIES
        )
        with httpx.Client(timeout=_UPSTREAM_TIMEOUT) as client:
            for rate_attempt in range(max_rate_limit_retries):
                for empty_attempt in range(max_empty_retries):
                    if _should_stream_chat_completion(command, request_path, prepared_payload):
                        response = self._fetch_streamed(
                            client=client,
                            url=url,
                            headers=headers,
                            payload=prepared_payload,
                            start=start,
                        )
                    else:
                        response = self._fetch_direct(
                            client=client,
                            command=command,
                            url=url,
                            headers=headers,
                            body=body,
                        )
                    response = _normalize_upstream_response(response)
                    last_response = response
                    payload = response.payload if isinstance(response.payload, dict) else {}
                    if _upstream_response_is_retryable_rate_limit(response):
                        break
                    if not payload_has_retryable_empty_content(payload):
                        return response
                    if empty_attempt + 1 < max_empty_retries:
                        log.debug(
                            "Retrying empty upstream chat completion (attempt %s/%s)",
                            empty_attempt + 2,
                            max_empty_retries,
                        )
                        time.sleep(min(2.0, 1.5 ** empty_attempt))
                if (
                    last_response is not None
                    and _upstream_response_is_retryable_rate_limit(last_response)
                    and rate_attempt + 1 < max_rate_limit_retries
                ):
                    backoff = _rate_limit_backoff_seconds(last_response, rate_attempt)
                    log.info(
                        "Retrying upstream rate limit (attempt %s/%s) after %.2fs",
                        rate_attempt + 2,
                        max_rate_limit_retries,
                        backoff,
                    )
                    time.sleep(backoff)
                    continue
                break
        if last_response is not None:
            return last_response
        return UpstreamResponse(
            body=b"{}",
            payload={},
            status=502,
            headers=httpx.Headers({"Content-Type": "application/json"}),
            first_token_latency_ms=None,
        )

    def _fetch_direct(
        self,
        *,
        client: httpx.Client,
        command: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
    ) -> UpstreamResponse:
        response = client.request(command, url, headers=headers, content=body)
        return UpstreamResponse(
            body=response.content,
            payload=_loads_json_bytes(response.content),
            status=response.status_code,
            headers=response.headers,
            first_token_latency_ms=None,
        )

    def _fetch_streamed(
        self,
        *,
        client: httpx.Client,
        url: str,
        headers: dict[str, str],
        payload: dict[str, Any],
        start: float,
    ) -> UpstreamResponse:
        stream_payload = dict(payload)
        stream_payload["stream"] = True
        stream_options = dict(stream_payload.get("stream_options") or {})
        stream_options["include_usage"] = True
        stream_payload["stream_options"] = stream_options

        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        role = "assistant"
        response_id: str | None = None
        response_model: str | None = None
        created: int | None = None
        finish_reason: str | None = None
        native_finish_reason: str | None = None
        usage: dict[str, Any] | None = None
        first_token_latency_ms: int | None = None

        stream_headers = {
            key: value
            for key, value in headers.items()
            if key.lower() not in {"content-length", "content-type"}
        }
        stream_headers["Content-Type"] = "application/json"

        with client.stream("POST", url, headers=stream_headers, json=stream_payload) as response:
            if response.status_code >= 400:
                response_body = response.read()
                return UpstreamResponse(
                    body=response_body,
                    payload=_loads_json_bytes(response_body),
                    status=response.status_code,
                    headers=response.headers,
                    first_token_latency_ms=None,
                )
            for line in response.iter_lines():
                data = _sse_data_from_line(line)
                if data is None:
                    continue
                if data == "[DONE]":
                    break
                chunk = _loads_json_text(data)
                if not isinstance(chunk, dict):
                    continue
                response_id = str(chunk.get("id") or response_id or "")
                response_model = str(chunk.get("model") or response_model or "")
                if chunk.get("created") is not None:
                    try:
                        created = int(chunk["created"])
                    except (TypeError, ValueError):
                        pass
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    if not isinstance(choice, dict):
                        continue
                    delta = tau.utils.get_dict(choice, "delta")
                    message = tau.utils.get_dict(choice, "message")
                    if delta.get("role"):
                        role = str(delta["role"])
                    elif message.get("role"):
                        role = str(message["role"])
                    token_seen = False
                    content = delta.get("content", message.get("content"))
                    if content is not None and content != "":
                        if append_stream_text_part(content_parts, content):
                            token_seen = True
                    reasoning = (
                        delta.get("reasoning")
                        or delta.get("reasoning_content")
                        or message.get("reasoning")
                        or message.get("reasoning_content")
                    )
                    if reasoning is not None and reasoning != "":
                        if append_stream_text_part(reasoning_parts, reasoning):
                            token_seen = True
                    if delta.get("tool_calls") or message.get("tool_calls"):
                        merge_stream_tool_call_delta(tool_calls_by_index, delta.get("tool_calls"))
                        merge_stream_tool_call_delta(tool_calls_by_index, message.get("tool_calls"))
                        token_seen = True
                    if token_seen and first_token_latency_ms is None:
                        first_token_latency_ms = int((time.monotonic() - start) * 1000)
                        if self._on_first_token is not None:
                            self._on_first_token()
                    finish_reason = choice.get("finish_reason") or finish_reason
                    native_finish_reason = (
                        choice.get("native_finish_reason") or native_finish_reason
                    )

        built_message: dict[str, Any] = {"role": role, "content": "".join(content_parts)}
        if tool_calls_by_index:
            built_message["tool_calls"] = [
                tool_calls_by_index[index] for index in sorted(tool_calls_by_index)
            ]
        if reasoning_parts:
            built_message["reasoning"] = "".join(reasoning_parts)
        built_payload: dict[str, Any] = {
            "id": response_id or f"chatcmpl-proxy-{int(time.time() * 1000)}",
            "object": "chat.completion",
            "created": created or int(time.time()),
            "model": response_model or str(payload.get("model") or ""),
            "choices": [
                {
                    "index": 0,
                    "message": built_message,
                    "finish_reason": finish_reason,
                },
            ],
            "usage": usage or {},
        }
        if native_finish_reason is not None:
            built_payload["choices"][0]["native_finish_reason"] = native_finish_reason
        built_body = json.dumps(built_payload).encode("utf-8")
        return UpstreamResponse(
            body=built_body,
            payload=built_payload,
            status=200,
            headers=httpx.Headers({"Content-Type": "application/json"}),
            first_token_latency_ms=first_token_latency_ms,
        )


class CachedUpstreamClient(UpstreamClient):
    """Wraps an UpstreamClient with disk-backed record+replay caching.

    With inner=None operates in replay-only mode: raises CacheMissError on a
    miss. With an inner client records the response to disk on miss so the next
    call is served from cache.
    """

    def __init__(self, cache_dir: Path, inner: UpstreamClient | None = None) -> None:
        self._cache = DiskCache(cache_dir)
        self._inner = inner

    def fetch(
        self,
        *,
        command: str,
        url: str,
        headers: dict[str, str],
        body: bytes | None,
        prepared_payload: Any,
        request_path: str,
        start: float,
        upstream_request_policy: UpstreamRequestPolicy | None = None,
    ) -> UpstreamResponse:
        if isinstance(prepared_payload, dict):
            key = json_sha256({"path": request_path, "body": prepared_payload})
            entry = self._cache.read(key)
            if entry is not None:
                cached_payload = entry.get("body") or {}
                return UpstreamResponse(
                    body=json.dumps(cached_payload).encode("utf-8"),
                    payload=cached_payload,
                    status=int(entry.get("status_code", 200)),
                    headers=httpx.Headers({"Content-Type": "application/json"}),
                    first_token_latency_ms=None,
                )
            if self._inner is None:
                raise CacheMissError(key)
        elif self._inner is None:
            raise CacheMissError("(non-dict payload)")

        response = self._inner.fetch(
            command=command,
            url=url,
            headers=headers,
            body=body,
            prepared_payload=prepared_payload,
            request_path=request_path,
            start=start,
            upstream_request_policy=upstream_request_policy,
        )
        if (
            isinstance(prepared_payload, dict)
            and isinstance(response.payload, dict)
            and response.status < 400
        ):
            self._cache.write(
                key,
                {"status_code": response.status, "body": response.payload},
            )
        return response


@dataclass(slots=True)
class OpenRouterProxy:
    openrouter_api_key: str
    solve_budget: SolveBudget | None = None
    bind_host: str | None = "127.0.0.1"
    bind_port: int = 0
    unix_socket_path: str | None = None
    enforced_model: str | None = None
    enforced_provider: dict[str, Any] | None = None
    enforced_sampling_params: dict[str, Any] | None = field(
        default_factory=lambda: dict(_VALIDATOR_SAMPLING_PARAMS)
    )
    upstream_request_policy: UpstreamRequestPolicy | None = None
    require_auth: bool = True
    auth_token: str = field(default_factory=lambda: secrets.token_urlsafe(24))
    rollout_event_sink: Callable[[dict[str, Any]], None] | None = None
    rollout_capture_bodies: bool = False
    cache_dir: Path | None = None
    cache_replay_only: bool = False
    upstream_base_url: str | None = None
    _server: _ReusableThreadingHTTPServer | None = field(default=None, init=False, repr=False)
    _unix_server: _ReusableThreadingUnixServer | None = field(default=None, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _unix_thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _usage: SolveUsageSummary = field(default_factory=SolveUsageSummary, init=False, repr=False)
    _upstream_client: UpstreamClient = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.upstream_base_url:
            self.upstream_base_url = normalize_base_url(self.upstream_base_url)
        httpx_client = HttpxUpstreamClient(on_first_token=self._record_first_token)
        if self.cache_dir is not None:
            inner = None if self.cache_replay_only else httpx_client
            self._upstream_client = CachedUpstreamClient(self.cache_dir, inner=inner)
        else:
            self._upstream_client = httpx_client

    def start(self) -> None:
        if self._server is not None or self._unix_server is not None:
            return

        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def handle(self) -> None:
                # Override to suppress BrokenPipeError from wfile.flush()
                # in handle_one_request after do_POST returns.
                try:
                    super().handle()
                except BrokenPipeError:
                    pass

            def do_GET(self) -> None:  # noqa: N802
                self._handle()

            def do_POST(self) -> None:  # noqa: N802
                self._handle()

            def address_string(self) -> str:
                # Unix sockets pass a string (often empty) as client_address
                # instead of the (host, port) tuple that the base class expects.
                if isinstance(self.client_address, str):
                    return self.client_address or "unix"
                return super().address_string()

            def log_message(self, format: str, *args: object) -> None:
                log.debug("proxy %s - %s", self.address_string(), format % args)

            def _handle(self) -> None:
                try:
                    proxy._handle_request(self)
                except BrokenPipeError:
                    log.debug("Client disconnected (broken pipe)")
                except Exception:  # noqa: BLE001
                    log.exception("OpenRouter proxy request failed")
                    try:
                        self.send_error(502, "Proxy request failed")
                    except BrokenPipeError:
                        pass

        if self.bind_host is not None:
            self._server = _ReusableThreadingHTTPServer((self.bind_host, self.bind_port), Handler)
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
            log.debug("OpenRouter proxy listening on %s:%s", self.host, self.port)

        if self.unix_socket_path:
            socket_dir = os.path.dirname(self.unix_socket_path)
            if socket_dir:
                os.makedirs(socket_dir, exist_ok=True)
            if os.path.exists(self.unix_socket_path):
                os.unlink(self.unix_socket_path)
            self._unix_server = _ReusableThreadingUnixServer(self.unix_socket_path, Handler)
            # Make socket world-writable so Docker containers (which drop
            # CAP_DAC_OVERRIDE) can connect from any UID.
            os.chmod(self.unix_socket_path, 0o777)
            self._unix_thread = threading.Thread(
                target=self._unix_server.serve_forever, daemon=True
            )
            self._unix_thread.start()
            log.debug("OpenRouter proxy listening on unix socket %s", self.unix_socket_path)

    def stop(self) -> None:
        if self._server is None and self._unix_server is None:
            return

        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._unix_server is not None:
            self._unix_server.shutdown()
            self._unix_server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        if self._unix_thread is not None:
            self._unix_thread.join(timeout=5)
        self._server = None
        self._unix_server = None
        self._thread = None
        self._unix_thread = None
        if self.unix_socket_path and os.path.exists(self.unix_socket_path):
            os.unlink(self.unix_socket_path)

    def __enter__(self) -> OpenRouterProxy:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @property
    def host(self) -> str:
        if self._server is None:
            raise RuntimeError("Proxy server is not running")
        # ThreadingHTTPServer always binds a TCP (AF_INET/AF_INET6) address,
        # so server_address[0] is always str despite the broad BaseServer stub.
        return cast(str, self._server.server_address[0])

    @property
    def port(self) -> int:
        if self._server is None:
            raise RuntimeError("Proxy server is not running")
        return int(self._server.server_address[1])

    def container_base_url(self, host_name: str) -> str:
        return f"http://{host_name}:{self.port}/v1"

    def usage_snapshot(self) -> SolveUsageSummary:
        with self._lock:
            return self._usage.snapshot()

    @property
    def budget_exceeded_reason(self) -> str | None:
        with self._lock:
            return self._usage.budget_exceeded_reason

    def _handle_request(self, handler: BaseHTTPRequestHandler) -> None:
        if handler.command == "HEAD":
            handler.send_response(200)
            handler.send_header("Content-Length", "0")
            handler.send_header("Connection", "close")
            handler.end_headers()
            handler.close_connection = True
            return
        if handler.command not in _ALLOWED_METHODS:
            self._reject_request(
                handler,
                reason=PROXY_ERROR_EXIT_REASON,
                status=405,
                error_type="proxy_policy_violation",
                message="Method not allowed",
                method=handler.command,
                path=handler.path,
                request_model=None,
            )
            return
        request_path = handler.path.split("?", 1)[0]
        if request_path not in _ALLOWED_PATHS:
            self._reject_request(
                handler,
                reason=PROXY_ERROR_EXIT_REASON,
                status=403,
                error_type="proxy_policy_violation",
                message="Endpoint not allowed",
                method=handler.command,
                path=handler.path,
                request_model=None,
            )
            return

        if self.require_auth:
            expected_auth = f"Bearer {self.auth_token}"
            auth_header = handler.headers.get("Authorization")
            api_key_header = handler.headers.get("x-api-key")
            if auth_header != expected_auth and api_key_header != self.auth_token:
                self._reject_request(
                    handler,
                    reason=PROXY_ERROR_EXIT_REASON,
                    status=401,
                    error_type="proxy_policy_violation",
                    message="Unauthorized",
                    method=handler.command,
                    path=handler.path,
                    request_model=None,
                )
                return

        try:
            content_length = int(handler.headers.get("Content-Length", "0") or "0")
        except ValueError:
            self._reject_request(
                handler,
                reason=PROXY_ERROR_EXIT_REASON,
                status=400,
                error_type="proxy_policy_violation",
                message="Invalid Content-Length header",
                method=handler.command,
                path=handler.path,
                request_model=None,
            )
            return
        if content_length < 0:
            self._reject_request(
                handler,
                reason=PROXY_ERROR_EXIT_REASON,
                status=400,
                error_type="proxy_policy_violation",
                message="Invalid Content-Length header",
                method=handler.command,
                path=handler.path,
                request_model=None,
            )
            return
        if content_length > _MAX_REQUEST_BODY_BYTES:
            self._reject_request(
                handler,
                reason=PROXY_ERROR_EXIT_REASON,
                status=413,
                error_type="proxy_policy_violation",
                message="Request body too large",
                method=handler.command,
                path=handler.path,
                request_model=None,
            )
            return
        body = handler.rfile.read(content_length) if content_length > 0 else None
        request_payload = _loads_json_bytes(body)
        if content_length > 0 and not isinstance(request_payload, dict):
            self._reject_request(
                handler,
                reason=PROXY_ERROR_EXIT_REASON,
                status=400,
                error_type="proxy_policy_violation",
                message="Request body must be a JSON object",
                method=handler.command,
                path=handler.path,
                request_model=None,
            )
            return
        request_model = _extract_request_model(request_payload)
        if (not request_model and not self.enforced_model) or not _request_payload_has_messages(
            request_payload
        ):
            self._reject_request(
                handler,
                reason=PROXY_ERROR_EXIT_REASON,
                status=400,
                error_type="proxy_policy_violation",
                message="Request body must include model and messages",
                method=handler.command,
                path=handler.path,
                request_model=request_model,
            )
            return
        body, rejection_reason = self._prepare_request_body(
            body=body, request_payload=request_payload
        )
        if rejection_reason:
            self._reject_request(
                handler,
                reason=rejection_reason,
                status=429,
                error_type="budget_exceeded",
                message="Solve budget exceeded",
                method=handler.command,
                path=handler.path,
                request_model=request_model,
            )
            return
        prepared_payload = _loads_json_bytes(body)
        if isinstance(prepared_payload, dict):
            request_model = _extract_request_model(prepared_payload) or request_model

        started_at = utc_now()
        start = time.monotonic()

        try:
            upstream = self._upstream_client.fetch(
                command=handler.command,
                url=f"{self._upstream_base_url()}{handler.path}",
                headers=self._build_upstream_headers(handler),
                body=body,
                prepared_payload=prepared_payload,
                request_path=request_path,
                start=start,
                upstream_request_policy=self.upstream_request_policy,
            )
        except CacheMissError:
            error_body = json.dumps(
                {
                    "error": {
                        "message": "No cached response available for this request",
                        "type": "proxy_cache_miss",
                        "code": PROXY_ERROR_EXIT_REASON,
                    },
                }
            ).encode("utf-8")
            with self._lock:
                self._usage.request_count -= 1  # undo the slot claimed in _prepare_request_body
                self._usage.rejected_request_count += 1
                self._usage.error_count += 1
                self._usage.requests.append(
                    ProxyRequestRecord(
                        method=handler.command,
                        path=handler.path,
                        status_code=503,
                        latency_ms=0,
                        request_model=request_model,
                        rejected=True,
                        error="proxy_cache_miss",
                    )
                )
            self._send_raw(handler, 503, error_body, content_type="application/json")
            return

        except httpx.HTTPError as exc:
            latency_ms = int((time.monotonic() - start) * 1000)
            finished_at = utc_now()
            request_record = ProxyRequestRecord(
                method=handler.command,
                path=handler.path,
                status_code=None,
                latency_ms=latency_ms,
                request_model=request_model,
                error=str(exc),
            )
            self._record_request(request_record)
            self._emit_rollout_llm_event(
                method=handler.command,
                path=handler.path,
                request_payload=prepared_payload if self.rollout_capture_bodies else None,
                response_payload={
                    "error": {"message": str(exc), "type": "upstream_transport_error"}
                },
                request_record=request_record,
                started_at=started_at,
                finished_at=finished_at,
            )
            raise

        latency_ms = int((time.monotonic() - start) * 1000)
        finished_at = utc_now()
        request_record = ProxyRequestRecord(
            method=handler.command,
            path=handler.path,
            status_code=upstream.status,
            latency_ms=latency_ms,
            request_model=request_model,
            response_model=_extract_response_model(upstream.payload),
            generation_id=_extract_generation_id(upstream.payload),
            first_token_latency_ms=upstream.first_token_latency_ms,
            prompt_tokens=_extract_prompt_tokens(upstream.payload),
            completion_tokens=_extract_completion_tokens(upstream.payload),
            total_tokens=_extract_total_tokens(upstream.payload),
            cached_tokens=_extract_cached_tokens(upstream.payload),
            cache_write_tokens=_extract_cache_write_tokens(upstream.payload),
            reasoning_tokens=_extract_reasoning_tokens(upstream.payload),
            cost=_extract_cost(upstream.payload),
            error=_extract_response_error(upstream.payload) if upstream.status >= 400 else None,
        )
        self._record_request(request_record)
        self._emit_rollout_llm_event(
            method=handler.command,
            path=handler.path,
            request_payload=prepared_payload if self.rollout_capture_bodies else None,
            response_payload=upstream.payload if self.rollout_capture_bodies else None,
            request_record=request_record,
            started_at=started_at,
            finished_at=finished_at,
        )

        handler.send_response(upstream.status)
        for key, value in upstream.headers.items():
            if key.lower() in _HOP_BY_HOP_HEADERS or key.lower() == "content-length":
                continue
            handler.send_header(key, value)
        handler.send_header("Content-Length", str(len(upstream.body)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(upstream.body)
        handler.wfile.flush()
        handler.close_connection = True

    def _upstream_base_url(self) -> str:
        return self.upstream_base_url or _upstream_base_url()

    @staticmethod
    def _send_raw(
        handler: BaseHTTPRequestHandler,
        status: int,
        body: bytes,
        *,
        content_type: str,
    ) -> None:
        handler.send_response(status)
        handler.send_header("Content-Type", content_type)
        handler.send_header("Content-Length", str(len(body)))
        handler.send_header("Connection", "close")
        handler.end_headers()
        handler.wfile.write(body)
        handler.wfile.flush()
        handler.close_connection = True

    def _emit_rollout_llm_event(
        self,
        *,
        method: str,
        path: str,
        request_payload: Any,
        response_payload: Any,
        request_record: ProxyRequestRecord,
        started_at: str,
        finished_at: str,
    ) -> None:
        if self.rollout_event_sink is None:
            return
        usage = {
            "prompt_tokens": request_record.prompt_tokens,
            "completion_tokens": request_record.completion_tokens,
            "total_tokens": request_record.total_tokens,
            "cached_tokens": request_record.cached_tokens,
            "cache_write_tokens": request_record.cache_write_tokens,
            "reasoning_tokens": request_record.reasoning_tokens,
        }
        secrets_tuple = tuple(
            value for value in (self.openrouter_api_key, self.auth_token) if value
        )
        event = build_llm_event(
            method=method,
            path=path,
            request_payload=request_payload,
            response_payload=response_payload,
            status_code=request_record.status_code,
            latency_ms=request_record.latency_ms,
            request_model=request_record.request_model,
            response_model=request_record.response_model,
            usage=usage,
            cost=request_record.cost,
            started_at=started_at,
            finished_at=finished_at,
            secrets=secrets_tuple,
        )
        try:
            self.rollout_event_sink(event)
        except Exception:
            log.exception("rollout LLM event sink failed")

    def _build_upstream_headers(self, handler: BaseHTTPRequestHandler) -> dict[str, str]:
        headers: dict[str, str] = {
            "Authorization": f"Bearer {self.openrouter_api_key}",
            "X-Title": "swe-eval",
        }
        for key, value in handler.headers.items():
            lowered = key.lower()
            if lowered in _HOP_BY_HOP_HEADERS or lowered in {"authorization", "x-api-key"}:
                continue
            headers[key] = value
        return headers

    def _prepare_request_body(
        self,
        *,
        body: bytes | None,
        request_payload: Any,
    ) -> tuple[bytes | None, str | None]:
        if isinstance(request_payload, dict) and self.enforced_model:
            request_payload["model"] = self.enforced_model
            body = json.dumps(request_payload).encode("utf-8")
        if isinstance(request_payload, dict) and self.enforced_provider is not None:
            request_payload["provider"] = dict(self.enforced_provider)
            body = json.dumps(request_payload).encode("utf-8")
        if isinstance(request_payload, dict) and self.enforced_sampling_params is not None:
            for key in _MINER_CONTROLLED_SAMPLING_PARAMS:
                request_payload.pop(key, None)
            request_payload.update(self.enforced_sampling_params)
            body = json.dumps(request_payload).encode("utf-8")
        if isinstance(request_payload, dict):
            if self.upstream_request_policy is not None:
                apply_upstream_request_policy(request_payload, self.upstream_request_policy)
            body = json.dumps(request_payload).encode("utf-8")
        if not self.solve_budget or not self.solve_budget.enabled():
            with self._lock:
                if self._usage.budget_exceeded_reason:
                    return body, self._usage.budget_exceeded_reason
                self._check_request_limit_locked()
                if self._usage.budget_exceeded_reason:
                    return body, self._usage.budget_exceeded_reason
                self._usage.request_count += 1
            return body, None

        if not isinstance(request_payload, dict):
            with self._lock:
                self._check_pre_request_budget_locked()
                if self._usage.budget_exceeded_reason:
                    return body, self._usage.budget_exceeded_reason
                self._check_request_limit_locked()
                if self._usage.budget_exceeded_reason:
                    return body, self._usage.budget_exceeded_reason
                self._usage.request_count += 1
            return body, None

        estimated_prompt_tokens = _estimate_prompt_tokens(request_payload)
        with self._lock:
            self._check_pre_request_budget_locked()
            if self._usage.budget_exceeded_reason:
                return body, self._usage.budget_exceeded_reason
            self._check_request_limit_locked()
            if self._usage.budget_exceeded_reason:
                return body, self._usage.budget_exceeded_reason
            self._check_estimated_request_budget_locked(
                estimated_prompt_tokens=estimated_prompt_tokens,
                request_payload=request_payload,
            )
            if self._usage.budget_exceeded_reason:
                return body, self._usage.budget_exceeded_reason
            self._clamp_request_tokens_locked(
                request_payload=request_payload,
                estimated_prompt_tokens=estimated_prompt_tokens,
            )
            if self._usage.budget_exceeded_reason:
                return body, self._usage.budget_exceeded_reason
            self._usage.request_count += 1

        return json.dumps(request_payload).encode("utf-8"), None

    def _check_pre_request_budget_locked(self) -> None:
        if self._usage.budget_exceeded_reason:
            return
        if not self.solve_budget:
            return
        if (
            self.solve_budget.max_cost is not None
            and self._usage.cost >= self.solve_budget.max_cost
        ):
            self._usage.budget_exceeded_reason = COST_LIMIT_EXIT_REASON
            return
        if (
            self.solve_budget.max_total_tokens is not None
            and self._usage.total_tokens >= self.solve_budget.max_total_tokens
        ):
            self._usage.budget_exceeded_reason = TOKEN_LIMIT_EXIT_REASON
            return
        if (
            self.solve_budget.max_prompt_tokens is not None
            and self._usage.prompt_tokens >= self.solve_budget.max_prompt_tokens
        ):
            self._usage.budget_exceeded_reason = TOKEN_LIMIT_EXIT_REASON
            return
        if (
            self.solve_budget.max_completion_tokens is not None
            and self._usage.completion_tokens >= self.solve_budget.max_completion_tokens
        ):
            self._usage.budget_exceeded_reason = TOKEN_LIMIT_EXIT_REASON

    def _check_request_limit_locked(self) -> None:
        if self._usage.budget_exceeded_reason or not self.solve_budget:
            return
        if (
            self.solve_budget.max_requests is not None
            and self._usage.request_count >= self.solve_budget.max_requests
        ):
            self._usage.budget_exceeded_reason = REQUEST_LIMIT_EXIT_REASON

    def _check_estimated_request_budget_locked(
        self,
        *,
        estimated_prompt_tokens: int,
        request_payload: dict[str, Any],
    ) -> None:
        if not self.solve_budget or self._usage.budget_exceeded_reason:
            return
        remaining_prompt_tokens = None
        if self.solve_budget.max_prompt_tokens is not None:
            remaining_prompt_tokens = max(
                0, self.solve_budget.max_prompt_tokens - self._usage.prompt_tokens
            )
            if estimated_prompt_tokens > remaining_prompt_tokens:
                self._usage.budget_exceeded_reason = TOKEN_LIMIT_EXIT_REASON
                return

        remaining_total_tokens = None
        if self.solve_budget.max_total_tokens is not None:
            remaining_total_tokens = max(
                0, self.solve_budget.max_total_tokens - self._usage.total_tokens
            )
            if estimated_prompt_tokens >= remaining_total_tokens:
                self._usage.budget_exceeded_reason = TOKEN_LIMIT_EXIT_REASON
                return

        requested_max_output_tokens = _extract_requested_max_output_tokens(request_payload)
        if requested_max_output_tokens is None or remaining_total_tokens is None:
            return
        if estimated_prompt_tokens + requested_max_output_tokens > remaining_total_tokens:
            log.debug(
                "Estimated request would exceed total token budget prompt_estimate=%s requested_max_output=%s remaining_total=%s",
                estimated_prompt_tokens,
                requested_max_output_tokens,
                remaining_total_tokens,
            )

    def _clamp_request_tokens_locked(
        self,
        *,
        request_payload: dict[str, Any],
        estimated_prompt_tokens: int,
    ) -> None:
        if not self.solve_budget:
            return
        limits: list[int] = []
        if self.solve_budget.max_tokens_per_request is not None:
            limits.append(self.solve_budget.max_tokens_per_request)
        if self.solve_budget.max_completion_tokens is not None:
            limits.append(
                max(0, self.solve_budget.max_completion_tokens - self._usage.completion_tokens)
            )
        if self.solve_budget.max_total_tokens is not None:
            remaining_total_tokens = max(
                0, self.solve_budget.max_total_tokens - self._usage.total_tokens
            )
            limits.append(max(0, remaining_total_tokens - estimated_prompt_tokens))
        average_cost_per_token = self._average_cost_per_token_locked()
        if (
            self.solve_budget.max_cost is not None
            and average_cost_per_token is not None
            and average_cost_per_token > 0
        ):
            remaining_cost = max(0.0, self.solve_budget.max_cost - self._usage.cost)
            estimated_prompt_cost = estimated_prompt_tokens * average_cost_per_token
            if estimated_prompt_cost >= remaining_cost:
                self._usage.budget_exceeded_reason = COST_LIMIT_EXIT_REASON
                return
            estimated_affordable_output_tokens = int(
                (remaining_cost - estimated_prompt_cost) / average_cost_per_token
            )
            limits.append(max(0, estimated_affordable_output_tokens))
        if not limits:
            return
        allowed_max_tokens = min(limits)
        if allowed_max_tokens <= 0:
            self._usage.budget_exceeded_reason = TOKEN_LIMIT_EXIT_REASON
            return
        _set_requested_max_output_tokens(request_payload, allowed_max_tokens)

    def _average_cost_per_token_locked(self) -> float | None:
        if self._usage.total_tokens <= 0 or self._usage.cost <= 0:
            return None
        return self._usage.cost / self._usage.total_tokens

    def _reject_request(
        self,
        handler: BaseHTTPRequestHandler,
        *,
        reason: str,
        status: int,
        error_type: str,
        message: str,
        method: str,
        path: str,
        request_model: str | None,
    ) -> None:
        request_record = ProxyRequestRecord(
            method=method,
            path=path,
            status_code=status,
            latency_ms=0,
            request_model=request_model,
            rejected=True,
            error=reason,
        )
        with self._lock:
            self._usage.budget_exceeded_reason = reason
            self._usage.rejected_request_count += 1
            self._usage.requests.append(request_record)
        response_payload = {
            "error": {
                "message": message,
                "type": error_type,
                "code": reason,
            },
        }
        now = utc_now()
        self._emit_rollout_llm_event(
            method=method,
            path=path,
            request_payload=None,
            response_payload=response_payload,
            request_record=request_record,
            started_at=now,
            finished_at=now,
        )
        self._send_raw(
            handler,
            status,
            json.dumps(response_payload).encode("utf-8"),
            content_type="application/json",
        )

    def _record_request(self, request: ProxyRequestRecord) -> None:
        with self._lock:
            self._usage.requests.append(request)
            if (
                request.status_code is not None
                and request.status_code < 400
                and request.error is None
            ):
                self._usage.success_count += 1
            else:
                self._usage.error_count += 1
            self._usage.prompt_tokens += int(request.prompt_tokens or 0)
            self._usage.completion_tokens += int(request.completion_tokens or 0)
            self._usage.total_tokens += int(request.total_tokens or 0)
            self._usage.cached_tokens += int(request.cached_tokens or 0)
            self._usage.cache_write_tokens += int(request.cache_write_tokens or 0)
            self._usage.reasoning_tokens += int(request.reasoning_tokens or 0)
            self._usage.cost += float(request.cost or 0.0)
            self._check_pre_request_budget_locked()

    def _record_first_token(self) -> None:
        with self._lock:
            self._usage.first_token_count += 1


def _normalize_upstream_response(response: UpstreamResponse) -> UpstreamResponse:
    payload = response.payload
    if not isinstance(payload, dict):
        return response
    normalized = normalize_chat_completion_payload(payload)
    return UpstreamResponse(
        body=json.dumps(normalized).encode("utf-8"),
        payload=normalized,
        status=response.status,
        headers=response.headers,
        first_token_latency_ms=response.first_token_latency_ms,
    )


def _loads_json_bytes(raw_body: bytes | None) -> Any:
    if not raw_body:
        return None
    try:
        return json.loads(raw_body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _loads_json_text(raw_text: str) -> Any:
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return None


def _sse_data_from_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or stripped.startswith(":"):
        return None
    if not stripped.startswith("data:"):
        return None
    return stripped[len("data:") :].strip()


def _should_stream_chat_completion(command: str, request_path: str, payload: Any) -> bool:
    if command != "POST" or request_path != "/v1/chat/completions":
        return False
    if not isinstance(payload, dict):
        return False
    # Miner agents get a normal non-streaming response, while the validator
    # proxy measures first-token latency from an upstream stream.
    return not bool(payload.get("stream"))


def _extract_request_model(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    return str(model) if isinstance(model, str) else None


def _extract_response_model(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    model = payload.get("model")
    return str(model) if isinstance(model, str) else None


def _extract_generation_id(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    generation_id = payload.get("id")
    return str(generation_id) if isinstance(generation_id, str) else None


def _extract_usage(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    usage = payload.get("usage")
    return usage if isinstance(usage, dict) else None


def _extract_prompt_tokens(payload: Any) -> int | None:
    usage = _extract_usage(payload)
    if not usage:
        return None
    for key in ("prompt_tokens", "input_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def _extract_completion_tokens(payload: Any) -> int | None:
    usage = _extract_usage(payload)
    if not usage:
        return None
    for key in ("completion_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def _extract_total_tokens(payload: Any) -> int | None:
    usage = _extract_usage(payload)
    if not usage:
        return None
    total_tokens = usage.get("total_tokens")
    if isinstance(total_tokens, int):
        return total_tokens
    prompt_tokens = _extract_prompt_tokens_from_usage(usage)
    completion_tokens = _extract_completion_tokens_from_usage(usage)
    if prompt_tokens is not None or completion_tokens is not None:
        return int(prompt_tokens or 0) + int(completion_tokens or 0)
    return None


def _extract_prompt_tokens_from_usage(usage: dict[str, Any]) -> int | None:
    for key in ("prompt_tokens", "input_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def _extract_completion_tokens_from_usage(usage: dict[str, Any]) -> int | None:
    for key in ("completion_tokens", "output_tokens"):
        value = usage.get(key)
        if isinstance(value, int):
            return value
    return None


def _extract_cached_tokens(payload: Any) -> int | None:
    usage = _extract_usage(payload)
    if not usage:
        return None
    cache_read = usage.get("cache_read_input_tokens")
    if isinstance(cache_read, int):
        return cache_read
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("cached_tokens")
    return value if isinstance(value, int) else None


def _extract_cache_write_tokens(payload: Any) -> int | None:
    usage = _extract_usage(payload)
    if not usage:
        return None
    cache_creation = usage.get("cache_creation_input_tokens")
    if isinstance(cache_creation, int):
        return cache_creation
    details = usage.get("prompt_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("cache_write_tokens")
    return value if isinstance(value, int) else None


def _extract_reasoning_tokens(payload: Any) -> int | None:
    usage = _extract_usage(payload)
    if not usage:
        return None
    details = usage.get("completion_tokens_details")
    if not isinstance(details, dict):
        return None
    value = details.get("reasoning_tokens")
    return value if isinstance(value, int) else None


def _extract_response_error(payload: Any) -> str | None:
    if not isinstance(payload, dict):
        return None
    error = payload.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("code") or error.get("type")
        return str(message) if message else None
    if isinstance(error, str):
        return error
    message = payload.get("message")
    return str(message) if message else None


def _upstream_response_is_retryable_rate_limit(response: UpstreamResponse) -> bool:
    if response.status != 429:
        return False
    error_text = str(_extract_response_error(response.payload) or "").lower()
    return (
        "rate limit" in error_text
        or "too many requests" in error_text
        or "high demand" in error_text
        or not error_text
    )


def _rate_limit_backoff_seconds(response: UpstreamResponse, attempt: int) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after:
        try:
            return min(30.0, max(0.25, float(retry_after)))
        except ValueError:
            pass
    return min(30.0, 2.0 * (1.5 ** attempt))


def _extract_cost(payload: Any) -> float | None:
    usage = _extract_usage(payload)
    if not usage:
        return None
    value = usage.get("cost")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _request_payload_has_messages(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    messages = payload.get("messages")
    return isinstance(messages, list) and len(messages) > 0


def _estimate_prompt_tokens(payload: dict[str, Any]) -> int:
    message_count = 0
    total_chars = 0
    messages = payload.get("messages")
    if isinstance(messages, list):
        message_count = len(messages)
        total_chars += sum(_estimate_content_chars(message) for message in messages)
    tools = payload.get("tools")
    if isinstance(tools, list):
        total_chars += sum(len(json.dumps(tool, sort_keys=True)) for tool in tools)
    response_format = payload.get("response_format")
    if response_format is not None:
        total_chars += len(json.dumps(response_format, sort_keys=True))
    prompt_tokens = (total_chars + (_ESTIMATED_CHARS_PER_TOKEN - 1)) // _ESTIMATED_CHARS_PER_TOKEN
    prompt_tokens += message_count * _ESTIMATED_MESSAGE_OVERHEAD_TOKENS
    if isinstance(tools, list):
        prompt_tokens += len(tools) * _ESTIMATED_TOOL_OVERHEAD_TOKENS
    return max(prompt_tokens, 1)


def _estimate_content_chars(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return len(value)
    if isinstance(value, (int, float, bool)):
        return len(str(value))
    if isinstance(value, dict):
        total = 0
        for key, item in value.items():
            total += len(str(key))
            total += _estimate_content_chars(item)
        return total
    if isinstance(value, list):
        return sum(_estimate_content_chars(item) for item in value)
    return len(str(value))


def _extract_requested_max_output_tokens(payload: dict[str, Any]) -> int | None:
    for key in ("max_tokens", "max_completion_tokens"):
        value = payload.get(key)
        if isinstance(value, int):
            return max(value, 0)
    return None


def _set_requested_max_output_tokens(payload: dict[str, Any], value: int) -> None:
    clamped_value = max(value, 0)
    existing_max_tokens = payload.get("max_tokens")
    if isinstance(existing_max_tokens, int):
        payload["max_tokens"] = min(existing_max_tokens, clamped_value)
    else:
        payload["max_tokens"] = clamped_value

    existing_max_completion_tokens = payload.get("max_completion_tokens")
    if isinstance(existing_max_completion_tokens, int):
        payload["max_completion_tokens"] = min(existing_max_completion_tokens, clamped_value)
