from __future__ import annotations

import base64
import hashlib
import logging
import subprocess
import threading
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)

class GitHubClient(ABC):
    @abstractmethod
    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def get(self, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def post(self, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def put(self, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def patch(self, url: str, **kwargs: Any) -> httpx.Response: ...
    @abstractmethod
    def delete(self, url: str, **kwargs: Any) -> httpx.Response: ...


class GitHubAuthRotatingClient(GitHubClient):
    """Small GitHub client wrapper with token rotation and 401 blacklisting."""

    def __init__(
        self,
        *,
        base_headers: dict[str, str],
        timeout: float,
        tokens: Sequence[str],
        rotate: bool,
        user_agent: str,
    ) -> None:
        self._client = httpx.Client(
            base_url="https://api.github.com",
            headers=base_headers,
            follow_redirects=True,
            timeout=timeout,
        )
        self._tokens = _dedupe_preserve_order([token for token in tokens if token])
        self._rotate = rotate
        self._user_agent = user_agent
        self._lock = threading.Lock()
        self._next_index = 0
        self._disabled_indexes: set[int] = set()
        self._all_tokens_disabled_logged = False

    def close(self) -> None:
        self._client.close()

    def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        attempts = self._token_attempts()
        last_response: httpx.Response | None = None
        for token_index, token in attempts:
            response = self._client.request(
                method,
                url,
                **self._request_kwargs_with_token(kwargs, token),
            )
            last_response = response
            if response.status_code == 401 and token_index is not None:
                self._disable_token(token_index)
                continue
            if token_index is not None and self._rotate:
                self._mark_success(token_index)
            return response
        if last_response is None:
            raise RuntimeError("GitHub client made no request")
        return last_response

    def get(self, url: str, **kwargs) -> httpx.Response:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs) -> httpx.Response:
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs) -> httpx.Response:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)

    def github_cache_namespace(self) -> str:
        attempts = self._token_attempts()
        token_index, token = attempts[0]
        if token_index is None or not token:
            return f"{self._user_agent}:unauthenticated"
        return f"{self._user_agent}:token:{_token_fingerprint(token)}"

    def _token_attempts(self) -> list[tuple[int | None, str | None]]:
        with self._lock:
            active_indexes = [idx for idx in range(len(self._tokens)) if idx not in self._disabled_indexes]
            if not active_indexes:
                if self._tokens and not self._all_tokens_disabled_logged:
                    self._all_tokens_disabled_logged = True
                    log.error(
                        "GitHub client %s exhausted all configured auth tokens after HTTP 401 responses; "
                        "falling back to unauthenticated requests",
                        self._user_agent,
                    )
                return [(None, None)]
            self._all_tokens_disabled_logged = False
            if not self._rotate:
                return [(idx, self._tokens[idx]) for idx in active_indexes]
            attempts: list[tuple[int | None, str | None]] = []
            for offset in range(len(self._tokens)):
                idx = (self._next_index + offset) % len(self._tokens)
                if idx in self._disabled_indexes:
                    continue
                attempts.append((idx, self._tokens[idx]))
            return attempts

    def _request_kwargs_with_token(self, kwargs: dict[str, Any], token: str | None) -> dict[str, Any]:
        request_kwargs = dict(kwargs)
        headers = dict(request_kwargs.get("headers") or {})
        if token:
            headers["Authorization"] = f"Bearer {token}"
        else:
            headers.pop("Authorization", None)
        request_kwargs["headers"] = headers
        return request_kwargs

    def _disable_token(self, token_index: int) -> None:
        with self._lock:
            if token_index in self._disabled_indexes:
                return
            self._disabled_indexes.add(token_index)
            remaining = len(self._tokens) - len(self._disabled_indexes)
            fingerprint = _token_fingerprint(self._tokens[token_index])
        log.warning(
            "GitHub client %s permanently blacklisted token #%d (%s) after HTTP 401; %d token(s) remain",
            self._user_agent,
            token_index + 1,
            fingerprint,
            remaining,
        )

    def _mark_success(self, token_index: int) -> None:
        with self._lock:
            self._next_index = (token_index + 1) % max(1, len(self._tokens))


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:10]


class LocalGitHubClient(GitHubClient):
    """Offline GitHubClient backed by a local git clone.

    Answers the read-only REST endpoints validate.py calls (repo metadata, branch
    head, commit resolve, compare, file contents) by shelling out to git, so a
    dry-run needs no network. `repos` maps owner/name to a local path. Write verbs
    raise, so a dry-run can't silently attempt a publish.
    """

    def __init__(self, repos: dict[str, str | Path]) -> None:
        self._repos = {name: Path(path) for name, path in repos.items()}

    def request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        if method.upper() != "GET":
            raise RuntimeError(
                f"LocalGitHubClient is read-only; refusing {method} {url} in dry-run"
            )
        return self.get(url, **kwargs)

    def get(self, url: str, **kwargs: Any) -> httpx.Response:
        params = kwargs.get("params") or {}
        try:
            return self._route(url, params)
        except _LocalGitHubError as exc:
            return _json_response(exc.status_code, exc.payload)

    def post(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> httpx.Response:
        return self.request("DELETE", url, **kwargs)

    def github_cache_namespace(self) -> str:
        return "local-dry-run"

    def close(self) -> None:
        pass

    def _route(self, url: str, params: dict[str, Any]) -> httpx.Response:
        parts = url.split("?", 1)[0].strip("/").split("/")
        if len(parts) < 3 or parts[0] != "repos":
            return _json_response(404, {"message": "Not Found"})
        repo = f"{parts[1]}/{parts[2]}"
        rest = parts[3:]
        path = self._repos.get(repo)
        if path is None:
            return _json_response(404, {"message": f"repo {repo} not found"})

        if not rest:
            return _json_response(200, {"full_name": repo, "private": False})
        if rest[0] == "branches" and len(rest) >= 2:
            return self._branch(path, "/".join(rest[1:]))
        if rest[0] == "commits" and len(rest) >= 2:
            return self._commit(path, "/".join(rest[1:]))
        if rest[0] == "compare" and len(rest) >= 2:
            return self._compare(path, "/".join(rest[1:]))
        if rest[0] == "contents" and len(rest) >= 2:
            return self._contents(path, "/".join(rest[1:]), str(params.get("ref") or "HEAD"))
        return _json_response(404, {"message": f"unsupported endpoint /{'/'.join(parts)}"})

    def _branch(self, repo_path: Path, branch: str) -> httpx.Response:
        sha = _git(repo_path, ["rev-parse", "--verify", f"{branch}^{{commit}}"])
        if sha is None:
            sha = _git(repo_path, ["rev-parse", "--verify", f"refs/heads/{branch}^{{commit}}"])
        if sha is None:
            return _json_response(404, {"message": f"branch {branch} not found"})
        return _json_response(200, {"name": branch, "commit": {"sha": sha}})

    def _commit(self, repo_path: Path, sha: str) -> httpx.Response:
        full = _git(repo_path, ["rev-parse", "--verify", f"{sha}^{{commit}}"])
        if full is None:
            return _json_response(404, {"message": f"commit {sha} not found"})
        return _json_response(200, {"sha": full})

    def _compare(self, repo_path: Path, spec: str) -> httpx.Response:
        if "..." not in spec:
            return _json_response(404, {"message": f"bad compare spec {spec}"})
        base, head = spec.split("...", 1)
        base_sha = _git(repo_path, ["rev-parse", "--verify", f"{base}^{{commit}}"])
        head_sha = _git(repo_path, ["rev-parse", "--verify", f"{head}^{{commit}}"])
        if base_sha is None or head_sha is None:
            return _json_response(404, {"message": "compare ref not found"})
        if base_sha == head_sha:
            status = "identical"
        elif _git_ok(repo_path, ["merge-base", "--is-ancestor", base_sha, head_sha]):
            status = "ahead"
        else:
            status = "diverged"
        return _json_response(200, {"status": status})

    def _contents(self, repo_path: Path, path: str, ref: str) -> httpx.Response:
        blob = _git_bytes(repo_path, ["show", f"{ref}:{path}"])
        if blob is None:
            return _json_response(404, {"message": f"contents {path}@{ref} not found"})
        sha = _git(repo_path, ["rev-parse", f"{ref}:{path}"]) or ""
        return _json_response(
            200,
            {"path": path, "sha": sha, "encoding": "base64", "content": base64.b64encode(blob).decode("ascii")},
        )


class _LocalGitHubError(Exception):
    def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
        super().__init__(payload.get("message", "error"))
        self.status_code = status_code
        self.payload = payload


def _json_response(status_code: int, payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(status_code, json=payload, request=httpx.Request("GET", "http://local"))


def _git(repo_path: Path, args: Sequence[str]) -> str | None:
    out = _git_bytes(repo_path, args)
    return out.decode("utf-8", "replace").strip() if out is not None else None


def _git_bytes(repo_path: Path, args: Sequence[str]) -> bytes | None:
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        timeout=60,
        check=False,
    )
    return result.stdout if result.returncode == 0 else None


def _git_ok(repo_path: Path, args: Sequence[str]) -> bool:
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args],
        capture_output=True,
        timeout=60,
        check=False,
    )
    return result.returncode == 0
