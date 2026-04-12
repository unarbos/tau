from __future__ import annotations

import json
import logging
import random
import shutil
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from urllib.parse import urlencode

import httpx


_GITHUB_API = "https://api.github.com"
log = logging.getLogger("swe-eval.github_miner")

_RATE_LIMIT_COOLDOWN = 60  # seconds to wait before reusing a rate-limited token


class GitHubTokenRotator:
    """Thread-safe round-robin over multiple GitHub PATs with rate-limit tracking."""

    def __init__(self, tokens: list[str]) -> None:
        if not tokens:
            raise ValueError("GitHubTokenRotator requires at least one token")
        self._tokens = list(tokens)
        self._index = 0
        self._lock = threading.Lock()
        self._cooldowns: dict[int, float] = {}
        log.info("Token rotator initialised with %d token(s)", len(self._tokens))

    @classmethod
    def from_env(cls, multi: str | None = None, single: str | None = None) -> GitHubTokenRotator | None:
        """Build a rotator from GITHUB_TOKENS (comma-separated) or GITHUB_TOKEN."""
        raw = multi or ""
        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if not tokens and single:
            tokens = [single]
        return cls(tokens) if tokens else None

    @property
    def size(self) -> int:
        return len(self._tokens)

    def get_token(self) -> str:
        """Return the next available token. Blocks if all tokens are cooling down."""
        with self._lock:
            n = len(self._tokens)
            now = time.monotonic()

            for _ in range(n):
                idx = self._index % n
                self._index += 1
                ready_at = self._cooldowns.get(idx, 0)
                if now >= ready_at:
                    return self._tokens[idx]

            earliest = min(self._cooldowns.values())
            wait = max(0, earliest - now)

        log.warning("All %d GitHub tokens rate-limited; sleeping %.0fs", n, wait)
        time.sleep(wait)
        return self.get_token()

    def mark_rate_limited(self, token: str) -> None:
        """Put *token* on cooldown so other threads skip it."""
        with self._lock:
            try:
                idx = self._tokens.index(token)
            except ValueError:
                return
            self._cooldowns[idx] = time.monotonic() + _RATE_LIMIT_COOLDOWN
            log.info("Token #%d rate-limited, cooldown %ds (%d/%d on cooldown)",
                     idx, _RATE_LIMIT_COOLDOWN,
                     sum(1 for v in self._cooldowns.values() if v > time.monotonic()),
                     len(self._tokens))


@dataclass(slots=True)
class CommitFile:
    filename: str
    status: str
    additions: int
    deletions: int
    changes: int
    patch: str | None

    @classmethod
    def from_dict(cls, payload: dict) -> "CommitFile":
        return cls(
            filename=str(payload.get("filename") or ""),
            status=str(payload.get("status") or ""),
            additions=int(payload.get("additions") or 0),
            deletions=int(payload.get("deletions") or 0),
            changes=int(payload.get("changes") or 0),
            patch=payload.get("patch"),
        )


@dataclass(slots=True)
class CommitCandidate:
    repo_full_name: str
    repo_clone_url: str
    commit_sha: str
    parent_sha: str
    message: str
    html_url: str
    author_name: str | None
    event_id: str
    files: list[CommitFile]

    @property
    def combined_patch(self) -> str:
        blocks: list[str] = []
        for item in self.files:
            if not item.patch:
                continue
            blocks.append(
                "\n".join(
                    [
                        f"diff --git a/{item.filename} b/{item.filename}",
                        f"--- a/{item.filename}",
                        f"+++ b/{item.filename}",
                        item.patch,
                    ],
                ),
            )
        return "\n".join(blocks).strip()

    @property
    def short_sha(self) -> str:
        return self.commit_sha[:12]

    @property
    def changed_files(self) -> list[str]:
        return [item.filename for item in self.files]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["combined_patch"] = self.combined_patch
        return data

    @classmethod
    def from_dict(cls, payload: dict) -> "CommitCandidate":
        files = payload.get("files") or []
        return cls(
            repo_full_name=str(payload.get("repo_full_name") or ""),
            repo_clone_url=str(payload.get("repo_clone_url") or ""),
            commit_sha=str(payload.get("commit_sha") or ""),
            parent_sha=str(payload.get("parent_sha") or ""),
            message=str(payload.get("message") or ""),
            html_url=str(payload.get("html_url") or ""),
            author_name=payload.get("author_name"),
            event_id=str(payload.get("event_id") or ""),
            files=[CommitFile.from_dict(item) for item in files if isinstance(item, dict)],
        )


_CODE_EXTENSIONS = frozenset({
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs", ".c", ".cpp",
    ".h", ".hpp", ".cs", ".rb", ".php", ".swift", ".kt", ".scala", ".sh",
    ".bash", ".zsh", ".pl", ".pm", ".r", ".lua", ".ex", ".exs", ".erl",
    ".hs", ".ml", ".mli", ".clj", ".cljs", ".vue", ".svelte", ".dart",
    ".zig", ".nim", ".cr", ".v", ".sql", ".m", ".mm",
})

_SKIP_FILENAMES = frozenset({
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "cargo.lock",
    "poetry.lock", "pipfile.lock", "gemfile.lock", "composer.lock",
    "go.sum", "flake.lock",
})

MIN_CODE_CHANGED_LINES = 100
MIN_CODE_FILES = 1


def _is_code_file(filename: str) -> bool:
    """Return True if the file has a recognized code extension."""
    lower = filename.lower()
    for ext in _CODE_EXTENSIONS:
        if lower.endswith(ext):
            return True
    return False


def _is_lockfile(filename: str) -> bool:
    """Return True if the file is a lockfile or auto-generated."""
    base = filename.rsplit("/", 1)[-1].lower()
    return base in _SKIP_FILENAMES


class GitHubMiner:
    """Sample random recent commits from public GitHub push events."""

    def __init__(
        self,
        *,
        token: str | None = None,
        token_rotator: GitHubTokenRotator | None = None,
        rng: random.Random,
        timeout: float = 30.0,
    ) -> None:
        self._rotator = token_rotator
        self._static_token = token
        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "swe-eval",
        }
        if not token_rotator and token:
            headers["Authorization"] = f"Bearer {token}"
        self._rng = rng
        self._client = httpx.Client(
            base_url=_GITHUB_API,
            headers=headers,
            timeout=timeout,
            follow_redirects=True,
        )
        self._use_gh_cli = bool(not token and not token_rotator and shutil.which("gh"))

    def close(self) -> None:
        log.debug("Closing HTTP client")
        self._client.close()

    def sample_commit(self, max_attempts: int = 25) -> CommitCandidate:
        last_error: str | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                log.debug("Mining attempt %s/%s", attempt, max_attempts)
                events = self._recent_push_events()
                if not events:
                    raise ValueError("No recent push events available")
                event = self._rng.choice(events)
                log.debug(
                    "Selected push event id=%s repo=%s created_at=%s",
                    event.get("id"),
                    event.get("repo", {}).get("name"),
                    event.get("created_at"),
                )
                commit_sha = self._pick_random_commit_sha(event)
                candidate = self._fetch_commit_candidate(
                    repo_full_name=event["repo"]["name"],
                    event_id=str(event.get("id", "")),
                    commit_sha=commit_sha,
                )
            except (httpx.HTTPError, KeyError, RuntimeError, ValueError) as exc:
                last_error = str(exc)
                log.debug("Mining attempt %s failed: %s", attempt, exc)
                continue

            if not candidate.combined_patch:
                last_error = "Sampled commit had no textual patch content"
                log.debug("Discarding commit without textual patch content")
                continue

            reject_reason = self._quality_check(candidate)
            if reject_reason:
                last_error = reject_reason
                log.debug("Discarding commit: %s", reject_reason)
                continue

            log.debug(
                "Sampled commit repo=%s sha=%s with %s changed files",
                candidate.repo_full_name,
                candidate.commit_sha,
                len(candidate.files),
            )
            return candidate
        raise RuntimeError(last_error or "Could not sample a usable GitHub commit")

    @staticmethod
    def _quality_check(candidate: CommitCandidate) -> str | None:
        """Return a rejection reason, or None if the commit is acceptable."""
        code_files = [
            f for f in candidate.files
            if f.patch and _is_code_file(f.filename) and not _is_lockfile(f.filename)
        ]
        if len(code_files) < MIN_CODE_FILES:
            return (
                f"Only {len(code_files)} code file(s), need {MIN_CODE_FILES}; "
                f"files: {[f.filename for f in candidate.files]}"
            )

        code_changed = sum(f.additions + f.deletions for f in code_files)
        if code_changed < MIN_CODE_CHANGED_LINES:
            return (
                f"Only {code_changed} code lines changed, need {MIN_CODE_CHANGED_LINES}"
            )

        # Prefer modification-heavy commits: agents produce more similar
        # patches when editing existing code vs writing entirely new files.
        modified_files = [f for f in code_files if f.status == "modified"]
        if not modified_files:
            return (
                f"No modified code files (all {len(code_files)} are added/removed); "
                "need at least 1 modified file for meaningful comparison"
            )

        return None

    def _recent_push_events(self) -> list[dict]:
        log.debug("Fetching recent public events page 1")
        response, payload = self._get_json("/events", page=1, per_page=30, return_response=True)
        events = list(payload)
        pages = self._extract_available_pages(response.headers.get("link", "")) if response else [1]
        valid_pages = [page for page in pages if page > 1]
        if valid_pages:
            page = self._rng.choice(valid_pages)
            log.debug("Fetching additional public events page %s", page)
            events.extend(self._get_json("/events", page=page, per_page=30))
        return [event for event in events if event.get("type") == "PushEvent"]

    def _pick_random_commit_sha(self, event: dict) -> str:
        commits = event.get("payload", {}).get("commits", [])
        if not commits:
            head_sha = event.get("payload", {}).get("head")
            if head_sha:
                return str(head_sha)
            raise ValueError("Push event contained no commits")
        commit = self._rng.choice(commits)
        sha = commit.get("sha")
        if not sha:
            raise ValueError("Push event commit missing sha")
        return sha

    def _fetch_commit_candidate(
        self,
        *,
        repo_full_name: str,
        event_id: str,
        commit_sha: str,
    ) -> CommitCandidate:
        log.debug("Fetching commit metadata for %s@%s", repo_full_name, commit_sha)
        payload = self._get_json(f"/repos/{repo_full_name}/commits/{commit_sha}")
        parents = payload.get("parents") or []
        if not parents:
            raise ValueError("Commit has no parent")

        files: list[CommitFile] = []
        for item in payload.get("files") or []:
            files.append(
                CommitFile(
                    filename=item.get("filename", ""),
                    status=item.get("status", ""),
                    additions=int(item.get("additions", 0)),
                    deletions=int(item.get("deletions", 0)),
                    changes=int(item.get("changes", 0)),
                    patch=item.get("patch"),
                ),
            )

        candidate = CommitCandidate(
            repo_full_name=repo_full_name,
            repo_clone_url=f"https://github.com/{repo_full_name}.git",
            commit_sha=payload["sha"],
            parent_sha=parents[0]["sha"],
            message=payload.get("commit", {}).get("message", "").strip(),
            html_url=payload.get("html_url", ""),
            author_name=(payload.get("commit", {}).get("author") or {}).get("name"),
            event_id=event_id,
            files=files,
        )
        if not candidate.files:
            raise ValueError("Commit had no changed files")
        return candidate

    @staticmethod
    def _extract_available_pages(link_header: str) -> list[int]:
        pages: set[int] = {1}
        for part in link_header.split(","):
            if "page=" not in part:
                continue
            fragment = part.split("page=", 1)[1]
            digits = []
            for char in fragment:
                if char.isdigit():
                    digits.append(char)
                else:
                    break
            if digits:
                pages.add(int("".join(digits)))
        return sorted(pages)

    def _get_json(self, path: str, return_response: bool = False, **params):
        used_token: str | None = None
        try:
            log.debug("GET %s params=%s", path, params or None)
            request_headers: dict[str, str] = {}
            if self._rotator:
                used_token = self._rotator.get_token()
                request_headers["Authorization"] = f"Bearer {used_token}"
            response = self._client.get(path, params=params or None, headers=request_headers or None)
            response.raise_for_status()
            payload = response.json()
            if return_response:
                return response, payload
            return payload
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 403:
                if self._rotator and used_token:
                    self._rotator.mark_rate_limited(used_token)
                if self._use_gh_cli:
                    log.debug("HTTP 403 for %s, falling back to gh api", path)
                    payload = self._get_json_via_gh(path, **params)
                    if return_response:
                        return None, payload
                    return payload
            raise

    @staticmethod
    def _get_json_via_gh(path: str, **params):
        endpoint = path.lstrip("/")
        if params:
            endpoint = f"{endpoint}?{urlencode(params)}"
        log.debug("Running gh api %s", endpoint)
        cmd = ["gh", "api", endpoint]
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or f"`gh api {path}` failed")
        return json.loads(result.stdout)
