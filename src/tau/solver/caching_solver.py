from __future__ import annotations

import hashlib
import json
from pathlib import Path

from openrouter_proxy import ProxyRequestRecord, SolveUsageSummary

from .base import Solver, SolveRequest, SolveResult


class CacheMissError(Exception):
    pass


def _usage_summary_from_dict(d: dict) -> SolveUsageSummary:
    requests = [ProxyRequestRecord(**r) for r in d.pop("requests", [])]
    return SolveUsageSummary(**d, requests=requests)


def _result_from_dict(d: dict) -> SolveResult:
    usage_raw = d.pop("usage_summary", None)
    usage = _usage_summary_from_dict(usage_raw) if usage_raw else None
    return SolveResult(**d, usage_summary=usage)


class CachingSolver(Solver):

    def __init__(
        self,
        model: str | None,
        timeout: int,
        config,
        inner: Solver | None,
        cache_dir: Path,
        read: bool = True,
        write: bool = True,
    ) -> None:
        super().__init__(model, timeout, config)
        self._inner = inner
        self._cache_dir = cache_dir
        self._read = read
        self._write = write

    def _cache_key(self, request: SolveRequest) -> str:
        parts = "|".join([
            request.task_name or "",
            request.solution_name or "",
            request.commit_sha or "",
        ])
        return hashlib.sha256(parts.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def load(self, request: SolveRequest) -> SolveResult | None:
        path = self._cache_path(self._cache_key(request))
        if not path.exists():
            return None
        return _result_from_dict(json.loads(path.read_text(encoding="utf-8")))

    def save(self, request: SolveRequest, result: SolveResult) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(self._cache_key(request))
        path.write_text(json.dumps(result.to_dict()), encoding="utf-8")

    def solve(self, request: SolveRequest) -> SolveResult:
        if self._read:
            cached = self.load(request)
            if cached is not None:
                return cached
        if self._inner is None:
            key = self._cache_key(request)
            raise CacheMissError(f"cache miss for key {key} and no inner solver configured")
        result = self._inner.solve(request)
        if self._write:
            self.save(request, result)
        return result
