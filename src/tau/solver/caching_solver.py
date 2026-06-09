from __future__ import annotations

import hashlib
import json
from pathlib import Path

from openrouter_proxy import ProxyRequestRecord, SolveUsageSummary

from .base import Solver, SolveRequest, SolveResult
from .utils_docker import _apply_patch_to_repo


class CacheMissError(Exception):
    pass


def _usage_summary_from_dict(d: dict) -> SolveUsageSummary:
    requests = [ProxyRequestRecord(**r) for r in d.pop("requests", [])]
    return SolveUsageSummary(**d, requests=requests)


def _result_from_dict(d: dict, rollout_output: str | None = None) -> SolveResult:
    usage_raw = d.pop("usage_summary", None)
    usage = _usage_summary_from_dict(usage_raw) if usage_raw else None
    return SolveResult(**d, usage_summary=usage, rollout_output=rollout_output)


class CachingSolver(Solver):

    def __init__(
        self,
        model: str | None,
        timeout: int,
        config,
        inner: Solver | None,
        cache_dir: Path,
        solver_type: str = "",
        read: bool = True,
        write: bool = True,
    ) -> None:
        super().__init__(model, timeout, config)
        self._inner = inner
        self._cache_dir = cache_dir
        self._solver_type = solver_type
        self._read = read
        self._write = write

    def _agent_identity(self) -> str:
        if self.config.solver_agent_source is not None:
            return json.dumps(self.config.solver_agent_source.to_dict(), sort_keys=True)
        return self.config.solve_agent or ""

    def _cache_key(self, request: SolveRequest) -> str:
        payload = {
            "task_name": request.task_name or "",
            "solution_name": request.solution_name or "",
            "commit_sha": request.commit_sha or "",
            "model": self.model or "",
            "solver_type": self._solver_type,
            "agent_identity": self._agent_identity(),
        }
        serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(serialized.encode()).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.json"

    def _rollout_cache_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.jsonl"

    def load(self, request: SolveRequest) -> SolveResult | None:
        key = self._cache_key(request)
        path = self._cache_path(key)
        if not path.exists():
            return None
        rollout_path = self._rollout_cache_path(key)
        rollout_output = rollout_path.read_text(encoding="utf-8") if rollout_path.exists() else None
        return _result_from_dict(json.loads(path.read_text(encoding="utf-8")), rollout_output)

    def save(self, request: SolveRequest, result: SolveResult) -> None:
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(request)
        self._cache_path(key).write_text(json.dumps(result.to_dict()), encoding="utf-8")
        if result.rollout_output:
            self._rollout_cache_path(key).write_text(result.rollout_output, encoding="utf-8")

    def solve(self, request: SolveRequest) -> SolveResult:
        if self._read:
            cached = self.load(request)
            if cached is not None:
                if request.repo_dir.is_dir():
                    _apply_patch_to_repo(repo_dir=request.repo_dir, patch_text=cached.solution_diff)
                return cached
        if self._inner is None:
            key = self._cache_key(request)
            raise CacheMissError(f"cache miss for key {key} and no inner solver configured")
        result = self._inner.solve(request)
        if self._write:
            self.save(request, result)
        return result
