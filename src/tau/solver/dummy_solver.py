from __future__ import annotations

import time

from .base import Solver, SolveRequest, SolveResult
from .constants import COMPLETED_EXIT_REASON


class DummySolver(Solver):

    def solve(self, request: SolveRequest) -> SolveResult:
        started = time.monotonic()
        return SolveResult(
            success=True,
            elapsed_seconds=time.monotonic() - started,
            raw_output="",
            model=self.model,
            solution_diff="",
            exit_reason=COMPLETED_EXIT_REASON,
        )
