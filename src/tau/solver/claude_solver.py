from solver_runner import solve_task
from tau.solver.base import Solver, SolveRequest, SolveResult


class ClaudeSolver(Solver):
    def solve(self, request: SolveRequest) -> SolveResult:
        return solve_task(
            repo_dir=request.repo_dir,
            task=request.task,
            model=self.model,
            timeout=self.timeout,
            config=self.config,
        )
