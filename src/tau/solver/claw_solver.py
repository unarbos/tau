from solver_runner import solve_task_claw
from tau.solver.base import Solver, SolveRequest, SolveResult


class ClawSolver(Solver):
    def solve(self, request: SolveRequest) -> SolveResult:
        return solve_task_claw(
            repo_dir=request.repo_dir,
            task=request.task,
            model=self.model,
            timeout=self.timeout,
            config=self.config,
        )
