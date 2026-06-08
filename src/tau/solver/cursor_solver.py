from cursor_runner import solve_task_with_cursor_in_docker
from tau.solver.base import Solver, SolveRequest, SolveResult


class CursorSolver(Solver):
    def solve(self, request: SolveRequest) -> SolveResult:
        return solve_task_with_cursor_in_docker(
            repo_dir=request.repo_dir,
            task=request.task,
            model=self.model,
            timeout=self.timeout,
            config=self.config,
            run_label=request.run_label,
        )
