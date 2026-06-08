from docker_solver import solve_task_in_docker
from tau.solver.base import Solver, SolveRequest, SolveResult


class DockerSolver(Solver):
    def solve(self, request: SolveRequest) -> SolveResult:
        return solve_task_in_docker(
            repo_dir=request.repo_dir,
            task=request.task,
            task_name=request.task_name,
            solution_name=request.solution_name,
            repo_full_name=request.repo_full_name,
            commit_sha=request.commit_sha,
            run_label=request.run_label,
            model=self.model,
            timeout=self.timeout,
            config=self.config,
        )
