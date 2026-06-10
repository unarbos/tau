import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from docker_solver import _harness_runner_script
from private_submission import agent_bundle_sha256, run_private_submission_checks

REPO_ROOT = Path(__file__).resolve().parent.parent
AGENT_DIR = REPO_ROOT / "agents" / "mini_swe_multifile"
HOTKEY = "5F3sa2TJAWMqDhXG6jhV4N8ko9SxwGy8TpaNS1repoTitleHkey"

# The validator-owned contract block of the public base agent (unarbos/ninja).
# The multi-file agent must keep every one of these lines verbatim so the
# submission scope guard sees them as unchanged context.
CONTRACT_BASE = '''\
import os
from typing import Any, Dict, Optional, Tuple

DEFAULT_MAX_STEPS = int(os.environ.get("AGENT_MAX_STEPS", "50"))
DEFAULT_COMMAND_TIMEOUT = int(os.environ.get("AGENT_COMMAND_TIMEOUT", "15"))

# VALIDATOR CONTRACT: These defaults are only fallbacks for local testing and
# validator wiring. During real validation the validator passes model, api_base,
# and api_key into solve(). Keep this code compatible with that path.
DEFAULT_MODEL = os.environ.get("AGENT_MODEL") or os.environ.get("NINJA_MODEL", "")
DEFAULT_API_BASE = (
    os.environ.get("AGENT_API_BASE")
    or os.environ.get("NINJA_INFERENCE_BASE_URL")
    or os.environ.get("OPENAI_BASE_URL", "")
)
DEFAULT_API_KEY = (
    os.environ.get("AGENT_API_KEY")
    or os.environ.get("NINJA_INFERENCE_API_KEY")
    or os.environ.get("OPENAI_API_KEY", "")
)
DEFAULT_MAX_TOKENS = int(os.environ.get("AGENT_MAX_TOKENS", "8192"))

MAX_OBSERVATION_CHARS = int(os.environ.get("AGENT_MAX_OBSERVATION_CHARS", "16000"))
MAX_TOTAL_LOG_CHARS = int(os.environ.get("AGENT_MAX_TOTAL_LOG_CHARS", "260000"))


def _normalize_api_base(api_base: str) -> str:
    base = api_base.rstrip("/")
    if base.endswith("/chat/completions"):
        return base[: -len("/chat/completions")]
    if base.endswith("/v1"):
        return base
    return base + "/v1"


def _resolve_inference_config(
    model: Optional[str],
    api_base: Optional[str],
    api_key: Optional[str],
) -> Tuple[str, str, str]:
    model_name = (model or DEFAULT_MODEL).strip()
    base = (api_base or DEFAULT_API_BASE).strip()
    key = (api_key if api_key is not None else DEFAULT_API_KEY).strip()

    if not model_name:
        raise ValueError("model is required; validators must pass the centrally managed model id")
    if not base:
        raise ValueError("api_base is required; validators must pass the managed inference proxy URL")
    if not key:
        raise ValueError("api_key is required; validators must pass the per-run proxy token")

    return model_name, _normalize_api_base(base), key


def build_initial_user_prompt(issue: str, repo_summary: str, preloaded_context: str = "") -> str:
    return issue


def solve(
    repo_path: str,
    issue: str,
    model: Optional[str] = None,
    api_base: Optional[str] = None,
    api_key: Optional[str] = None,
    max_steps: int = DEFAULT_MAX_STEPS,
    command_timeout: int = DEFAULT_COMMAND_TIMEOUT,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> Dict[str, Any]:
    return {"patch": "", "logs": "", "steps": 0, "cost": None, "success": True}
'''


def load_agent_files() -> dict:
    files = {}
    for path in sorted(AGENT_DIR.rglob("*.py")):
        relative = path.relative_to(AGENT_DIR)
        if any(part == "__pycache__" for part in relative.parts):
            continue
        files[relative.as_posix()] = path.read_text(encoding="utf-8")
    return files


class ScriptedModelServer:
    """Inference-free stand-in for the validator's OpenAI-compatible proxy."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.requests = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                outer.requests.append(payload)
                index = min(len(outer.requests) - 1, len(outer.replies) - 1)
                reply = {
                    "choices": [
                        {"message": {"role": "assistant", "content": outer.replies[index]}}
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 7},
                }
                data = json.dumps(reply).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, fmt, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}/v1"

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc_info):
        self.server.shutdown()
        self.server.server_close()
        return False


def make_task_repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir()
    (repo / "file.txt").write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=a@b.c", "-c", "user.name=test", "commit", "-qm", "init"],
        cwd=repo,
        check=True,
    )
    return repo


def load_agent_module():
    agent_path = AGENT_DIR / "agent.py"
    if str(AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(AGENT_DIR))
    spec = importlib.util.spec_from_file_location("mini_swe_multifile_agent", agent_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class MiniSweMultifileLegalityTest(unittest.TestCase):
    def test_agent_passes_private_submission_checks(self):
        files = load_agent_files()
        self.assertIn("agent.py", files)
        self.assertGreater(len(files), 1)
        result = run_private_submission_checks(
            hotkey=HOTKEY,
            base_agent_py=CONTRACT_BASE,
            submitted_files=files,
            openrouter_judge=lambda payload: {
                "verdict": "pass",
                "overall_score": 90,
                "real_edit_score": 90,
                "summary": "stub",
            },
        )
        self.assertEqual(result.checks["agent_smoke"].status, "passed", result.checks["agent_smoke"].findings)
        self.assertEqual(result.checks["scope_guard"].status, "passed", result.checks["scope_guard"].findings)
        self.assertTrue(result.accepted)
        self.assertEqual(result.agent_sha256, agent_bundle_sha256(files))


class MiniSweMultifileSolveTest(unittest.TestCase):
    def test_solve_edits_repo_and_submits(self):
        replies = [
            "Let me read the file first.\n\n```bash\ncat file.txt\n```",
            "Minimal edit now.\n\n```bash\nprintf 'after\\n' > file.txt\n```",
            "Submitting.\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
        ]
        module = load_agent_module()
        with ScriptedModelServer(replies) as server, tempfile.TemporaryDirectory() as tmp:
            repo = make_task_repo(Path(tmp))
            result = module.solve(
                repo_path=str(repo),
                issue="change file.txt content from before to after",
                model="test-model",
                api_base=server.base_url,
                api_key="proxy-token",
            )
        self.assertTrue(result["success"], result)
        self.assertEqual(result["steps"], 3)
        self.assertIn("-before", result["patch"])
        self.assertIn("+after", result["patch"])
        self.assertIn("Submitted", result["message"])
        system_text = server.requests[0]["messages"][0]["content"]
        self.assertIn("bash", system_text)
        task_text = server.requests[0]["messages"][1]["content"]
        self.assertIn("change file.txt content", task_text)
        self.assertIn("COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", task_text)

    def test_solve_recovers_from_format_error(self):
        replies = [
            "I will just describe my plan without any command.",
            "Sorry, here is the edit.\n\n```bash\nprintf 'after\\n' > file.txt\n```",
            "Submitting.\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
        ]
        module = load_agent_module()
        with ScriptedModelServer(replies) as server, tempfile.TemporaryDirectory() as tmp:
            repo = make_task_repo(Path(tmp))
            result = module.solve(
                repo_path=str(repo),
                issue="change file.txt content from before to after",
                model="test-model",
                api_base=server.base_url,
                api_key="proxy-token",
            )
        self.assertTrue(result["success"], result)
        self.assertIn("+after", result["patch"])
        self.assertIn("format retry", result["logs"])

    def test_solve_returns_disk_diff_when_model_is_unreachable(self):
        module = load_agent_module()
        with tempfile.TemporaryDirectory() as tmp:
            repo = make_task_repo(Path(tmp))
            (repo / "file.txt").write_text("edited-before-crash\n", encoding="utf-8")
            result = module.solve(
                repo_path=str(repo),
                issue="anything",
                model="test-model",
                api_base="http://127.0.0.1:9/v1",
                api_key="proxy-token",
            )
        self.assertIn("+edited-before-crash", result["patch"])
        self.assertTrue(result["success"])
        self.assertEqual(result["steps"], 0)


class MiniSweMultifileHarnessRunnerTest(unittest.TestCase):
    def test_validator_harness_runner_loads_and_runs_the_agent(self):
        replies = [
            "Minimal edit.\n\n```bash\nprintf 'after\\n' > file.txt\n```",
            "Submitting.\n\n```bash\necho COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\n```",
        ]
        with ScriptedModelServer(replies) as server, tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = make_task_repo(root)
            runner = root / "run_single_file_agent.py"
            runner.write_text(_harness_runner_script() + "\n", encoding="utf-8")
            prompt = root / "task.txt"
            prompt.write_text("change file.txt content from before to after\n", encoding="utf-8")
            env = os.environ.copy()
            env.update(
                {
                    "TAU_AGENT_FILE": str(AGENT_DIR / "agent.py"),
                    "TAU_REPO_DIR": str(repo),
                    "TAU_PROMPT_FILE": str(prompt),
                    "TAU_HARNESS_RUNNER": str(runner),
                    "AGENT_MODEL": "test-model",
                    "OPENAI_BASE_URL": server.base_url,
                    "OPENAI_API_KEY": "proxy-token",
                }
            )
            completed = subprocess.run(
                [sys.executable, str(runner)],
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
            self.assertTrue(payload["ok"], payload)
            result = payload["result"]
            self.assertTrue(result["success"])
            self.assertIn("+after", result["patch"])
            events_path = root / "tau_events.jsonl"
            self.assertTrue(events_path.is_file())
            event_types = {
                json.loads(line).get("type")
                for line in events_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            self.assertTrue(event_types)


if __name__ == "__main__":
    unittest.main()
